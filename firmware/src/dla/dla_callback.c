/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License version 2
 * as published by the Free Software Foundation; or, when distributed
 * separately from the Linux kernel or incorporated into other
 * software packages, subject to the following license:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdarg.h>
#include <string.h>
#include "device_init.h"
#include "dla_debug.h"
#include "dla_callback.h"
#include "aipu_io.h"
#include "metal/interrupt.h"
//#include <metal/drivers/sifive_pl2cache0.h>
#include "aipu_timer.h"
#include "aipu_scu.h"
//#include "opendla_2048_full.h"//glb test

extern struct aipu_timer aipu_timer0;
extern struct metal_interrupt *global_intr;

void *dla_memset(void *src, int ch, uint64_t len)
{
	int32_t i = 0;
	#pragma clang loop vectorize(enable)
	for(i = 0; i < len; i++){
		*(char *)(src+i) = ch;
	}
	return 0;
}

void *dla_memcpy(void *dst, const void *src, uint64_t len)
{
	int32_t i = 0;
	#pragma clang loop vectorize(enable)
	for(i = 0; i < len; i++){
		*(char *)(dst+i) = *(char *)(src+i);
	}
	return 0;
}

int64_t dla_get_time_us(void)
{
	return 0;
}

void wait_for_completion(int32_t *event_notifier, int32_t *err){
	debug_trace("Enter %s, event_notifier is %d\n", __func__, *event_notifier);
	uint64_t t1 = get_timer_us(&aipu_timer0);
	uint64_t t2 = 0;
	while(!(*event_notifier)){
		//add timeout?
		t2 = get_timer_us(&aipu_timer0);
		if((t1 - t2) > 4000000){ //100ms timeout

			time_out_debug(&t1, &t2);

			*err = TIMEOUT;
			//mac reset
			mac_soft_reset();
			break;
		}
	}
	(*event_notifier) = 0;
	debug_trace("Exit %s\n", __func__);
}

void dla_reg_write(void *driver_context, uint32_t addr, uint32_t reg)
{
	struct dla_device *dla_dev =
			(struct dla_device *)driver_context;

	if (!dla_dev)
		return;
	REG_RW(dla_dev->base, addr) = reg;
	debug_trace("Write Reg(0x%08x): 0x%08x to the address 0x%08x, bus address is %08x\n", REG_RW(dla_dev->base, addr), reg,  addr, ((unsigned long)dla_dev->base + addr));
}

uint32_t dla_reg_read(void *driver_context, uint32_t addr)
{
	struct dla_device *dla_dev =
			(struct dla_device *)driver_context;
	uint32_t ret = 0;
	if (!dla_dev)
		return 0;

	ret = REG_RW(dla_dev->base, addr);
//	debug_trace("Read Reg: 0x%08x from the address 0x%08x, bus address is %08x\n", ret,  addr, ((unsigned long)dla_dev->base + addr));
	return ret ;
}

int32_t dla_read_cpu_address(void *driver_context, void *task_data,
						int16_t index, void *dst, uint32_t destination)
{
	uint64_t *temp = (uint64_t *)dst;
	struct device_dla_task *task = (struct device_dla_task *)task_data;
	uint64_t *addr_list = (uint64_t *)task->address_list;
    if (index == -1 || index > task->num_addresses){
		return -EINVAL;
        debug_trace("Error in %s: index = %d, num_addresses = %d.\n", __func__, index, task->num_addresses);
    }
	*temp = addr_list[index];
    debug_trace("Exit %s.\n", __func__);
	return 0;
}

int32_t dla_data_write(void *driver_context, void *task_data,
				void *src, uint64_t dst,
				uint32_t size, uint64_t offset)
{
//	debug_trace("Write data from address: 0x%x, to address:0x%x, len:%d\n",src, dst, size); 
	memcpy((void *)((char *)dst + offset), src, size);
//	int32_t i = 0;
//	for(i = 0; i < size; i++){
//		*(volatile char *)(dst+i) = *(volatile char *)(src+offset+i);
//	}
	return 0;

}

int32_t dla_data_read(void *driver_context, void *task_data,
				uint64_t src, void *dst,
				uint32_t size, uint64_t offset)
{

	debug_trace("Read data from address: 0x%x, to address:0x%x, len:%d\n",src, dst, size);
//	int32_t i = 0;
//	for(i = 0; i < size; i++){
//		*(volatile char *)(dst+i) = *(volatile char *)(src+offset+i);
//		debug_trace("%#x,", *(volatile char *)(src+offset+i));
//	}
	memcpy(dst, (void *)((char *)src+offset), size);//nostdlib
	return 0;
}

int32_t dla_task_submit(struct dla_device *dla_dev, struct device_dla_task *task)
{
	int32_t err = 0;
	uint32_t task_complete = 0;

	dla_dev->task = task;

	err = dla_execute_task(dla_dev->engine_context, (void *)task, dla_dev->config_data);
	if (err) {
		debug_trace("Task execution failed\n");
		goto exit;
	}

	debug_trace("Wait for task complete\n");

	while (1) {
		wait_for_completion(&dla_dev->event_notifier, &err);// close interrupt when simulate
		if(err)
			goto exit;

		metal_interrupt_disable(global_intr, dla_dev->dla_irq);
		err = dla_process_events(dla_dev->engine_context, &task_complete);
		metal_interrupt_enable(global_intr, dla_dev->dla_irq);


		if (err || task_complete)
			break;
	}
exit:
	dla_clear_task(dla_dev->engine_context);
	mac_soft_reset();
	return err;
}
