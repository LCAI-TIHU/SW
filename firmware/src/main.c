/* Copyright 2019 Inspur Corporation. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
    
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdarg.h>
#include "metal/cpu.h"
#include "metal/machine.h"
/* #include "dla_callback.h" */
#include "device_init.h"
#include "cpu_callback.h"
#include "dla_debug.h"
#include "aipu_timer.h"

extern struct aipu_timer aipu_timer0;

static struct device_dla_task dla_tsk = {
	.num_addresses = 0,
	//.dla_dev = NULL,
	.address_list = NULL};
static struct cpu_task cpu_tsk = {
	.num_addresses = 0,
	.cpu_dev = NULL,
	.address_list = NULL,
	// .cpu_parameters = NULL
	.cpu_task_pt = NULL};

static struct dla_device dla_dev = {
	.dla_irq = 2,
	.base = NULL,
	.task = &dla_tsk,
	.config_data = NULL,
	.event_notifier = 0,
	.engine_context = NULL};

struct cpu_device cpu_dev = {
	.cpu_irq = 1,
	.base = NULL,
	.task = &cpu_tsk,
	.config_data = NULL,
	.event_notifier = 0};

static struct pcie_device pcie_dev = {
	.pcie_irq = 16,
	.pcie_task_done = true};

int main()
{
	int32_t err = 0, flag = 0, task_num = 0;
	int32_t i = 0;
	int32_t address;
	err = cpu_init(&dla_dev, &pcie_dev);
	uint64_t t_start0, t_end0, t_start1, t_end1, t_start2, t_end2;
	uint64_t *cpu_address_list;
	debug_info("\n******************************************************************\nWaiting task!!!\n");
	while (1)
	{
		if (!pcie_dev.pcie_task_done)
		{
			/* debug_info("\n\tGet task and start !!!\n"); */
			/* aipu_timer_loadvalue_set(&aipu_timer0, 0xffffffff, 0xffffffff); */
			t_start0 = get_timer_us(&aipu_timer0);
			/* sifive_pl2cache0_flush((uintptr_t)DLA_TASK_ADDR); // */
			/* sifive_pl2cache0_flush((uintptr_t)DLA_BASE_ADDR); // */
			/* sifive_pl2cache0_flush((uintptr_t)COMPILER_BASE_ADDR); // */
 
			/* debug_info("\n\tTask flush Done !!!\n"); */
			device_init(&cpu_dev, &dla_dev);
			task_num = 0;
			struct device_task_tmp *dev_task_tmp = (struct device_task_tmp *)DLA_TASK_ADDR;
			while ((void *)dev_task_tmp != NULL)
			{
                /* debug_info("\n\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n\n"); */
				// debug_info("%d  :Task parameters address is 0x%08x.\n",task_num++, (uint64_t)dev_task_tmp);
			    if (dev_task_tmp->dev_type == 0) 
				{
					debug_info("enter Dla******** \n");
					dla_dev.task->num_addresses = dev_task_tmp->num_addresses;
					dla_dev.task->address_list = (uint64_t *)dev_task_tmp->address_list;
					device_debug(&cpu_dev, &dla_dev, 0);
					t_start2 = get_timer_us(&aipu_timer0);
					err = dla_task_submit(&dla_dev, dla_dev.task);
					t_end2 = get_timer_us(&aipu_timer0);
                    //debug_info("Dla used time = %ldus.\n", t_start2 - t_end2);
					if (err)
						break;
				}
				 else if (dev_task_tmp->dev_type == 1) 
				{
					cpu_dev.task->num_addresses = dev_task_tmp->num_addresses;
					cpu_dev.task->address_list = (uint64_t *)dev_task_tmp->address_list;
					cpu_address_list = cpu_dev.task->address_list;
					cpu_dev.task->cpu_task_pt = (struct cpu_task_package *)dev_task_tmp->task_pointer;
					// debug_info("cpu_dev.task->address_list is 0x%08x.\n",cpu_dev.task->address_list);
					// debug_info("cpu_dev.task->cpu_task_pt is 0x%08x.\n",cpu_dev.task->cpu_task_pt);
					// debug_info("&(cpu_dev.task->cpu_task_pt->next) is 0x%08x.\n", &(cpu_dev.task->cpu_task_pt->next));
					// debug_info("cpu_dev.task->cpu_task_pt->next is 0x%08x.\n", cpu_dev.task->cpu_task_pt->next);
                    uintptr_t flash_addr; 
					while ((cpu_dev.task->cpu_task_pt) != NULL)
					{
                        /* debug_info("\n\t-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^\n"); */
						flash_addr = cpu_dev.task->cpu_task_pt->cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.address; 
                        device_debug(&cpu_dev, &dla_dev, 1);
						cpu_param_debug(cpu_dev.task->cpu_task_pt->cpu_parameters);
						cpu_input_debug(cpu_dev.task->cpu_task_pt->cpu_parameters);

                        t_start1 = get_timer_us(&aipu_timer0);
						err = cpu_task_submit(&cpu_dev, cpu_dev.task);
                        t_end1 = get_timer_us(&aipu_timer0);
						debug_info("Optype = %d, used time = %ldus.\n", 
                                   cpu_dev.task->cpu_task_pt->cpu_parameters.cpu_operation.common_only_op.common.op_type, 
                                   t_start1 - t_end1);
						cpu_dev.task->cpu_task_pt = (struct cpu_task_package *)cpu_dev.task->cpu_task_pt->next;
					}
					// debug_info("\n-----------------------------------------\n");
					// set flush delay
                    /* sifive_pl2cache0_flush(flash_addr); */
					/* set_timer_udelay(&aipu_timer0, 1000); */
					if (err)
						break;
				}
				dev_task_tmp = (struct device_task_tmp *)dev_task_tmp->next;
			}

			pcie_dev.pcie_task_done = true;
			pcie_intr_set();

			t_end0 = get_timer_us(&aipu_timer0);

			if (err)
			{
				if (err == TIMEOUT)
				{
					debug_trace("Time out error\n");
					debug_info("Time out error\n");
					cpu_init(&dla_dev, &pcie_dev);
				}
				else if (err == NO_CPU_TASK)
				{
					debug_trace("Task done, and no cpu task\n");
					debug_info("Task done, and no cpu task\n");
				}
				else
				{
					debug_trace("Task failed,err code is %d\n", err);
					debug_info("Task failed, err code is %d\n", err);
				}
			}
			else
			{
				debug_info("Task_done, Task use %lu us\n", t_start0 - t_end0);
			}
		}
		else
		{
			aipu_timer_loadvalue_set(&aipu_timer0, 0xffffffff, 0xffffffff);
			set_timer_udelay(&aipu_timer0, 1); // make the if condition happen, don't know why
		}
	}
}
