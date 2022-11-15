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

#include <aipu_uart.h>
#include "device_init.h"
#include "dla_callback.h" 
#include "dla_sched.h" 
#include <stdlib.h>
#include "watchdog.h"
#include "metal/cpu.h"
#include "metal/machine.h"
#include "dla_debug.h"
#include "metal/drivers/riscv_cpu.h"
#include "aipu_io.h"
#include "aipu_timer.h"
#include "aipu_uart.h"

struct metal_cpu *cpu;
struct metal_interrupt *cpu_intr, *tmr_intr;
struct metal_interrupt *global_intr;
int tmr_id;
extern struct cpu_device cpu_dev;
struct metal_clock sys_clk = {
    .vtable = &__metal_driver_vtable_aipu_clock.clock,
};

struct aipu_timer aipu_timer0 = {
    .vtable = &aipu_timer_vtable,
    .timer_base = TIMER0_BASE,
};

struct aipu_uart aipu_uart0 = {
    .vtable = &aipu_uart_vtable,
    .uart_base = UART0_BASE,
};
/* pay attention to the defferents between interrupts of cup, timer and interrupts of plic, global.
 * pay attention to the interrupt controller differents between plic interrupt and globle interrupt
 */

// warning, when interrupt happens, the irq and the device of the interrupt must be clear.
void dla_engine_isr(int id, void *data) 
{ 
     //	debug_info("Enter %s\n", __func__); 
     struct dla_device *dla_dev_isr = (struct dla_device *)data; 
     metal_interrupt_disable(global_intr, dla_dev_isr->dla_irq); 
     if (!dla_dev_isr) 
         debug_info("ERROR:Can not get nvdla device"); 

     dla_isr_handler(dla_dev_isr->engine_context); 
     dla_dev_isr->event_notifier++; 
     metal_interrupt_enable(global_intr, dla_dev_isr->dla_irq); 
     // should not unmask interrupt source here 
} 

void pcie_isr(int id, void *data)
{
    // mask all interrupt source
    clean_sw_intr0();
   // debug_info("enter:%s\n", __func__);
    struct pcie_device *pcie_dev_isr = (struct pcie_device *)data;
    metal_interrupt_disable(global_intr, pcie_dev_isr->pcie_irq);
    pcie_dev_isr->pcie_task_done = false;
    // unmask interrupt source
    metal_interrupt_enable(global_intr, pcie_dev_isr->pcie_irq);
   // debug_info("Exit:%s, pcie_task_done = %d\n", __func__, pcie_dev_isr->pcie_task_done);
}

void timer_isr(int id, void *data)
{
  //  debug_info("ENTER %s\n",__func__);
  //  debug_info("Feed Watch Dog here!\n");
    // wdt_feed(SYSTEM_CLK);
    metal_cpu_set_mtimecmp(cpu, metal_cpu_get_mtime(cpu) + 10 * SYSTEM_CLK);
}

int32_t cpu_init(struct dla_device *dla_dev, struct pcie_device *pcie_dev)
{
    struct metal_interrupt *plic;
    int rc = 0;

    metal_clock_set_rate_hz(&sys_clk, SYSTEM_CLK);

    aipu_uart_init(&aipu_uart0, &sys_clk, UART0_BAUD_RATE); // don't move
 
    aipu_timer_init(&aipu_timer0, &sys_clk);
  
    aipu_timer_1_prescale_set(&aipu_timer0, 1);
    aipu_timer_2_prescale_set(&aipu_timer0, 1);
    //debug_info(" timer0 prescales are %u, %u\n", aipu_timer_1_prescale_get(&aipu_timer0),
              //  aipu_timer_2_prescale_get(&aipu_timer0));
    /* aipu_timer_1_enable(&aipu_timer0); */
    aipu_timer_loadvalue_set(&aipu_timer0, 0xffffffff, 0xffffffff);
    /*
        rc = wdt_init(SYSTEM_CLK); //watchdog
        if(rc)
            debug_info("Watchdog init faild;\n");
    */
  //  debug_info("system cloclk is %ld\n", metal_clock_get_rate_hz(&sys_clk));
    // Lets get the CPU and and its interrupt
    cpu = metal_cpu_get(metal_cpu_get_current_hartid());
    if (cpu == NULL)
    {
        debug_info("CPU null.\n");
        return 2;
    }
    cpu_intr = metal_cpu_interrupt_controller(cpu);
    if (cpu_intr == NULL)
    {
        debug_info("CPU interrupt controller is null.\n");
        return 3;
    }
    metal_interrupt_init(cpu_intr);

    // Setup Timer interrupt for feeding watchdog
    tmr_intr = metal_cpu_timer_interrupt_controller(cpu);
    if (tmr_intr == NULL)
    {
        debug_info("Abort. TIMER interrupt controller is  null.\n");
        return 4;
    }
    metal_interrupt_init(tmr_intr);
    tmr_id = metal_cpu_timer_get_interrupt_id(cpu);
    rc = metal_interrupt_register_handler(tmr_intr, tmr_id, timer_isr, cpu);
    if (rc < 0)
    {
        debug_info("Failed. TIMER interrupt handler registration failed\n");
        return (rc * -1);
    }
    // Set timeout of 1s, and enable timer interrupt
    metal_cpu_set_mtimecmp(cpu, metal_cpu_get_mtime(cpu) + 1 * SYSTEM_CLK);
    // metal_interrupt_enable(tmr_intr, tmr_id); // does not need watch dog 
    metal_interrupt_disable(tmr_intr, tmr_id); // does not need watch dog

    // Check we this target has a plic. If not gracefull exit
    plic = metal_interrupt_get_controller(METAL_PLIC_CONTROLLER, 0);
    if (plic == NULL)
    {
        debug_info("Exit. This example need a plic interrupt controller for MAC and PCIE.\n");
        return 0;
    }

    // init global interrupt
    global_intr = (struct metal_interrupt *)&__metal_dt_global_external_interrupts;
    metal_interrupt_init(global_intr);
    // init uart0

    // MAC interrupt and PCIE interrupt share the global interrupt
     rc = metal_interrupt_register_handler(global_intr, dla_dev->dla_irq, dla_engine_isr, dla_dev); 
     if (rc < 0) 
     { 
         debug_info("MAC interrupt handler registration failed\n"); 
         return (rc * -1); 
     } 

    // PCIE interrupt
    rc = metal_interrupt_register_handler(global_intr, pcie_dev->pcie_irq, pcie_isr, pcie_dev);
    if (rc < 0)
    {
        debug_info("PCIE interrupt handler registration failed\n");
        return (rc * -1);
    }
    // Lets enable MAC and PCIE interrupts
    metal_interrupt_set_threshold(global_intr, 0);

    metal_interrupt_set_priority(global_intr, dla_dev->dla_irq, 1);
    metal_interrupt_set_priority(global_intr, pcie_dev->pcie_irq, 1);
    //  write32(1,PCIE_INTR_PRIORITY);
    // write32(1,MAC_INTR_PRIORITY);

    if (metal_interrupt_enable(global_intr, dla_dev->dla_irq) == -1)
    {
        debug_info("MAC interrupt enable failed\n");
        return 5;
    }
    if (metal_interrupt_enable(global_intr, pcie_dev->pcie_irq) == -1)
    {
        debug_info("PCIE interrupt enable failed\n");
        return 5;
    }

    dla_register_driver((void **)(&dla_dev->engine_context), (void *)dla_dev); 

    // Lastly CPU interrupt

    if (metal_interrupt_enable(cpu_intr, 0) == -1)
    {
        debug_info("CPU interrupt enable failed\n");
        return 6;
    }
    debug_info("exit:%s\n", __func__);
    return rc;
}

static struct dla_config dla_config_data = { 
     .atom_size = 32, 
     .bdma_enable = true, 
     .rubik_enable = true, 
     .weight_compress_support = true}; 

static struct cpu_config cpu_config_data = {
    .atom_size = 32,
    .vlen = 256};

int32_t device_init(struct cpu_device *cpu_dev, struct dla_device *dla_dev)
{
    struct dla_device *m_dla_dev = dla_dev;
    struct cpu_device *m_cpu_dev = cpu_dev;
    /*read lmu ram at here */
    int32_t i;
    m_cpu_dev->base = 0x00000000;
    m_cpu_dev->config_data = &cpu_config_data;
    m_cpu_dev->cpu_irq = 1;
    m_cpu_dev->event_notifier = 0;
    m_cpu_dev->task->cpu_dev = (struct cpu_device *)&cpu_dev;


     m_dla_dev->base = (uint64_t *)0x40400000; 
     m_dla_dev->config_data = &dla_config_data; 
     m_dla_dev->event_notifier = 0; 
     m_dla_dev->dla_irq = 2; 

    return 0;
}

#ifdef DEV_DEBUG


int32_t cpu_input_debug(struct cpu_param cpu_parameters)
{
    debug_info("\t\n*********Enter %s\n", __func__);
    uint32_t i, h, w, c, sx, sy, surf, offset, size;
    uint64_t pSrc;
    uint32_t dat_h, dat_w, dat_c, data_type;
    int32_t line_stride_in, surf_stride_in;
    /* cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.address = (int8_t *)cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.index]; */
    cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.address = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.index];
    int32_t input_num = cpu_parameters.cpu_operation.common_only_op.common.input_num;
    for (i = 0; i < input_num; i++)
    {
        if (input_num == 1)
        {
            offset = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.index;
            pSrc = cpu_dev.task->address_list[offset];
            dat_h = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.height;
            dat_w = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.width;
            dat_c = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.channel;
            line_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.line_stride;
            surf_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.surf_stride;
            data_type = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.datatype;
            size = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.size;
        }
        else
        {
            if (cpu_parameters.cpu_operation.common_only_op.common.op_type == CONCAT)
            {
                offset = cpu_parameters.cpu_operation.concat_op.src_data[i].index;
                pSrc = cpu_dev.task->address_list[offset];
                dat_h = cpu_parameters.cpu_operation.concat_op.src_data[i].height;
                dat_w = cpu_parameters.cpu_operation.concat_op.src_data[i].width;
                dat_c = cpu_parameters.cpu_operation.concat_op.src_data[i].channel;
                line_stride_in = cpu_parameters.cpu_operation.concat_op.src_data[i].line_stride;
                surf_stride_in = cpu_parameters.cpu_operation.concat_op.src_data[i].surf_stride;
                data_type = cpu_parameters.cpu_operation.concat_op.src_data[i].datatype;
                size = cpu_parameters.cpu_operation.concat_op.src_data[i].size;
            }
            else
            {
                if (i == 0)
                {
                    offset = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.index;
                    pSrc = cpu_dev.task->address_list[offset];
                    dat_h = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.height;
                    dat_w = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.width;
                    dat_c = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.channel;
                    line_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.line_stride;
                    surf_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.surf_stride;
                    data_type = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.datatype;
                    size = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.size;
                }
                else
                {
                    if (cpu_parameters.cpu_operation.common_only_op.common.op_type == TAKE)
                    {
                        debug_info("Take parameters:\n");
                        offset = cpu_parameters.cpu_operation.take_op.indices.index;
                        pSrc = cpu_dev.task->address_list[offset];
                        dat_h = cpu_parameters.cpu_operation.take_op.indices.height;
                        dat_w = cpu_parameters.cpu_operation.take_op.indices.width;
                        dat_c = cpu_parameters.cpu_operation.take_op.indices.channel;
                        line_stride_in = cpu_parameters.cpu_operation.take_op.indices.line_stride;
                        surf_stride_in = cpu_parameters.cpu_operation.take_op.indices.surf_stride;
                        data_type = cpu_parameters.cpu_operation.take_op.indices.datatype;
                        size = cpu_parameters.cpu_operation.take_op.indices.size;
                    }
                    else
                    {
                        offset = cpu_parameters.cpu_operation.with_weight_op.weight.index;
                        pSrc = cpu_dev.task->address_list[offset];
                        dat_h = cpu_parameters.cpu_operation.with_weight_op.weight.height;
                        dat_w = cpu_parameters.cpu_operation.with_weight_op.weight.width;
                        dat_c = cpu_parameters.cpu_operation.with_weight_op.weight.channel;
                        line_stride_in = cpu_parameters.cpu_operation.with_weight_op.weight.line_stride;
                        surf_stride_in = cpu_parameters.cpu_operation.with_weight_op.weight.surf_stride;
                        data_type = cpu_parameters.cpu_operation.with_weight_op.weight.datatype;
                        size = cpu_parameters.cpu_operation.with_weight_op.weight.size;
                    }
                }
            }
        }
        debug_info("src_data: %d/%d\n"
                   "\t data_type:%u;\n"
                   "\t batchsize: 1;\n"
                   "\t height : %u\n"
                   "\t width : %u\n"
                   "\t channel: %u\n"
                   "\t line_stride : %d\n"
                   "\t surf_stride: %d\n"
                   "\t size: %u\n"
                   "\t offset: %u\n"
                   "\t address: 0x%08x\n",
                   i,input_num,
                   data_type,
                   dat_h,
                   dat_w,
                   dat_c,
                   line_stride_in,
                   surf_stride_in,
                   size,
                   offset,
                   pSrc);

#ifdef INPUT_DEBUG
        // debug_info("INPUT DATA %d:\n", i);
        if (line_stride_in > 0 || surf_stride_in > 0)
        {
            debug_info("\ninput:surf-hwc\n");
            for (surf = 0; surf < ((dat_c - 1) / C_ATM + 1); surf++)
            {
                if (surf > DEBUG_SURF) break;
                debug_info("\nsurf %d\n",surf);
                for (h = 0; h < dat_h; h++)
                {
                    if (h > DEBUG_H) break;
                    debug_info("\nh %d:\n",h);
                    for (w = 0; w < dat_w; w++)
                    {
                        if (w > DEBUG_W) break;
                        debug_info("\n\tw %d:\n", w);
                        for (c = 0; c < C_ATM; c++)
                        {
                            if (c > DEBUG_C) break;
                            int8_t *input_addr = (int8_t *)(pSrc + c + w * C_ATM + h * line_stride_in + surf * surf_stride_in);
                            int8_t input_data = *input_addr;
                            debug_info("%d ", input_data);
                        }
                    }
                }
            }
        }
        else
        {
            debug_info("\ninput:hwc\n");
            for (h = 0; h < dat_h; h++)
            {
                if (h > DEBUG_H) break;
                debug_info("\nh %d:\n", h);
                for (w = 0; w < dat_w; w++)
                {
                    if (w > DEBUG_W) break;
                    debug_info("\n\tw %d:\n", w);
                    for (c = 0; c < dat_c; c++)
                    {
                        if (c > DEBUG_C) break;
                        if (data_type == RINT8)
                        {
                            int8_t *addr = (int8_t *)(pSrc + c + w * dat_c + h * dat_w * dat_c);
                            debug_info("%d ", *addr);
                        }
                        else if (data_type == RFLOAT)
                        {
                            float_t fl_dat = *(volatile float_t *)(pSrc + (c + w * dat_c + h * dat_w * dat_c) * sizeof(float_t));
                            print_float(fl_dat);
                        }
                        else if (data_type == RINT)
                        {
                            int32_t *addr = (int32_t *)(pSrc + (c + w * dat_c + h * dat_w * dat_c) * sizeof(int32_t));
                            debug_info("%d ", *addr);
                        }
                    }
                }
            }
        }
        debug_info("\n");
#endif
    }
    return 0;
}
int32_t cpu_output_debug(struct cpu_param cpu_parameters)
{
    uint32_t i, h, w, c, sx, sy, surf, offset;
    debug_info("*********Enter %s\n", __func__);
    uint32_t output_num = cpu_parameters.cpu_operation.common_only_op.common.output_num;
    uint32_t op_type = cpu_parameters.cpu_operation.common_only_op.common.op_type;
    uint64_t pDst;
    uint32_t dat_h, dat_w, dat_c;
    int32_t line_stride_out, surf_stride_out;
    if (output_num == 1) {
        cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.address = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.index];
        pDst = (uint64_t)(cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.address);
        dat_h = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.height;
        dat_w = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.width;
        dat_c = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.channel;
        line_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.line_stride;
        surf_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.surf_stride;
        debug_info("dst_data: \n"
                   "\t data_type:%u;\n"
                   "\t batchsize: %u;\n"
                   "\t height : %u\n"
                   "\t width : %u\n"
                   "\t channel: %u\n"
                   "\t line_stride : %d\n"
                   "\t surf_stride: %d\n"
                   "\t size: %u\n"
                   "\t offset: %u\n"
                   "\t address: 0x%08x\n",
                   cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.datatype,
                   cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.batchsize,
                   cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.height,
                   cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.width,
                   cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.channel,
                   cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.line_stride,
                   cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.surf_stride,
                   cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.size,
                   cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.index,
                   cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.address);
    } else if (output_num > 1) {
        if ( op_type == SPLIT ) {
            for (i = 0; i < output_num; i++) {
                pDst = (uint64_t)(cpu_dev.task->address_list[cpu_parameters.cpu_operation.split_op.dst_data[i].index]);
                dat_h = cpu_parameters.cpu_operation.split_op.dst_data[i].height;
                dat_w = cpu_parameters.cpu_operation.split_op.dst_data[i].width;
                dat_c = cpu_parameters.cpu_operation.split_op.dst_data[i].channel;
                line_stride_out = cpu_parameters.cpu_operation.split_op.dst_data[i].line_stride;
                surf_stride_out = cpu_parameters.cpu_operation.split_op.dst_data[i].surf_stride;
                debug_info("dst_data: \n"
                           "\t data_type:%u;\n"
                           "\t batchsize: %u;\n"
                           "\t height : %u\n"
                           "\t width : %u\n"
                           "\t channel: %u\n"
                           "\t line_stride : %d\n"
                           "\t surf_stride: %d\n"
                           "\t size: %u\n"
                           "\t offset: %u\n"
                           "\t address: 0x%08x\n",
                           cpu_parameters.cpu_operation.split_op.dst_data[i].datatype,
                           cpu_parameters.cpu_operation.split_op.dst_data[i].batchsize,
                           dat_h,
                           dat_w,
                           dat_c,
                           line_stride_out,
                           surf_stride_out,
                           cpu_parameters.cpu_operation.split_op.dst_data[i].size,
                           cpu_parameters.cpu_operation.split_op.dst_data[i].index,
                           pDst
                           );
            }

        } else {
            debug_info("Warning, need to support new op_type.\n");
        }
    }
#ifdef OUTPUT_DEBUG
    for (i = 0; i < output_num; i++) {
        if (line_stride_out > 0 || surf_stride_out > 0)
        {
            debug_info("\noutput:surf_hwc\n");
            for (surf = 0; surf < ((dat_c - 1) / C_ATM + 1); surf++)
            {
                if (surf > DEBUG_SURF) break;
                debug_info("\nsurf %d\n",surf);
                for (h = 0; h < dat_h; h++)
                {
                    if (h > DEBUG_H) break;
                    debug_info("\nh %d:\n",h);
                    for (w = 0; w < dat_w; w++)
                    {
                        if (w > DEBUG_W) break;
                        debug_info("\n\tw %d:\n", w);
                        for (c = 0; c < C_ATM; c++)
                        {
                            if (c > DEBUG_C) break;
                            if (cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.datatype == RINT8)
                            {
                                int8_t *output_addr = (int8_t *)(pDst + c + w * C_ATM + h * line_stride_out + surf * surf_stride_out);
                                int8_t output_data = *output_addr;
                                debug_info("%d ", output_data);
                            }
                            else
                            {
                                debug_info("\n data type error\n");
                                return -1;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            debug_info("\noutput:hwc\n");
            for (h = 0; h < dat_h; h++)
            {
                if (h > DEBUG_H) break;
                debug_info("\nh %d:\n", h);
                for (w = 0; w < dat_w; w++)
                {
                    if (w > DEBUG_W) break;
                    debug_info("\n\tw %d:\n", w);
                    for (c = 0; c < dat_c; c++)
                    {
                        if (c > DEBUG_C) break;
                        if (cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.datatype == RINT8)
                        {
                            int8_t *output_addr = (int8_t *)(pDst + c + w * dat_c + h * dat_w * dat_c);
                            int8_t output_data = *output_addr;
                            debug_info("%d ", output_data);
                        }
                        else if (cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.datatype == RFLOAT)
                        {
                            float_t output_data = *(volatile float_t *)(pDst + (c + w * dat_c + h * dat_w * dat_c) * sizeof(float_t));
                            print_float(output_data);
                        }
                        else if (cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.datatype == RINT)
                        {
                            int32_t *output_addr = (int32_t *)(pDst + (c + w * dat_c + h * dat_w * dat_c) * sizeof(int32_t));
                            int32_t output_data = *output_addr;
                            debug_info("%d ", output_data);
                        }
                    }
                }
            }
        }
    }
    debug_info("\n");
#endif
    return 0;
}

void device_debug(struct cpu_device *cpu_dev, struct dla_device *dla_dev, uint32_t dev_type)
{
    uint32_t i;
    uint64_t *addr_list;
    if (dev_type)
    {
        debug_info("*********Riscv TASK:\n"
                   "\t num_addresses = 0x%x;\n",
                   cpu_dev->task->num_addresses);
        // debug_info("\n");
        debug_info("\t address_list_addr = 0x%x;\n",
                   cpu_dev->task->address_list);
        // debug_info("\n");
        debug_info("\t riscv_task_pointer = 0x%x;\n",
                   cpu_dev->task->cpu_task_pt);
        // debug_info("\n");
        // debug_info("\n");
        addr_list = (uint64_t *)cpu_dev->task->address_list;
        debug_info("Riscv address list is :\n[");
        for (i = 0; i < cpu_dev->task->num_addresses; i++)
        {

            debug_info("0x%08x, ", addr_list[i]);
        }
        debug_info("]\n");
    }
    else
    {
        debug_info("*********DLA TASK:\n"
                   "\t num_address = %d;\n",
                   dla_dev->task->num_addresses);

        addr_list = (uint64_t *)dla_dev->task->address_list;
        debug_info("\t address_list = %#x\n", addr_list);
        debug_info("DLA address list is :\n[");
        for (i = 0; i < dla_dev->task->num_addresses; i++)
        {

            debug_info("0x%08x, ", addr_list[i]);
        }
        debug_info("]\n");
    }
}

void cpu_param_debug(struct cpu_param cpu_parameters)
{

    uint64_t pSrc;
    uint32_t dat_h, dat_w, dat_c, data_type;
    int32_t line_stride_in, surf_stride_in;
    int32_t i;

    debug_info("*********Enter %s\n", __func__);
    // softmax, exp, sigmoid, reshape, upsample, sqrt, erf, tanh, relu
    debug_info("\n  common:\n"
               "\t op_type:%u;\n"
               "\t input_num: %u;\n"
               "\t output_num : %u\n",
               cpu_parameters.cpu_operation.common_only_op.common.op_type,
               cpu_parameters.cpu_operation.common_only_op.common.input_num,
               cpu_parameters.cpu_operation.common_only_op.common.output_num);
    debug_info("\tinput scale factor:");
    print_float(cpu_parameters.cpu_operation.common_only_op.common.input_scale_factor[0]);
    print_float(cpu_parameters.cpu_operation.common_only_op.common.input_scale_factor[1]);
    print_float(cpu_parameters.cpu_operation.common_only_op.common.input_scale_factor[2]);
    print_float(cpu_parameters.cpu_operation.common_only_op.common.input_scale_factor[3]);
    debug_info("\n\toutput scale factor:");
    print_float(cpu_parameters.cpu_operation.common_only_op.common.output_scale_factor[0]);
    print_float(cpu_parameters.cpu_operation.common_only_op.common.output_scale_factor[1]);
    print_float(cpu_parameters.cpu_operation.common_only_op.common.output_scale_factor[2]);
    print_float(cpu_parameters.cpu_operation.common_only_op.common.output_scale_factor[3]);
    debug_info("\n\t");
    // add, less, batch_matmul, power, divide, max, substract, dense
    if ((cpu_parameters.cpu_operation.with_weight_op.common.op_type == ADD) ||
        (cpu_parameters.cpu_operation.with_weight_op.common.op_type == LESS) ||
        (cpu_parameters.cpu_operation.with_weight_op.common.op_type == BATCH_MATMUL) ||
        (cpu_parameters.cpu_operation.with_weight_op.common.op_type == POWER) ||
        (cpu_parameters.cpu_operation.with_weight_op.common.op_type == MULTIPLY) ||
        (cpu_parameters.cpu_operation.with_weight_op.common.op_type == DIVIDE) ||
        (cpu_parameters.cpu_operation.with_weight_op.common.op_type == MAX) ||
        (cpu_parameters.cpu_operation.with_weight_op.common.op_type == SUBTRACT) ||
        (cpu_parameters.cpu_operation.with_weight_op.common.op_type == DENSE))
    {

        debug_info("\n\t weight_ioscale :");
        print_float(cpu_parameters.cpu_operation.with_weight_op.weight_ioscale);
        debug_info("\n\t sub_op_type : %u\n", cpu_parameters.cpu_operation.with_weight_op.sub_op_type); // 1 add/multiply with 1x1xc , 0 add/multiply with a const

        debug_info("weight: \n"
                   "\t data_type:%u;\n"
                   "\t batchsize: %u;\n"
                   "\t height : %u\n"
                   "\t width : %u\n"
                   "\t channel: %u\n"
                   "\t line_stride : %d\n"
                   "\t surf_stride: %d\n"
                   "\t size: %u\n"
                   "\t offset: %u\n"
                   "\t address: 0x%08x\n",
                   cpu_parameters.cpu_operation.with_weight_op.weight.datatype,
                   cpu_parameters.cpu_operation.with_weight_op.weight.batchsize,
                   cpu_parameters.cpu_operation.with_weight_op.weight.height,
                   cpu_parameters.cpu_operation.with_weight_op.weight.width,
                   cpu_parameters.cpu_operation.with_weight_op.weight.channel,
                   cpu_parameters.cpu_operation.with_weight_op.weight.line_stride,
                   cpu_parameters.cpu_operation.with_weight_op.weight.surf_stride,
                   cpu_parameters.cpu_operation.with_weight_op.weight.size,
                   cpu_parameters.cpu_operation.with_weight_op.weight.index,
                   cpu_parameters.cpu_operation.with_weight_op.weight.address);
    }

    // tanspose, squeeze
    if ((cpu_parameters.cpu_operation.common_only_op.common.op_type == TRANSPOSE) ||
        (cpu_parameters.cpu_operation.common_only_op.common.op_type == SQUEEZE))
    {

        for (i = 0; i < 4; i++)
        {
            debug_info("\t\n  axis[%d] : %d\n", i, cpu_parameters.cpu_operation.transform_op.axis[i]);
        }
    }

    // simulated_quantize_op_param
    /* if (cpu_parameters.cpu_operation.common_only_op.common.op_type == SIMULATED_QUANTIZE) */
    /* { */

    /*     for (i = 0; i < 3; i++) */
    /*     { */
    /*         pSrc = cpu_parameters.cpu_operation.simulated_quantize_op.src_data[i].address; */
    /*         dat_h = cpu_parameters.cpu_operation.simulated_quantize_op.src_data[i].height; */
    /*         dat_w = cpu_parameters.cpu_operation.simulated_quantize_op.src_data[i].width; */
    /*         dat_c = cpu_parameters.cpu_operation.simulated_quantize_op.src_data[i].channel; */
    /*         line_stride_in = cpu_parameters.cpu_operation.simulated_quantize_op.src_data[i].line_stride; */
    /*         surf_stride_in = cpu_parameters.cpu_operation.simulated_quantize_op.src_data[i].surf_stride; */
    /*         data_type = cpu_parameters.cpu_operation.simulated_quantize_op.src_data[i].datatype; */
    /*     } */
    /* } */

    // pool2d_param
    if (cpu_parameters.cpu_operation.common_only_op.common.op_type == POOL2D)
    {
        debug_info("\t pool_type:%d;\n"
                   "\t layout: %u;\n",
                   cpu_parameters.cpu_operation.pool2d_op.pool_type,
                   cpu_parameters.cpu_operation.pool2d_op.layout); // 0 max, 1 average

        for (i = 0; i < 2; i++)
        {
            debug_info("\t\n  kernel[%d] : %d\n", i, cpu_parameters.cpu_operation.pool2d_op.kernel[i]);
        }
        for (i = 0; i < 2; i++)
        {
            debug_info("\t\n  kernel[%d] : %d\n", i, cpu_parameters.cpu_operation.pool2d_op.strides[i]);
        }
        for (i = 0; i < 4; i++)
        {
            debug_info("\t\n  kernel[%d] : %d\n", i, cpu_parameters.cpu_operation.pool2d_op.padding[i]);
        }

        debug_info("\t ceil_mode:%d;\n"
                   "\t count_include_pad: %u;\n",
                   cpu_parameters.cpu_operation.pool2d_op.ceil_mode,          // When true, will use ceil instead of floor to compute the output shape
                   cpu_parameters.cpu_operation.pool2d_op.count_include_pad); // only for average, When true, will include padding to compute the average
    }

    // resize_param
    if (cpu_parameters.cpu_operation.common_only_op.common.op_type == RESIZE)
    {
        debug_info("\t layout:%u;\n"
                   "\t method: %u;\n"
                   "\t coordinate_transf_mode: %u;\n"
                   "\t rounding_method: %u;\n",
                   cpu_parameters.cpu_operation.resize_op.layout,
                   cpu_parameters.cpu_operation.resize_op.method,
                   cpu_parameters.cpu_operation.resize_op.coordinate_transf_mode);
        debug_info("\n\t bicubic_alpha :");
        print_float(cpu_parameters.cpu_operation.resize_op.bicubic_alpha);
        debug_info("\t bicubic_exclude:%d;\n", cpu_parameters.cpu_operation.resize_op.bicubic_exclude);
    }
    // concat_param
    if (cpu_parameters.cpu_operation.common_only_op.common.op_type == CONCAT)
    {
        debug_info("\t axis:%d;\n", cpu_parameters.cpu_operation.concat_op.axis);

        for (i = 0; i < cpu_parameters.cpu_operation.concat_op.common.input_num; i++) /*input max number is 10 */
        {
            debug_info("src_data[%d]:\n"
                       "\t data_type:%u;\n"
                       "\t batchsize: %u;\n"
                       "\t height : %u\n"
                       "\t width : %u\n"
                       "\t channel: %u\n"
                       "\t line_stride : %d\n"
                       "\t surf_stride: %d\n"
                       "\t size: %u\n"
                       "\t offset: %u\n"
                       "\t address: 0x%08x\n",
                       i,
                       cpu_parameters.cpu_operation.concat_op.src_data[i].datatype,
                       cpu_parameters.cpu_operation.concat_op.src_data[i].batchsize,
                       cpu_parameters.cpu_operation.concat_op.src_data[i].height,
                       cpu_parameters.cpu_operation.concat_op.src_data[i].width,
                       cpu_parameters.cpu_operation.concat_op.src_data[i].channel,
                       cpu_parameters.cpu_operation.concat_op.src_data[i].line_stride,
                       cpu_parameters.cpu_operation.concat_op.src_data[i].surf_stride,
                       cpu_parameters.cpu_operation.concat_op.src_data[i].size,
                       cpu_parameters.cpu_operation.concat_op.src_data[i].index,
                       cpu_parameters.cpu_operation.concat_op.src_data[i].address);
        }
    }
    // slice_param
    if (cpu_parameters.cpu_operation.common_only_op.common.op_type == STRIDED_SLICE)
    {
        debug_info("\t slice_dims:%u;\n", cpu_parameters.cpu_operation.slice_op.slice_dims);
        for (i = 0; i < 4; i++)
        {
            debug_info("\t\n  begin[%d] : %d\n", i, cpu_parameters.cpu_operation.slice_op.begin[i]);
        }
        for (i = 0; i < 4; i++)
        {
            debug_info("\t\n  end[%d] : %d\n", i, cpu_parameters.cpu_operation.slice_op.end[i]);
        }
        for (i = 0; i < 4; i++)
        {
            debug_info("\t\n  stride[%d] : %d\n", i, cpu_parameters.cpu_operation.slice_op.stride[i]);
        }

        debug_info("\t slice_mode:%u;\n", cpu_parameters.cpu_operation.slice_op.slice_mode);
    }
    // take_param
    if (cpu_parameters.cpu_operation.common_only_op.common.op_type == TAKE)
    {
        debug_info("\t axis:%d;\n"
                   "\t take_mode:%d;\n",
                   cpu_parameters.cpu_operation.take_op.axis,
                   cpu_parameters.cpu_operation.take_op.take_mode);
        debug_info("indices: \n"
                   "\t data_type:%u;\n"
                   "\t batchsize: %u;\n"
                   "\t height : %u\n"
                   "\t width : %u\n"
                   "\t channel: %u\n"
                   "\t line_stride : %d\n"
                   "\t surf_stride: %d\n"
                   "\t size: %u\n"
                   "\t offset: %u\n"
                   "\t address: 0x%08x\n",
                   cpu_parameters.cpu_operation.take_op.indices.datatype,
                   cpu_parameters.cpu_operation.take_op.indices.batchsize,
                   cpu_parameters.cpu_operation.take_op.indices.height,
                   cpu_parameters.cpu_operation.take_op.indices.width,
                   cpu_parameters.cpu_operation.take_op.indices.channel,
                   cpu_parameters.cpu_operation.take_op.indices.line_stride,
                   cpu_parameters.cpu_operation.take_op.indices.surf_stride,
                   cpu_parameters.cpu_operation.take_op.indices.size,
                   cpu_parameters.cpu_operation.take_op.indices.index,
                   cpu_parameters.cpu_operation.take_op.indices.address);
    }
    // expand_dims_param
    if (cpu_parameters.cpu_operation.common_only_op.common.op_type == EXPAND_DIMS)
    {
        debug_info("\t axis:%d;\n"
                   "\t layout: %d;\n",
                   cpu_parameters.cpu_operation.expand_dims_op.axis,
                   cpu_parameters.cpu_operation.expand_dims_op.num_newaxis);
    }
    // one_hot_param
    if (cpu_parameters.cpu_operation.common_only_op.common.op_type == ONE_HOT)
    {
        debug_info("\t depth:%d;\n"
                   "\t axis: %d;\n",
                   cpu_parameters.cpu_operation.one_hot_op.depth,
                   cpu_parameters.cpu_operation.one_hot_op.axis);
        debug_info("\t on_val: "); print_float(cpu_parameters.cpu_operation.one_hot_op.on_value);
        debug_info("\n\t off_val: "); print_float(cpu_parameters.cpu_operation.one_hot_op.off_value); debug_info("\n");
    }
    // cast_param
    if (cpu_parameters.cpu_operation.common_only_op.common.op_type == CAST)
    {
        debug_info("\t cast datatype:%u;\n", cpu_parameters.cpu_operation.cast_op.datatype);
    }

    // reduce_op: sum, mean, max, min, all, any, argmax, argmin
    if ((cpu_parameters.cpu_operation.common_only_op.common.op_type == SUM) ||
        (cpu_parameters.cpu_operation.common_only_op.common.op_type == MEAN) ||
        (cpu_parameters.cpu_operation.common_only_op.common.op_type == MAX) ||
        (cpu_parameters.cpu_operation.common_only_op.common.op_type == MIN) ||
        (cpu_parameters.cpu_operation.common_only_op.common.op_type == ALL) ||
        (cpu_parameters.cpu_operation.common_only_op.common.op_type == ANY) ||
        (cpu_parameters.cpu_operation.common_only_op.common.op_type == ARGMAX)||
        (cpu_parameters.cpu_operation.common_only_op.common.op_type == ARGMIN))
    {
        for (i = 0; i < 4; i++)
        {
            debug_info("\t\n  axis[%d] : %d\n", i, cpu_parameters.cpu_operation.reduce_op.axis[i]);
        }
        debug_info("\t keepdims:%d;\n"
                   "\t exclude: %d;\n",
                   cpu_parameters.cpu_operation.reduce_op.keepdims,
                   cpu_parameters.cpu_operation.reduce_op.exclude);
    }

    // split_param
    if (cpu_parameters.cpu_operation.common_only_op.common.op_type == SPLIT)
    {
        debug_info("\t\n  indices[0] : %d\n", cpu_parameters.cpu_operation.split_op.indices[0]);
        debug_info("\t axis:%d;\n", cpu_parameters.cpu_operation.split_op.axis);
        for (i = 0; i < cpu_parameters.cpu_operation.split_op.common.output_num; i++) /*input max number is 10 */
        {
            debug_info("dst_data: \n"
                       "\t data_type:%u;\n"
                       "\t batchsize: %u;\n"
                       "\t height : %u\n"
                       "\t width : %u\n"
                       "\t channel: %u\n"
                       "\t line_stride : %d\n"
                       "\t surf_stride: %d\n"
                       "\t size: %u\n"
                       "\t offset: %u\n"
                       "\t address: 0x%08x\n",
                       cpu_parameters.cpu_operation.split_op.dst_data[i].datatype,
                       cpu_parameters.cpu_operation.split_op.dst_data[i].batchsize,
                       cpu_parameters.cpu_operation.split_op.dst_data[i].height,
                       cpu_parameters.cpu_operation.split_op.dst_data[i].width,
                       cpu_parameters.cpu_operation.split_op.dst_data[i].channel,
                       cpu_parameters.cpu_operation.split_op.dst_data[i].line_stride,
                       cpu_parameters.cpu_operation.split_op.dst_data[i].surf_stride,
                       cpu_parameters.cpu_operation.split_op.dst_data[i].size,
                       cpu_parameters.cpu_operation.split_op.dst_data[i].index,
                       cpu_parameters.cpu_operation.split_op.dst_data[i].address);
        }
    }
}

#endif
