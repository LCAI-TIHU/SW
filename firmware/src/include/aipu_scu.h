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

#ifndef AIPU_SCU_H
#define AIPU_SCU_H
#include <scu_reg.h>
#include <aipu_io.h>
#include <stdint.h>
#include <dla_debug.h>

#define SCU_REG_RW(offset) 			REG_RW(SCU_BASE, offset)
#define SW_INTERRUPT0_CLR			~(1<<0)
#define SW_INTERRUPT1_CLR			~(1<<1)
#define PCIE_INTERRUPT1_SET			1<<0
#define PCIE_INTERRUPT1_CLR			~(1<<0)
static __inline__ uint32_t read_sw_intr0_1(void)
{
	return (uint32_t)SCU_REG_RW(SW_INTERRUPT);
}

static __inline__ void clean_sw_intr0(void)
{
	SCU_REG_RW(SW_INTERRUPT) &= SW_INTERRUPT0_CLR;
}
static __inline__ void clean_sw_intr1(void)
{
	SCU_REG_RW(SW_INTERRUPT) &= SW_INTERRUPT1_CLR;
}
static __inline__ uint32_t check_pcie_intr_ack(void)
{
	return SCU_REG_RW(PCIE_INT_ACK)&0x00000001;
}
static __inline__ void pcie_intr_set(void)
{
	SCU_REG_RW(PCIE_INTERRUPT) |= PCIE_INTERRUPT1_SET;
}
static __inline__ void pcie_intr_clr(void)
{
	SCU_REG_RW(PCIE_INTERRUPT) &= PCIE_INTERRUPT1_CLR;
}
static __inline__ void pcie_intr_set_clr(void)
{
	pcie_intr_set();
	while(check_pcie_intr_ack() == 0x00000000)
		{
			debug_trace("Pcie ack == 0\n");
		}
	pcie_intr_clr();//pecie test
}
static __inline__ void mac_soft_reset(void)
{
	SCU_REG_RW(MODULE_RST_CTRL) |= (1<<1);
	__asm__ volatile("nop");
	__asm__ volatile("nop");
	__asm__ volatile("nop");
	__asm__ volatile("nop");
	__asm__ volatile("nop");
	__asm__ volatile("nop");
	__asm__ volatile("nop");
	__asm__ volatile("nop");
	__asm__ volatile("nop");
	__asm__ volatile("nop");
	SCU_REG_RW(MODULE_RST_CTRL) &= ~(1<<1);
}
#endif
