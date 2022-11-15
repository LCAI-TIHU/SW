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

#ifndef __AIPU_IO_H_
#define __AIPU_IO_H_

#define __METAL_ACCESS_ONCE(x) (*(__typeof__(*x) volatile *)(x))
#define REG_ADDR(control_base, offset) ((unsigned long)control_base + offset)
#define REG_RW(control_base,offset)                                                      \
    (__METAL_ACCESS_ONCE((uint32_t *)REG_ADDR(control_base,offset)))

#define readb(a)            (*(volatile uint8_t *)(a))
#define readw(a)            (*(volatile uint16_t *)(a))
#define readl(a)            (*(volatile uint32_t *)(a))
#define readq(a)            (*(volatile uint64_t *)(a))

#define writeb(v,a)        (*(volatile uint8_t *)(a) = (v))
#define writew(v,a)        (*(volatile uint16_t *)(a) = (v))
#define writel(v,a)        (*(volatile uint32_t *)(a) = (v))
#define writeq(v,a)        (*(volatile uint64_t *)(a) = (v))

#define read64(a)     readq(a)
#define read32(a)     readl(a)
#define read16(a)     readw(a)
#define read8(a)      readb(a)

#define write64(v,a)  writeq(v,a)
#define write32(v,a)  writel(v,a)
#define write16(v,a)  writew(v,a)
#define write8(v,a)   writeb(v,a)


//TODO: device I/O, relaxed memory? depends on PMA?PMP?
#define read_reg(a)     read32(a) //read32(a)
#define write_reg(v, a) write32(v,a) //write32(v,a)

#endif
