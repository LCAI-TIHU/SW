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

#ifndef _INSPUR_TIMER_REG_H_
#define _INSPUR_TIMER_REG_H_
#define  TIMER0_BASE						0x40110000
#define  TIMER1_BASE 						0x40120000

#define  REGISTER_BANK_CLOCK_ENABLE_SET     0x00000000
#define  REGISTER_BANK_CLOCK_ENABLE_CLR     0x00000004
#define  REGISTER_BANK_CLOCK_ENABLE_STATUS  0x00000008
#define  KERNEL_CLOCK_ENABLE_SET            0x0000000C
#define  KERNEL_CLOCK_ENABLE_CLR            0x00000010
#define  KERNEL_CLOCK_ENABLE_STATUS         0x00000014
#define  CHIPID                             0x00000018
#define  MODULEID                           0x0000001C
#define  CHECKSUM                           0x00000020
#define  FUNC_INT_0_MASK_SET                0x00000024
#define  FUNC_INT_0_MASK_CLEAR              0x00000028
#define  FUNC_INT_0_MASK_STATUS             0x0000002C
#define  FUNC_INT_0_SET                     0x00000030
#define  FUNC_INT_0_CLR                     0x00000034
#define  FUNC_INT_0_STATUS_MASKED           0x00000038
#define  FUNC_INT_0_STATUS                  0x0000003C
#define  FUNC_INT_1_MASK_SET                0x00000040
#define  FUNC_INT_1_MASK_CLEAR              0x00000044
#define  FUNC_INT_1_MASK_STATUS             0x00000048
#define  FUNC_INT_1_SET                     0x0000004C
#define  FUNC_INT_1_CLR                     0x00000050
#define  FUNC_INT_1_STATUS_MASKED           0x00000054
#define  FUNC_INT_1_STATUS                  0x00000058
#define  TIMER1LOAD                         0x0000005c
#define  TIMER1CURVALUE                     0x00000060
#define  TIMER1CTRL                         0x00000064
#define  TIMER1READREQ                      0x00000068
#define  TIMER2LOAD                         0x0000006c
#define  TIMER2CURVALUE                     0x00000070
#define  TIMER2CTRL                         0x00000074
#define  TIMER2READREQ                      0x00000078

#endif
