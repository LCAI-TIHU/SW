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

#ifndef __CPU_CALLBACK_H
#define __CPU_CALLBACK_H
#include "device_init.h"

int32_t read_cpu_address(struct cpu_device *cpu_dev, int16_t index, void *dst);
int32_t cpu_data_read(struct cpu_device *cpu_dev, uint64_t src, void *dst, uint32_t size, uint64_t offset);
int32_t cpu_task_submit(struct cpu_device *dla_dev, struct cpu_task *task);


#endif
