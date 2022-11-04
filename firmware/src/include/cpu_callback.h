#ifndef __CPU_CALLBACK_H
#define __CPU_CALLBACK_H
#include "device_init.h"

int32_t read_cpu_address(struct cpu_device *cpu_dev, int16_t index, void *dst);
int32_t cpu_data_read(struct cpu_device *cpu_dev, uint64_t src, void *dst, uint32_t size, uint64_t offset);
int32_t cpu_task_submit(struct cpu_device *dla_dev, struct cpu_task *task);


#endif
