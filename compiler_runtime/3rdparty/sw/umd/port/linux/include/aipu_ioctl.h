/*
 * This file is part of the Xilinx DMA IP Core driver for Linux
 *
 * Copyright (c) 2016-present,  Xilinx, Inc.
 * All rights reserved.
 *
 * This source code is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * The full GNU General Public License is included in this distribution in
 * the file called "COPYING".
 */

/*
 * Inspur.
 * This is a new or modified file.
 */

#ifndef _XDMA_IOCALLS_POSIX_H_
#define _XDMA_IOCALLS_POSIX_H_

//#include <linux/ioctl.h>
#include "nvdla_ioctl.h"
#define IOCTL_XDMA_PERF_V1 (1)
#define XDMA_ADDRMODE_MEMORY (0)
#define XDMA_ADDRMODE_FIXED (1)

/*
 * S means "Set" through a ptr,
 * T means "Tell" directly with the argument value
 * G means "Get": reply by setting through a pointer
 * Q means "Query": response is on the return value
 * X means "eXchange": switch G and S atomically
 * H means "sHift": switch T and Q atomically
 *
 * _IO(type,nr)             no arguments
 * _IOR(type,nr,datatype)   read data from driver
 * _IOW(type,nr.datatype)   write data to driver
 * _IORW(type,nr,datatype)  read/write data
 *
 * _IOC_DIR(nr)             returns direction
 * _IOC_TYPE(nr)            returns magic
 * _IOC_NR(nr)              returns number
 * _IOC_SIZE(nr)            returns size
 */

struct xdma_performance_ioctl {
        /* IOCTL_XDMA_IOCTL_Vx */
        uint32_t version;
        uint32_t transfer_size;
        /* measurement */
        uint32_t stopped;
        uint32_t iterations;
        uint64_t clock_cycle_count;
        uint64_t data_cycle_count;
        uint64_t pending_count;
};



/* IOCTL codes */

#define IOCTL_XDMA_PERF_START   _IOW('q', 1, struct xdma_performance_ioctl *)
#define IOCTL_XDMA_PERF_STOP    _IOW('q', 2, struct xdma_performance_ioctl *)
#define IOCTL_XDMA_PERF_GET     _IOR('q', 3, struct xdma_performance_ioctl *)
#define IOCTL_XDMA_ADDRMODE_SET _IOW('q', 4, int)
#define IOCTL_XDMA_ADDRMODE_GET _IOR('q', 5, int)
#define IOCTL_XDMA_ALIGN_GET    _IOR('q', 6, int)
#define IOCTL_XDMA_RISCV_IRQ    _IOW('q', 7, int) // shenfw add
#define IOCTL_XDMA_HOST_IRQ    _IOW('q', 8, int) // shenfw add
//#define IOCTL_AIPU_SUBMIT_TASK    _IOW('q', 9, struct nvdla_gem_destroy_args) // shenfw add

#endif /* _XDMA_IOCALLS_POSIX_H_ */
