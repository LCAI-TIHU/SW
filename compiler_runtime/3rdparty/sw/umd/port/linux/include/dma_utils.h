/*
 * This file is part of the Xilinx DMA IP Core driver tools for Linux
 *
 * Copyright (c) 2016-present,  Xilinx, Inc.
 * All rights reserved.
 *
 * This source code is licensed under BSD-style license (found in the
 * LICENSE file in the root directory of this source tree)
 */

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <sys/types.h>

/*
 * man 2 write:
 * On Linux, write() (and similar system calls) will transfer at most
 * 	0x7ffff000 (2,147,479,552) bytes, returning the number of bytes
 *	actually transferred.  (This is true on both 32-bit and 64-bit
 *	systems.)
 */

//#define RW_MAX_SIZE	0x7ffff000 //shenfw comment out


//uint64_t getopt_integer(char *optarg);
//ssize_t read_to_buffer(char *fname, int fd, char *buffer, uint64_t size, uint64_t base);
ssize_t write_from_buffer(char *fname, int fd, char *buffer, uint64_t size, uint64_t base);
