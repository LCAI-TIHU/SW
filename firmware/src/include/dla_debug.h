/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __FIRMWARE_DLA_DEBUG_H_
#define __FIRMWARE_DLA_DEBUG_H_

#include <stdio.h>
#include <math.h>


#define EPERM  	1 //operation not allowed
#define ENOENT  2 //derectory can't be fined
#define EFAULT  14 //address error
#define EINVAL	22 //parameters error
#define AXIS_ERR 11
#define TIMEOUT 33 //time out error
#define NO_CPU_TASK 44
#define RESHAPE_ERR 45
#define TYPE_ERR 55
#define CONCAT_ERR 66
#define SHAPE_ERR 77
#define MULTIPLY_ERR 88
#define INPUT_ERR 99
#define IRQ_NONE 0

#define STRINGIFY(s) #s
#define DEFER_STRINGIFY(s) STRINGIFY(s)
#define FILELINE DEFER_STRINGIFY(__LINE__)
#define FILENAME DEFER_STRINGIFY(__FILE__)

#define LOG_EVENT_BDMA_SHIFT		0U
#define LOG_EVENT_CONV_SHIFT		4U
#define LOG_EVENT_SDP_SHIFT		8U
#define LOG_EVENT_PDP_SHIFT		12U
#define LOG_EVENT_CDP_SHIFT		16U
#define LOG_EVENT_RBK_SHIFT		20U
#define LOG_EVENT_GROUP_SHIFT		24U
#define LOG_EVENT_ROI_SHIFT		28U

#define LOG_TASK_START			1
#define LOG_TASK_END			2
#define LOG_READ_OP_CONFIG_START	3
#define LOG_READ_OP_CONFIG_END		4
#define LOG_READ_SURF_CONFIG_START	5
#define LOG_READ_SURF_CONFIG_END	6
#define LOG_PROGRAM_START		7
#define LOG_PROGRAM_END			8
#define LOG_OPERATION_START		9
#define LOG_OPERATION_END		10

#define LOG_EVENT(roi, group, processor, event)

/**
 * Used to enable/disable reading stat registers
 */
#define STAT_ENABLE		1

/**
 * Used to print debug network data
 */
#define DEBUG_NETWORK_DATA		0

/* #define assert(condition) */

#define RETURN(err) { return (err); }

#define DEBUG_ASSERT

#ifdef DEBUG_ASSERT
#define ASSERT_GOTO(_condition, _ret, _err_value, _goto)	\
do {								\
	if (!(_condition)) {					\
		printf("Assertion Fail(" FILENAME FILELINE "):" \
					STRINGIFY(_condition));	\
		_ret = _err_value;				\
		goto _goto;					\
	} else {						\
		_ret = 0;					\
	}							\
} while (0)
#else
#define ASSERT_GOTO(_condition, _ret, _err_value, _goto) assert(condition)
#endif /* DEBUG_ASSERT */


// #define PCIE_DEBUG
// #define UART_DEBUG 
#ifdef UART_DEBUG
	void debug_trace(const char *format, ...);
	//void debug_info(const char *format, ...);
     static inline void debug_info(const char *format, ...) {} 
#else
	static inline void debug_trace(const char *format, ...) {}
	void debug_info(const char *format, ...);
	//static inline void debug_info(const char *format, ...) {}
#endif

#define pr_dump_stack(format, ...)
#define dla_trace(...) debug_trace(__VA_ARGS__)

//#define OP_DEBUG
#ifdef OP_DEBUG
void debug_op(const char *format, ...);
#else
static inline void debug_op(const char *format, ...){}
//static inline void print_float(float_t vec){}
#endif

void print_float(float_t vec);


//#define TIMEOUT_DEBUG
#ifdef TIMEOUT_DEBUG
	void time_out_debug(uint64_t *t1, uint64_t *t2);
#else
	static inline void time_out_debug(uint64_t *t1, uint64_t *t2){}
#endif

#define DLA_OUT_DEBUG

#endif
