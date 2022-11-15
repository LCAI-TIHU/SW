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

#ifndef _DLA_H_
#define _DLA_H_
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <aipu_parent_clock.h>
#include "aipu_scu.h"

//#define __iomem __attribute__((at(0x20000000)))
/**
 * host writes dla_task and cpu_task to addresses below
 */
#define SYSTEM_CLK 20000000UL
#define DLA_TASK_ADDR 0xa0000000
#define DLA_BASE_ADDR 0x88000000 // firmware memory space for runtime 384M,0x88000000 ~ 0xa0000000
#define COMPILER_BASE_ADDR 0xa0100000 // firmware memory space for compiler,0xa0000000 ~ 0xffffffff

#define PLIC_PENDING1_ADDR  0x40b01000
#define PLIC_PENDING2_ADDR  0x40b01004
#define PLIC_ENABLE1_ADDR   0x40b02000
#define PLIC_ENABLE2_ADDR   0x40b02004
#define PLIC_THRESHODE_ADDR 0x40d00000
#define PCIE_INTR_PRIORITY  0x40b00000 + 4 * 16
#define MAC_INTR_PRIORITY   0x40b00000 + 4 * 2

#define C_ATM 32
#define INPUT_MAX 12
#define DEBUG_SURF 4
#define DEBUG_H 4
#define DEBUG_W 4
#define DEBUG_C 4*32
/**
 * @brief           Task information submitted from driver
 *
 * num_addresses        Number of addresses in address list
 * dla_dev          Pointer to NVDLA device
 * address_list         Address list
 */
struct device_dla_task
{
    uint32_t num_addresses;
    // struct dla_device *dla_dev;
    uint64_t *address_list;
} __attribute__((packed, aligned(4)));

// struct device_dla_task_tmp {
//   uint32_t num_addresses;
//   uint64_t *address_list;
// };
struct device_task_tmp
{
    uint32_t dev_type;
    uint32_t num_addresses;
    uint64_t address_list;
    // uint64_t param;
    uint64_t task_pointer;
    uint64_t next;
} __attribute__((packed, aligned(4)));

/**
 * @brief           WUYISHAN device
 *
 * dla_irq              Interrupt number associated with this device
 * base             IO mapped base address for device
 * dla_task         Pointer to dla_task in execution
 * config_data          Pointer to the configuration data
 * pdev             Pointer to NVDLA platform device
 * event_notifier       Completion object used to wait for events from mac HW
 * engine_context       Private data passed from engine in dla_engine_init
 */
struct dla_device
{
    uint32_t dla_irq;
    uint64_t *base;
    struct device_dla_task *task;
    struct dla_config *config_data;
    int32_t event_notifier; // realize completion as linux
    void *engine_context;
};

typedef enum operator_enum_
{
    CONV2D = 1U,
    DENSE = 2U,
    RELU = 3U,
    SOFTMAX = 4U,
    RESHAPE = 5U,
    POOL2D = 6U,
    ADD = 7U,
    RESIZE = 8U,
    CONCAT = 9U,
    SIGMOID = 10U,
    STRIDED_SLICE = 11U,
    MULTIPLY = 12U,
    EXP = 13U,
    EXPAND_DIMS = 14U,
    TAKE = 15U,
    ONE_HOT = 16U,
    LESS = 17U,
    BATCH_MATMUL = 18U,
    TRANSPOSE = 19U,
    CAST = 20U,
    POWER = 21U,
    DIVIDE = 22U,
    MAX = 23U,
    SQRT = 24U,
    ERF = 25U,
    TANH = 26U,
    SUBTRACT = 27U,
    SUM = 28U,
    SPLIT = 29U,
    MEAN = 30U,
    SQUEEZE = 31U,
    UNKNOWN = 32U,
    MIN = 33U,
    ALL = 34U,
    ANY = 35U,
    ARGMAX = 36U,
    ARGMIN = 37U,
} operator_enum;

// assume all elements are scalar
typedef enum datatype_enum
{
    RINT = 1U,
    RUINT = 2U,
    RFLOAT = 3U,
    RBFLOAT = 4U,
    RINT8 = 5U,
} datatype_enum;

typedef enum resize_layout_enum
{
    NCHW = 1U,
    NHWC = 2U,
} resize_layout_enum;

typedef enum resize_method_enum
{
    NEAREST = 1U,
    BILINEAR = 2U,
    BICUBIC = 3U,
} resize_method_enum;

typedef enum resize_rounding_method_enum
{ // indicates how to find the "nearest" pixel in NEAREST method
    ROUND = 1U,
    FLOOR = 2U,
    CEIL = 3U,
} resize_rounding_method_enum;

typedef enum resize_coordinate_transformation_mode_enum
{
    HALF_PIXEL = 1U,
    ALIGN_CORNERS = 2U,
    ASYMMETRIC = 3U,
} resize_coordinate_transformation_mode_enum;

typedef enum stride_slice_mode_enum
{
    END = 1U,
} stride_slice_mode_enum;

typedef enum pad_mode_enum_
{
    CONSTANT = 1U,
    EDGE = 2U,
    REFLECT = 3U,
} pad_mode_enum;

typedef enum take_mode_enum_{
    CLIP = 1U, //"clip"
    FAST = 2U, //"fast"
    WRAP = 3U, //"wrap"
}take_mode_enum;


struct common_parameters
{
    operator_enum op_type;
    uint32_t input_num;
    uint32_t output_num;
    float input_scale_factor[INPUT_MAX];
    float output_scale_factor[4];
} __attribute__((packed, aligned(64)));

struct op_buffer_desc
{
    uint64_t address;
    uint32_t index;
    uint32_t size;

    datatype_enum datatype;

    uint32_t batchsize;
    uint32_t height;
    uint32_t width;
    uint32_t channel;

    int32_t line_stride;
    int32_t surf_stride;
} __attribute__((packed, aligned(64)));

// common_param only operator
struct common_only_op_param
{
    struct common_parameters common;
} __attribute__((packed, aligned(4)));

// common_param + weight op
struct with_weight_op_param
{
    struct common_parameters common;
    float weight_ioscale;
    uint32_t sub_op_type; // 1 add/multiply with 1x1xc , 0 add/multiply with a const
    struct op_buffer_desc weight;
} __attribute__((packed, aligned(4)));

// coordinate transform operator
struct transform_op_param
{
    struct common_parameters common;
    int32_t axis[4];
} __attribute__((packed, aligned(4)));

struct simulated_quantize_op_param
{
    struct common_parameters common;
    struct op_buffer_desc src_data[3];
} __attribute__((packed, aligned(4)));

struct pool2d_param
{
    struct common_parameters common;
    int pool_type; // 0 max, 1 average
    resize_layout_enum layout;
    int kernel[2];
    int strides[2];
    int padding[4];
    int ceil_mode;         // When true, will use ceil instead of floor to compute the output shape
    int count_include_pad; // only for average, When true, will include padding to compute the average
} __attribute__((packed, aligned(4)));

struct pad_param
{
    struct common_parameters common;
    pad_mode_enum pad_mode;
    int32_t pad_width[INPUT_MAX];
    struct op_buffer_desc src_data;
} __attribute__((packed, aligned(4)));

struct resize_param
{
    struct common_parameters common;
    resize_layout_enum layout;
    resize_method_enum method;
    resize_coordinate_transformation_mode_enum coordinate_transf_mode;
    resize_rounding_method_enum rounding_method;
    double bicubic_alpha;
    int32_t bicubic_exclude;
} __attribute__((packed, aligned(4)));

struct concat_param
{
    struct common_parameters common;
    int32_t axis;
    /*input max number is 10 */
    struct op_buffer_desc src_data[INPUT_MAX];
} __attribute__((packed, aligned(4)));

struct slice_param
{
    struct common_parameters common;
    uint32_t slice_dims;
    int32_t begin[4];
    int32_t end[4];
    int32_t stride[4];
    stride_slice_mode_enum slice_mode;
} __attribute__((packed, aligned(4)));

struct take_param {
    struct common_parameters common;
    int32_t axis;
    take_mode_enum take_mode;
    struct op_buffer_desc indices;
}__attribute__((packed, aligned(4)));

struct split_param
{
    struct common_parameters common;
    int32_t indices[INPUT_MAX]; //整数情况放在indices[q]
    int32_t axis;
    struct op_buffer_desc dst_data[INPUT_MAX];
} __attribute__((packed, aligned(4)));

struct expand_dims_param
{
    struct common_parameters common;
    int32_t axis;
    int32_t num_newaxis;
} __attribute__((packed, aligned(4)));

struct one_hot_param
{
    struct common_parameters common;
    int32_t depth;
    int32_t axis;
    float on_value;
    float off_value;
} __attribute__((packed, aligned(4)));

struct cast_param
{
    struct common_parameters common;
    enum datatype_enum datatype;
} __attribute__((packed, aligned(4)));

struct reduce_op_param
{
    struct common_parameters common;
    int32_t axis[4];
    int16_t keepdims;
    int16_t exclude;
} __attribute__((packed, aligned(4)));

union cpu_operation_container
{
    // softmax, exp, sigmoid, reshape, upsample, sqrt, erf, tanh, relu
    struct common_only_op_param common_only_op;
    // add, less, batch_matmul, power, divide, max, substract, dense
    struct with_weight_op_param with_weight_op;
    // tanspose, squeeze
    struct transform_op_param transform_op;
    // reduce_op: sum, mean, max, min, all, any, argmax, argmin
    struct reduce_op_param reduce_op;
    // other
    struct pool2d_param pool2d_op;
    struct pad_param pad_op;
    struct resize_param resize_op;
    struct concat_param concat_op;
    struct slice_param slice_op;
    struct take_param take_op;
    struct expand_dims_param expand_dims_op;
    struct one_hot_param one_hot_op;
    struct cast_param cast_op;
    struct split_param split_op;
};

struct op_buffers_desc
{
    struct op_buffer_desc src_data;
    struct op_buffer_desc dst_data;
} __attribute__((packed, aligned(4)));

union cpu_operation_buffer_container
{
    struct op_buffers_desc common_only_op_buffers;
    struct op_buffers_desc with_weight_op_buffers;
    struct op_buffers_desc transform_op_buffers;
    struct op_buffers_desc resize_buffers;
    struct op_buffers_desc concat_buffers;
    struct op_buffers_desc slice_buffers;
    struct op_buffers_desc take_buffers;
    struct op_buffers_desc expand_dims_buffers;
    struct op_buffers_desc one_hot_buffers;
    struct op_buffers_desc cast_buffers;
    struct op_buffers_desc reduce_buffers;
    struct op_buffers_desc split_buffers;
    struct op_buffers_desc simulated_quantize_buffers;
    struct op_buffers_desc pool2d_buffers;
    struct op_buffers_desc pad_buffers;
};

struct cpu_param
{
    union cpu_operation_buffer_container cpu_operation_buffer;
    union cpu_operation_container cpu_operation;
    // operator_enum op_type; //yuanyue 20220630
} __attribute__((packed, aligned(4)));

/**
 * @brief           Task information submitted from driver
 *
 * num_addresses        Number of addresses in address list
 * cpu_dev              Pointer to CPU device
 * address_list         Address list
 */

struct cpu_task_package
{
    struct cpu_param cpu_parameters;
    uint64_t next;
    // struct cpu_task_package *next;
} __attribute__((packed, aligned(4)));

struct cpu_task
{
    uint32_t num_addresses;
    struct cpu_device *cpu_dev;
    uint64_t *address_list;
    // struct cpu_param *cpu_parameters;
    struct cpu_task_package *cpu_task_pt;
} __attribute__((packed, aligned(4)));

/**
 * @brief           Configuration parameters supported by the CPU
 *
 * atom_size            Memory smallest access size
 * vlen                 vector register length in bits
 */
struct cpu_config
{
    uint32_t atom_size;
    uint64_t vlen;
} __attribute__((packed, aligned(4)));

/**
 * @brief           WUYISHAN device
 *
 * cpu_irq              Interrupt number associated with this device
 * base             IO mapped base address for device
 * cpu_task         Pointer to cpu_task in execution
 * config_data          Pointer to the configuration data
 * event_notifier       Completion object used to wait for events from mac HW
 */
struct cpu_device
{
    uint32_t cpu_irq;
    uint64_t *base;
    struct cpu_task *task;
    struct cpu_config *config_data;
    int32_t event_notifier; // realize completion as linux
} __attribute__((packed, aligned(4)));

/**
 * PCIE DEVICE
 */
struct pcie_device
{
    uint32_t pcie_irq;
    bool pcie_task_done;
} __attribute__((packed, aligned(4)));

/**
 * struct dla_submit_task structure for single task information
 *
 * @num_addresses       total number of entries in address_list
 * @reserved            Reserved for padding
 * @address_list        pointer to array of struct nvdla_mem_handle
 *
 */
struct dla_submit_task
{
#define NVDLA_MAX_BUFFERS_PER_TASK (6144)
    uint32_t num_addresses;
#define NVDLA_NO_TIMEOUT (0xffffffff)
    uint32_t timeout;
    uint64_t address_list;
};

/**
 * struct nvdla_submit_args structure for task submit
 *
 * @tasks       pointer to array of struct nvdla_ioctl_submit_task
 * @num_tasks       number of entries in tasks
 * @flags       flags for task submit, no flags defined yet
 * @version     version of task structure
 *
 */
struct dla_submit_args
{
    uint64_t tasks;
    uint64_t num_tasks;
#define NVDLA_MAX_TASKS_PER_SUBMIT 24
#define NVDLA_SUBMIT_FLAGS_ATOMIC (1 << 0)
    uint64_t flags;
    uint32_t version;
};

/**
 * @brief           Submit task
 *
 * This function submits task to NVDLA engine.
 *
 * @param nvdla_dev     Pointer to NVDLA device
 * @param task          Pointer to task
 * @return          0 on success and negative on error
 *
 */
int32_t dla_task_submit(struct dla_device *dla_dev, struct device_dla_task *task);
/**
 * @brief           get nvdla device
 *
 * This function gets NVDLA device.
 *
 */
// extern struct dla_device dla_dev;
// extern struct cpu_device cpu_dev;
// extern struct pcie_device pcie_dev;
// extern struct device_dla_task dla_tsk;
// extern struct cpu_task cpu_tsk;

int32_t cpu_init(struct dla_device *dla_dev, struct pcie_device *pcie_dev);
int32_t device_init(struct cpu_device *cpu_dev, struct dla_device *dla_dev);
// #define INPUT_DEBUG 
// #define OUTPUT_DEBUG 
// #define DEV_DEBUG 
#ifdef DEV_DEBUG
void device_debug(struct cpu_device *cpu_dev, struct dla_device *dla_dev, uint32_t dev_type);
int32_t cpu_input_debug(struct cpu_param cpu_parameters);
int32_t cpu_output_debug(struct cpu_param cpu_parameters);
void cpu_param_debug(struct cpu_param cpu_parameters);
#else
static inline void device_debug(struct cpu_device *cpu_dev, struct dla_device *dla_dev, uint32_t dev_type)
{
}
static inline int32_t cpu_input_debug(struct cpu_param cpu_parameters) { return 0; }
static inline int32_t cpu_output_debug(struct cpu_param cpu_parameters) { return 0; }
static inline void cpu_param_debug(struct cpu_param cpu_parameters) {}
#endif

#endif
