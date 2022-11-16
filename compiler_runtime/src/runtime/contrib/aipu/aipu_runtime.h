/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Inspur.
 * This is a new or modified file.
 */

/*!
 * \file aipu_runtime.h
 * \brief Execution handling of aipu-N command streams.
 */
#ifndef TVM_RUNTIME_CONTRIB_AIPU_AIPU_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_AIPU_AIPU_RUNTIME_H_

#include <tvm/runtime/packed_func.h>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstring>
#if 1
#define NVDLA_UTILS_ERROR_TAG "DLA_TEST"
//#include "../../../relay/backend/contrib/aipu/codegen_aipu.h"
#endif
#include "nvdla_inf.h"
/* #include "nvdla_os_inf.h" */
#include "RuntimeTest.h"

namespace tvm {
namespace runtime {
namespace contrib{
  
enum operator_enum {
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
    GELU = 38U,
    NORM = 39U,
    FEATURETOWEIGHT = 255U
};

  // assume all elements are scalar
enum datatype_enum {
    RINT = 1U,
    RUINT = 2U,
    RFLOAT = 3U,
    RBFLOAT = 4U,
    RINT8 = 5U,
};

enum resize_layout_enum{
    NCHW = 1U,
    NHWC = 2U,
};

enum resize_method_enum{
    NEAREST = 1U,
    BILINEAR = 2U,
    BICUBIC = 3U,
};

enum resize_rounding_method_enum{ //indicates how to find the "nearest" pixel in NEAREST method
    ROUND = 1U, 
    FLOOR = 2U,
    CEIL = 3U,
};

enum resize_coordinate_transformation_mode_enum{
    HALF_PIXEL = 1U,
    ALIGN_CORNERS = 2U,
    ASYMMETRIC = 3U,
};

enum stride_slice_mode_enum{
    END = 1U,
};

typedef enum pad_mode_enum_ {
    CONSTANT = 1U,
    EDGE = 2U,
    REFLECT = 3U,
} pad_mode_enum;

enum take_mode_enum{
    CLIP = 1U, //"clip"
    FAST = 2U, //"fast"
    WRAP = 3U, //"wrap"
};

typedef struct Fused_function_offsets_ {
    // current fused function id
    int id;
    std::vector<size_t> input_offsets;
    std::vector<size_t> output_offsets;
    std::vector<size_t> input_size;
    std::vector<size_t> output_size;
    // fused function input id
    // std::vector<int> input_fused;
}Fused_function_offsets;

struct common_parameters {
    operator_enum op_type;
    uint32_t input_num;
    uint32_t output_num;
    float input_scale_factor[INPUT_MAX];
    float output_scale_factor[4];
}__attribute__((packed, aligned(64)));

struct op_buffer_desc {
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
}__attribute__((packed, aligned(64)));

//common_param only operator
struct common_only_op_param {
    struct common_parameters common;
}__attribute__((packed, aligned(4)));

//common_param + weight op
struct with_weight_op_param {
    struct common_parameters common;
    float weight_ioscale;
    uint32_t sub_op_type; // 1 add/multiply with 1x1x1xc , 0 add/multiply with 1x1x1x1 , 2 add/multiply with other
    struct op_buffer_desc weight;
}__attribute__((packed, aligned(4)));

//coordinate transform operator
struct transform_op_param {
    struct common_parameters common;
    int32_t axis[4];
}__attribute__((packed, aligned(4)));

struct pool2d_param {
    struct common_parameters common;
    int pool_type;          // 0 max, 1 average
    resize_layout_enum layout;
    int kernel[2];
    int strides[2];
    int padding[4];
    int ceil_mode;  // When true, will use ceil instead of floor to compute the output shape
    int count_include_pad;  // only for average, When true, will include padding to compute the average
}__attribute__((packed, aligned(4)));

struct pad_param {
    struct common_parameters common;
    pad_mode_enum pad_mode;
    int32_t pad_width[INPUT_MAX];
    struct op_buffer_desc src_data;
}__attribute__((packed, aligned(4)));

struct resize_param {
    struct common_parameters common;
    resize_layout_enum layout;
    resize_method_enum method;
    resize_coordinate_transformation_mode_enum coordinate_transf_mode;
    resize_rounding_method_enum rounding_method;
    double bicubic_alpha;
    int32_t bicubic_exclude;
}__attribute__((packed, aligned(4)));

struct concat_param {
    struct common_parameters common;
    int32_t axis;
    /*input max number is 10 */
    struct op_buffer_desc src_data[INPUT_MAX];
}__attribute__((packed, aligned(4)));

struct slice_param {
    struct common_parameters common;
    uint32_t slice_dims;
    int32_t begin[4];
    int32_t end[4];
    int32_t  stride[4];
    stride_slice_mode_enum slice_mode;
}__attribute__((packed, aligned(4)));

struct take_param {
    struct common_parameters common;
    int32_t axis;
    take_mode_enum take_mode;
    struct op_buffer_desc indices;
}__attribute__((packed, aligned(4)));

struct split_param {
    struct common_parameters common;
    int32_t indices[INPUT_MAX];//整数情况放在indices[q]
    int32_t axis;
    struct op_buffer_desc dst_data[INPUT_MAX];
}__attribute__((packed, aligned(4)));

struct expand_dims_param {
    struct common_parameters common;
    int32_t axis;
    int32_t num_newaxis;
}__attribute__((packed, aligned(4)));

struct one_hot_param {
    struct common_parameters common;
    int32_t depth;
    int32_t axis;
    float on_value;
    float off_value;
}__attribute__((packed, aligned(4)));

struct cast_param {
    struct common_parameters common;
    enum datatype_enum datatype;
}__attribute__((packed, aligned(4)));


struct reduce_op_param {
    struct common_parameters common;
    int32_t axis[4];  //0-3 in corresponding position do reduce op, -1 not do reduce op
    int16_t keepdims;
    int16_t exclude;
}__attribute__((packed, aligned(4)));

union cpu_operation_container {
    //softmax, exp, sigmoid, reshape, upsample, sqrt, erf, tanh, relu
    struct common_only_op_param    common_only_op;
    //add, less, batch_matmul, power, divide, max, substract, dense
    struct with_weight_op_param    with_weight_op;
    //tanspose, squeeze
    struct transform_op_param        transform_op;
    // reduce_op: sum, mean, max, min, all, any, argmax, argmin
    struct reduce_op_param              reduce_op;
    //other
    struct pool2d_param                 pool2d_op;
    struct pad_param                       pad_op;
    struct resize_param                 resize_op;
    struct concat_param                 concat_op;
    struct slice_param                   slice_op;
    struct take_param                     take_op;
    struct expand_dims_param       expand_dims_op;
    struct one_hot_param               one_hot_op;
    struct cast_param                     cast_op;
    struct split_param                   split_op;
};

struct op_buffers_desc {
    struct op_buffer_desc src_data;
    struct op_buffer_desc dst_data;
}__attribute__((packed, aligned(4)));

union cpu_operation_buffer_container {
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

struct cpu_param {
    union cpu_operation_buffer_container cpu_operation_buffer;
    union cpu_operation_container cpu_operation;
    // operator_enum op_type; //yuanyue 20220630
}__attribute__ ((packed, aligned(4)));

struct CpuTaskPackage {
    struct cpu_param cpu_op_param;
    uint64_t next;
}__attribute__ ((packed, aligned(4)));

/*
 * how to transmit task to firmware:
 * 1.if mac task,
 */
struct AipuTask{
	uint32_t dev_type;
	uint32_t num_addresses;
	uint64_t address_list; // address list pointer in the target memory. The addresses refer to the addresses of weights
	uint64_t task_pointer; // cpu task package pointer in the target memory
	uint64_t next;
}__attribute__ ((packed, aligned(4)));

struct AipuDlaTask {
    uint32_t num_addresses;
    uint64_t *address_list;
};

/*
 *Loadable_pair: DLA loadables' size and address;
 *Riscv_vector: RISCV tasks, runtime need to put cpu_param into a table;
 *Riscv_addr_list: include all address-offsets that riscv op needs, every Riscv task has an address list;
 *      pair<int32_t, size_t>: weight index and op offset, runtime use offset to allocate memory address for cpu_param,
 *      weight index is used to find weight source address and size in Riscv_wt_list, then runtime write weights to the address allocated by op offset;
 *Riscv_wt_list: weight source address that allocate by compiler and weight size;
 *Network_io: Network input/output table, address, size.
 */
typedef std::vector<std::pair<uint64_t, uint8_t*>> Loadable_pair;
//typedef std::vector<std::pair<std::vector<std::pair<int32_t,size_t>>, std::vector<cpu_param *>>> Riscv_vector;
typedef std::vector<std::vector<cpu_param *>> Riscv_vector;
typedef std::vector<std::vector<std::pair<int32_t,size_t>>> Riscv_addr_list;
typedef std::vector<std::pair<size_t, void*>> Riscv_wt_list;
typedef std::map<std::string, std::pair<size_t, size_t>> Network_io;//first is address, seconde is size
typedef std::vector<int> Execution_order;


class AIPUModuleNode : public runtime::ModuleNode {
public:
    
    AIPUModuleNode(Loadable_pair& lpair, Riscv_vector& rvector, Riscv_addr_list& addr_list, Riscv_wt_list& riscv_weight, Execution_order& eorder,
              std::vector<Fused_function_offsets>& io_offset, Network_io& input, Network_io& output, size_t& data_memory_used) {
        data_memory_used_ = data_memory_used;
        for (size_t i = 0; i < lpair.size(); i++) {
            loadable_.push_back(lpair[i]);
            // LOG(INFO) << lpair[i].first << " " << (static_cast<const void *>(lpair[i].second));
        }
        for (size_t i = 0; i < rvector.size(); i++) {
            std::vector<cpu_param *> param_tmp;
            for (size_t j = 0; j < rvector[i].size(); j++) {
                param_tmp.push_back(rvector[i][j]);
            }
            riscv_code_.push_back(param_tmp);
        }

        for (size_t i = 0; i < addr_list.size(); i++) {
            std::vector<std::pair<int32_t,size_t>> list_tmp;
            for (size_t j = 0; j < addr_list[i].size(); j++) {
                list_tmp.push_back(addr_list[i][j]);
            }
            riscv_addr_list_.push_back(list_tmp);
        }

        for (size_t i = 0; i < riscv_weight.size(); i++) {
            riscv_wt_list_.push_back(riscv_weight[i]);
        }
        for (size_t i = 0; i < eorder.size(); i++) {
            execution_order_.push_back(eorder[i]);
            // LOG(INFO) << eorder[i];
        }
        for (size_t i = 0; i < io_offset.size(); i++) {
            io_offset_.push_back(io_offset[i]);
            // LOG(INFO) << "id " << io_offset[i].id << " output_offset " << io_offset[i].output_offset;
            // for (size_t j = 0; j < io_offset[i].input_offsets.size(); j++)
            //   LOG(INFO) << "input_offset " << io_offset[i].input_offsets[j];
        }
        for (auto i = input.begin(); i != input.end(); i++) {
            input_[i->first] = i->second;
        }
        
        for (auto i = output.begin(); i != output.end(); i++) {
            output_[i->first] = i->second;
        }
    }

    AIPUModuleNode(std::vector<std::pair<long unsigned int, unsigned char*> >&, 
		    std::vector<std::vector<cpu_param*> >&, std::vector<int>&, 
		    std::vector<tvm::runtime::contrib::Fused_function_offsets_>&, 
		    long unsigned int&) {
	}

    // destructor
    ~AIPUModuleNode() {
      // LOG(INFO)<<"free 0";
    }

    const char* type_key() const final { return "aipu"; }


    PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final
    {
        return PackedFunc();
    }

    Loadable_pair GetLoadable() {
        return loadable_;
    }

    Riscv_vector GetRiscv() {
        return riscv_code_;
    }

    Execution_order GetExecutionOrder() {
        return execution_order_;
    }
    Riscv_addr_list GetRiscvAddrList() {
        return riscv_addr_list_;
    }
    Riscv_wt_list GetRiscvWtList() {
        return riscv_wt_list_;
    }
    Network_io GetInputParam() {
        return input_;
    }
    Network_io GetOutputParam() {
        return output_;
    }
    std::vector<Fused_function_offsets> GetFusedOffset() {
        return io_offset_;
    }
  private:
    Loadable_pair loadable_;
    Riscv_vector riscv_code_;
    Network_io input_;
    Network_io output_;
    Riscv_addr_list riscv_addr_list_; 
    Riscv_wt_list riscv_wt_list_;
    // execution order, positive for loadable, negative for riscv
    Execution_order execution_order_;
    // TODO: need some examples to understand and support multiple output
    std::vector<Fused_function_offsets> io_offset_;

    // used for separate fused function memory space and inner memory space of fused function
    size_t data_memory_used_;
};


}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_AIPU_AIPU_RUNTIME_H_
