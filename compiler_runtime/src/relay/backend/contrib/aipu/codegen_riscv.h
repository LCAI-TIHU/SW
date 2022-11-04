
/*
 * Inspur.
 * This is a new or modified file.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_AIPU_CODEGEN_RISCV_H_
#define TVM_RELAY_BACKEND_CONTRIB_AIPU_CODEGEN_RISCV_H_

#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../../../runtime/contrib/aipu/aipu_runtime.h"
#include "../../utils.h"
#include "graph_plan_memory.h"  // yuanyue 20220622 plan memory
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"

//yuanyue 20220516 broadcast
#define RISCV_BROADCAST_ON 0 ///1 on; 0 off

namespace tvm {
namespace relay {
namespace contrib {
namespace aipu {

using datatype_enum = tvm::runtime::contrib::datatype_enum;
using operator_enum = tvm::runtime::contrib::operator_enum;
using stride_slice_mode_enum = tvm::runtime::contrib::stride_slice_mode_enum;
using resize_layout_enum = tvm::runtime::contrib::resize_layout_enum;
using resize_method_enum = tvm::runtime::contrib::resize_method_enum;
using resize_rounding_method_enum = tvm::runtime::contrib::resize_rounding_method_enum;
using take_mode_enum = tvm::runtime::contrib::take_mode_enum;
using resize_coordinate_transformation_mode_enum =
    tvm::runtime::contrib::resize_coordinate_transformation_mode_enum;
using pad_mode_enum = tvm::runtime::contrib::pad_mode_enum;

using Riscv_vector = tvm::runtime::contrib::Riscv_vector;
using Riscv_addr_list = tvm::runtime::contrib::Riscv_addr_list;
using Riscv_wt_list = tvm::runtime::contrib::Riscv_wt_list;
using Network_io = tvm::runtime::contrib::Network_io;
using Execution_order = tvm::runtime::contrib::Execution_order;
using Cpu_param = tvm::runtime::contrib::cpu_param;
using Cpu_operation_buffer_container = tvm::runtime::contrib::cpu_operation_buffer_container;
using Cp_buffers_desc = tvm::runtime::contrib::op_buffers_desc;
using Cpu_operation_container = tvm::runtime::contrib::cpu_operation_container;
using Common_parameters = tvm::runtime::contrib::common_parameters;
using Op_buffer_desc = tvm::runtime::contrib::op_buffer_desc;
using Fused_function_offsets = tvm::runtime::contrib::Fused_function_offsets;

using Common_only_op_param = tvm::runtime::contrib::common_only_op_param;
using With_weight_op_param = tvm::runtime::contrib::with_weight_op_param;
using Transform_op_param = tvm::runtime::contrib::transform_op_param;

using Resize_param = tvm::runtime::contrib::resize_param;
using Concat_param = tvm::runtime::contrib::concat_param;
using Slice_param = tvm::runtime::contrib::slice_param;
using Take_param = tvm::runtime::contrib::take_param;
using Expand_dims_param = tvm::runtime::contrib::expand_dims_param;
using One_hot_param = tvm::runtime::contrib::one_hot_param;
using Cast_param = tvm::runtime::contrib::cast_param;
using Reduce_op_param = tvm::runtime::contrib::reduce_op_param;
using Split_param = tvm::runtime::contrib::split_param;
using pool2d_param = tvm::runtime::contrib::pool2d_param;

// using Pad_param = tvm::runtime::contrib::Pad_param;

using IntegerArray = Array<Integer>;
using namespace backend;

size_t divRoundUp(size_t size, size_t word_size) { return (size + word_size - 1) / word_size; }

class CodegenAIPU : public ExprVisitor {
 public:
  explicit CodegenAIPU(const Function& func,
                       const Map<Expr, Array<IntegerArray>>& storage_device_map,
                       const Map<Expr, Array<IntegerArray>>& storage_input_output_map,
                       Map<Expr, Array<IntegerArray>>& aid_dtype_map,
                       std::map<int, size_t> temporary_data_storage,
                       const size_t& total_memory_used) {
    storage_device_map_ = storage_device_map;
    storage_input_output_map_ = storage_input_output_map;
    func_body = func->body;
    aid_dtype_map_ = aid_dtype_map;
    temporary_data_storage_ = temporary_data_storage;
    data_memory_used_ = total_memory_used;
  }
  CodegenAIPU() {}

  void OffSet_Push(Expr expr, int storage_id, Op_buffer_desc& data, int id_index = 0) {
    if (aid_dtype_map_.find(expr) != aid_dtype_map_.end()){
      uint32_t type_index = static_cast<uint32_t> (aid_dtype_map_[expr][0][id_index]);
      if (type_index == 5){ //int8
        data.datatype = datatype_enum::RINT8;
      }
      else if (type_index == 3){ //flaot32
        data.datatype = datatype_enum::RFLOAT;
      }
      else if (type_index == 1){ //int32
        data.datatype = datatype_enum::RINT;
      }
      else{
        LOG(INFO) << "type_index: " << type_index;
        LOG(INFO) << AsText(expr, false);
        LOG(FATAL) << "This type is inconsistent with our suppose.";
      }
    }
    else{
      LOG(FATAL) << "This expr is not in aid_dtype_!!!!";
    }
    if ((static_cast<uint32_t>(aid_dtype_map_[expr][1][id_index])) == 2) {  //dla
      data.line_stride = LINE_STRIDE(data.width);
      data.surf_stride = SURF_STRIDE(data.height, data.width);
    } else {
      data.line_stride = -1;
      data.surf_stride = -1;
    }
    data.size = static_cast<uint32_t>(storage_device_map_[expr][2][id_index]);

    size_t Get_Offset;
    if (temporary_data_offset_.find(storage_id) == temporary_data_offset_.end()) {
      Get_Offset = data_memory_used_;
      temporary_data_offset_.insert(std::pair<int, size_t>(storage_id, Get_Offset));
      data_memory_used_ += temporary_data_storage_[storage_id];
      //LOG_INFO << " off.end ";
      //LOG_INFO << " data_memory_used_ " << data_memory_used_;
      //LOG_INFO << " temporary_data_offset_ " << temporary_data_offset_.size();
      //LOG_INFO << " Get_Offset " << Get_Offset;
    } else {
      Get_Offset = temporary_data_offset_[storage_id];
      //LOG_INFO << " not off.end ";
      //LOG_INFO << "storage_id " << storage_id;
      //LOG_INFO << " data_memory_used_ " << data_memory_used_;
      //LOG_INFO << " temporary_data_offset_ " << temporary_data_offset_.size();
      //LOG_INFO << " Get_Offset " << Get_Offset;
    }

    // yuanyue 20220707
    auto iter = std::find_if(riscv_code_offset_.begin(),riscv_code_offset_.end(),[&](const std::pair<int32_t,size_t> &item)->bool
                              {return (item.second == temporary_data_offset_[storage_id]);});
    if (iter != riscv_code_offset_.end()){
      data.index = iter-riscv_code_offset_.begin();
    }
    else{
      data.index = riscv_code_offset_.size();
      riscv_code_offset_.push_back(std::pair<int32_t, size_t>(-1, temporary_data_offset_[storage_id]));
    }
    /*
    auto jt = expr_index_map_.find(expr);
    if (jt == expr_index_map_.end()) {
      data.index = riscv_code_offset_.size();
      uint32_t src_insert = data.index;
      expr_index_map_.insert(std::pair<Expr, uint32_t>(expr, src_insert));
      riscv_code_offset_.push_back(
          std::pair<int32_t, size_t>(-1, temporary_data_offset_[storage_id]));
    } else {
      data.index = jt->second;
    }
    */
  }

  void Weight_OffSet_Push(Expr expr, size_t weight_size) {
    auto weights = reinterpret_cast<void*>(new char[weight_size]);
    auto weight_const = Downcast<Constant>(expr);
    weight_offset_ -= weight_size;
    memcpy(weights, reinterpret_cast<void*>(weight_const->data->data), weight_size);
    riscv_code_offset_.push_back(
        std::pair<int32_t, size_t>(riscv_wt_list_.size() + weight_base_size_, weight_offset_));
    riscv_wt_list_.push_back(std::pair<size_t, void*>(weight_size, weights));
  }

  operator_enum set_op_type(const CallNode* call) {
    operator_enum op_type;
    if (call->op == tvm::relay::Op::Get("nn.conv2d")) {
      op_type = operator_enum::CONV2D;
    } else if (call->op == tvm::relay::Op::Get("nn.dense")) {
      op_type = operator_enum::DENSE;
    } else if (call->op == tvm::relay::Op::Get("nn.relu")) {
      op_type = operator_enum::RELU;
    } else if (call->op == tvm::relay::Op::Get("nn.softmax")) {
      op_type = operator_enum::SOFTMAX;
    } else if (call->op == tvm::relay::Op::Get("reshape")) {
      op_type = operator_enum::RESHAPE;
    } else if (call->op == tvm::relay::Op::Get("nn.avg_pool2d")) {
      op_type = operator_enum::POOL2D;
    } else if (call->op == tvm::relay::Op::Get("nn.max_pool2d")) {
      op_type = operator_enum::POOL2D;
    } else if (call->op == tvm::relay::Op::Get("add")) {
      op_type = operator_enum::ADD;
    } else if (call->op == tvm::relay::Op::Get("image.resize")) {
      op_type = operator_enum::RESIZE;
    } else if (call->op == tvm::relay::Op::Get("concatenate")) {
      op_type = operator_enum::CONCAT;
    } else if (call->op == tvm::relay::Op::Get("sigmoid")) {
      op_type = operator_enum::SIGMOID;
    } else if (call->op == tvm::relay::Op::Get("strided_slice")) {
      op_type = operator_enum::STRIDED_SLICE;
    } else if (call->op == tvm::relay::Op::Get("multiply")) {
      op_type = operator_enum::MULTIPLY;
    } else if (call->op == tvm::relay::Op::Get("exp")) {
      op_type = operator_enum::EXP;
    } else if (call->op == tvm::relay::Op::Get("expand_dims")) {
      op_type = operator_enum::EXPAND_DIMS;
    } else if (call->op == tvm::relay::Op::Get("take")) {
      op_type = operator_enum::TAKE;
    } else if (call->op == tvm::relay::Op::Get("one_hot")) {
      op_type = operator_enum::ONE_HOT;
    } else if (call->op == tvm::relay::Op::Get("less")) {
      op_type = operator_enum::LESS;
    } else if (call->op == tvm::relay::Op::Get("nn.batch_matmul")) {
      op_type = operator_enum::BATCH_MATMUL;
    } else if (call->op == tvm::relay::Op::Get("transpose")) {
      op_type = operator_enum::TRANSPOSE;
    } else if (call->op == tvm::relay::Op::Get("cast")) {
      op_type = operator_enum::CAST;
    } else if (call->op == tvm::relay::Op::Get("power")) {
      op_type = operator_enum::POWER;
    } else if (call->op == tvm::relay::Op::Get("divide")) {
      op_type = operator_enum::DIVIDE;
    } else if (call->op == tvm::relay::Op::Get("sqrt")) {
      op_type = operator_enum::SQRT;
    } else if (call->op == tvm::relay::Op::Get("erf")) {
      op_type = operator_enum::ERF;
    } else if (call->op == tvm::relay::Op::Get("tanh")) {
      op_type = operator_enum::TANH;
    } else if (call->op == tvm::relay::Op::Get("subtract")) {
      op_type = operator_enum::SUBTRACT;
    } else if (call->op == tvm::relay::Op::Get("sum")) {
      op_type = operator_enum::SUM;
    } else if (call->op == tvm::relay::Op::Get("mean")) {
      op_type = operator_enum::MEAN;
    } else if (call->op == tvm::relay::Op::Get("max")) {
      op_type = operator_enum::MAX;
    } else if (call->op == tvm::relay::Op::Get("min")) {
      op_type = operator_enum::MIN;
    } else if (call->op == tvm::relay::Op::Get("all")) {
      op_type = operator_enum::ALL;
    } else if (call->op == tvm::relay::Op::Get("any")) {
      op_type = operator_enum::ANY;
    } else if (call->op == tvm::relay::Op::Get("argmax")) {
      op_type = operator_enum::ARGMAX;
    } else if (call->op == tvm::relay::Op::Get("argmin")) {
      op_type = operator_enum::ARGMIN;
    } else if (call->op == tvm::relay::Op::Get("split")) {
      op_type = operator_enum::SPLIT;
    } else if (call->op == tvm::relay::Op::Get("mean")) {
      op_type = operator_enum::MEAN;
    } else if (call->op == tvm::relay::Op::Get("squeeze")) {
      op_type = operator_enum::SQUEEZE;
    } else if (call->op == tvm::relay::Op::Get("featuretoweight")) {
      op_type = operator_enum::FEATURETOWEIGHT;
    } else {
      LOG_INFO << "operators not listed " << call->op;
    }
    return op_type;
  }

  void common_parameters_get(const CallNode* callnode, Common_parameters& common) {
    common.op_type = set_op_type(callnode);
    uint32_t arg_input = 0;
    for (auto arg : callnode->args) {
      arg = GetNotQuantizedExpr(arg);
      if (arg->checked_type().as<TensorTypeNode>()) {
        arg_input += 1;
      } else if (arg->checked_type().as<TupleTypeNode>()) {
        auto type_node = arg->checked_type().as<TupleTypeNode>();
        for (auto field : type_node->fields) {
          arg_input += 1;
        }
      }
    }
    common.input_num = arg_input;
    auto call = GetRef<Expr>(callnode);
    auto type_node = call->checked_type().as<TupleTypeNode>();
    if (type_node) {
      common.output_num = type_node->fields.size();
    } else {
      common.output_num = 1;
    }
    
    for (size_t i=0; i<common.input_num; i++) {
      common.input_scale_factor[i] = iscale_[i];
      if(debug_info) {
        LOG(INFO) << "pass iscale_[" << i << "]: " << iscale_[i];
      }
    }
    
    common.output_scale_factor[0] = oscale_;
    if(debug_info) {
      LOG(INFO) << "pass oscale_: " << oscale_;
    }
  }

  void shape_size(std::vector<int> shape, Op_buffer_desc& data) {
    if (shape.size() == 2) {
      data.batchsize = 1;
      data.channel = shape[1];
      data.height = 1;
      data.width = shape[0];
    } else if (shape.size() == 3) {
      data.batchsize = 1;
      data.channel = shape[2];
      data.height = shape[0];
      data.width = shape[1];
    } else if (shape.size() == 4) {
      data.batchsize = shape[0];
      data.channel = shape[3];
      data.height = shape[1];
      data.width = shape[2];
    }
    else if (shape.size() == 1) {
      data.batchsize = 1;
      data.channel = shape[0];
      data.height = 1;
      data.width = 1;
    }
    else if (!shape.size()){
      data.batchsize = 1;
      data.channel = 1;
      data.height = 1;
      data.width = 1;
    }
    else{
      LOG(FATAL) << "riscv not suppose shape size: " << shape.size();
    }
  }

  std::vector<Type> shape_type_get(Expr expr) {
    std::vector<Type> tensor_check_type;
    if (expr->checked_type().as<TensorTypeNode>()) {
      Type CheckType = expr->checked_type();
      tensor_check_type.push_back(CheckType);
    } else if (expr->checked_type().as<TupleTypeNode>()) {
      auto type_node = expr->checked_type().as<TupleTypeNode>();
      bool tmp_flag_first = true;
      for (auto field : type_node->fields) {
        ICHECK(field->IsInstance<TensorTypeNode>());
        if (tmp_flag_first == true) {
          Type CheckType = field;
          tensor_check_type.push_back(CheckType);
          tmp_flag_first = false;
        } else {
          Type CheckType = field;
          tensor_check_type.push_back(CheckType);
        }
      }
    }
    return tensor_check_type;
  }

  Op_buffer_desc op_buffer_desc_get(Expr expr, Type tensor_check_type, Op_buffer_desc& data,
                                    int data_type, int id_index = 0) {
    const TensorTypeNode* type = tensor_check_type.as<TensorTypeNode>();
    std::vector<int> shape = GetShape(tensor_check_type);
    shape_size(shape, data);
    //LOG_INFO << "type " << tensor_check_type;
    //LOG_INFO << "shape.size " << shape.size();
    //LOG_INFO << "type " << type->dtype.bits() * type->dtype.lanes();
    //LOG_INFO << "expr " << AsText(expr, false);

    int storage_id = storage_device_map_[expr][0][id_index];
    //LOG_INFO << "storage_id " << storage_id;
    if (data_type == 0 || data_type == 2) {
      OffSet_Push(expr, storage_id, data,id_index);
    } else if (data_type == 1) {
      if (expr->IsInstance<CallNode>() || expr->IsInstance<VarNode>()) {
        OffSet_Push(expr, storage_id, data,id_index);
      } else {
        size_t weight_size = 1;
        for (size_t i = 0; i < shape.size(); i++) {
          weight_size *= static_cast<size_t>(shape[i]);
        }
        weight_size = weight_size * divRoundUp(type->dtype.bits() * type->dtype.lanes(), 8);
        data.size = weight_size;
        data.datatype = datatype_enum::RFLOAT;
        data.line_stride = -1;
        data.surf_stride = -1;
        data.index = riscv_code_offset_.size();
        Weight_OffSet_Push(expr, weight_size);
      }
    } else {
      LOG(FATAL) << "data_type should be 0(scr), 1(aux), 2(dst), bad data_type " << data_type;
    }

    return data;
  }

  bool broadcast_weight_buffer_desc_get(Expr aux_expr, std::vector<int> src_shape, Op_buffer_desc& aux_data) {
    auto wshape = GetShape(aux_expr->checked_type());
    auto wttype = aux_expr->checked_type().as<TensorTypeNode>();

    auto weight_const = Downcast<Constant>(aux_expr);
    size_t weight_size = 1;
    for (size_t i = 0; i < wshape.size(); i++) {
      weight_size *= static_cast<size_t>(wshape[i]);
      // LOG(INFO)<<"wshape[i]: "<<wshape[i];
    }
    for (size_t t =0; t< wshape.size()-4; t++){
      wshape.insert(wshape.begin(),1);
    }

    weight_size = weight_size * divRoundUp(wttype->dtype.bits() * wttype->dtype.lanes(), 8);
    size_t weight_size2 = storage_device_map_[aux_expr][2][0];
    ICHECK_EQ(weight_size, weight_size2);

    std::vector <int> broadcast_wshape=Broadcast_Shape(wshape,src_shape);
    size_t broadcast_weight_size = 1;
    for (size_t i = 0; i < broadcast_wshape.size(); i++) {
      // LOG(INFO)<<"broadcast_wshape[i]: "<<broadcast_wshape[i];
      broadcast_weight_size *= static_cast<size_t>(broadcast_wshape[i]);
    }
    broadcast_weight_size = broadcast_weight_size * divRoundUp(wttype->dtype.bits() * wttype->dtype.lanes(), 8);

    if (broadcast_weight_size == weight_size)
      return false; 
    //LOG(INFO)<< "--------------------------IN BROADCASTING----------------------------------";
    //LOG(INFO)<< "broadcast_weight_size: "<< broadcast_weight_size << "  weight_size:"<< weight_size;
    auto weights = (void*)(new char[weight_size]);
    memcpy(weights, (void*)(weight_const->data->data), weight_size);
    /*
    FILE* fp= fopen("1weight0.txt","a+");
    for (int i =0; i< wshape[2];i++){
      for (int j =0; j< wshape[3]; j++){
        //fputc(((float*)weights)[i*wshape[3]+j],fp);
        fprintf(fp,"%f",((float*)weights)[i*wshape[3]+j]);
      }
    }
    fclose(fp);
    */
    auto broadcast_weights = (void*)(new char[broadcast_weight_size]);
    Broadcast_weightdata(weights, broadcast_weights,wshape, broadcast_wshape, wttype); 
    /*
    FILE* fp2= fopen("1Broadcast_weightd0.txt","a+");
    for (int i =0; i< broadcast_wshape[2];i++){
      for (int j =0; j< broadcast_wshape[3]; j++){
        //fputc(((float*)weights)[i*wshape[3]+j],fp);
        fprintf(fp2,"%f",((float*)broadcast_weights)[i*broadcast_wshape[3]+j]);
      }
    }
    fclose(fp2);
    */
    shape_size(broadcast_wshape, aux_data);
    // LOG(INFO)<<"broadcast_weight_size: "<< broadcast_weight_size << " weight_size: " << weight_size;
    // LOG(INFO)<<"broadcast_wshape: "<< broadcast_wshape.size() << " weight_size: " << wshape.size();
    aux_data.datatype = datatype_enum::RFLOAT;
    aux_data.line_stride = -1;
    aux_data.surf_stride = -1;
    aux_data.index = riscv_code_offset_.size();
    aux_data.size = broadcast_weight_size;
    weight_offset_ -= broadcast_weight_size;  
    riscv_code_offset_.push_back(std::pair<int32_t, size_t>(riscv_wt_list_.size() + weight_base_size_, weight_offset_)); //yuanyue 0704
    riscv_wt_list_.push_back(std::pair<size_t, void*>(broadcast_weight_size, broadcast_weights)); //yuanyue 0704
    return true; 
  }

  // datatype_enum set_data_type(const TensorTypeNode* type, datatype_enum dy_type) {
  //   if (type->dtype.code() == DataType::kInt)
  //     dy_type = datatype_enum::RINT;
  //   else if (type->dtype.code() == DataType::kUInt)
  //     dy_type = datatype_enum::RUINT;
  //   else if (type->dtype.code() == DataType::kFloat)
  //     dy_type = datatype_enum::RFLOAT;
  //   else if (type->dtype.code() == DataType::kBFloat && type->dtype.bits() == 16)
  //     dy_type = datatype_enum::RBFLOAT;
  //   else
  //     LOG(FATAL) << "Datatype not supported ";
  //   return dy_type;
  // }

  void output_info(Common_parameters& common, Op_buffer_desc& src_data, Op_buffer_desc& dst_data) {
    LOG_INFO << "  ############ common: ";
    LOG_INFO << "op_type " << common.op_type;
    LOG_INFO << "input_num " << common.input_num;
    LOG_INFO << "input_scale " << common.input_scale_factor[0];
    LOG_INFO << "output_num " << common.output_num;
    LOG_INFO << "output_scale " << common.output_scale_factor[0];
    LOG_INFO << " ############src_data: ";
    LOG_INFO << "index " << src_data.index;
    LOG_INFO << "address " << (void *)src_data.address;
    LOG_INFO << "size " << src_data.size;
    LOG_INFO << "datatype " << src_data.datatype;
    LOG_INFO << "batchsize " << src_data.batchsize;
    LOG_INFO << "width " << src_data.width;
    LOG_INFO << "height " << src_data.height;
    LOG_INFO << "channel " << src_data.channel;
    LOG_INFO << "line " << src_data.line_stride;
    LOG_INFO << "surf " << src_data.surf_stride;
    LOG_INFO << " ############ dst_data: ";
    LOG_INFO << "index " << dst_data.index;
    LOG_INFO << "address " << (void *)dst_data.address;
    LOG_INFO << "size " << dst_data.size;
    LOG_INFO << "datatype " << dst_data.datatype;
    LOG_INFO << "batchsize " << dst_data.batchsize;
    LOG_INFO << "width " << dst_data.width;
    LOG_INFO << "height " << dst_data.height;
    LOG_INFO << "channel " << dst_data.channel;
    LOG_INFO << "line " << dst_data.line_stride;
    LOG_INFO << "surf " << dst_data.surf_stride;
  }

  void output_info_other(Op_buffer_desc& data) {
    LOG_INFO << " ############ data: ";
    LOG_INFO << "index " << data.index;
    LOG_INFO << "address " << (void *)data.address;
    LOG_INFO << "size " << data.size;
    LOG_INFO << "datatype " << data.datatype;
    LOG_INFO << "batchsize " << data.batchsize;
    LOG_INFO << "width " << data.width;
    LOG_INFO << "height " << data.height;
    LOG_INFO << "channel " << data.channel;
    LOG_INFO << "line " << data.line_stride;
    LOG_INFO << "surf " << data.surf_stride;
  }

  void weight_info(Op_buffer_desc& weight) {
    LOG_INFO << "riscv_wt_list_ " << riscv_wt_list_.size();
    //for (unsigned int i = 0; i < riscv_wt_list_.size(); i++) {
    int i = riscv_code_offset_[weight.index].first;
    if (i >= 0) {
      LOG_INFO << "i " << i;
      LOG_INFO << "riscv_wt_list_[i].first " << riscv_wt_list_[i].first;
      LOG_INFO << "riscv_wt_list_[i].second " << riscv_wt_list_[i].second;
    }
    //auto weight_data = tvm::runtime::NDArray::Empty({weight.width, weight.channel},
    //                                                DLDataType{kDLFloat, 32, 1}, {kDLCPU, 0});
    //weight_data.tvm::runtime::NDArray::CopyFromBytes(riscv_wt_list_[i].second,
    //                                                  riscv_wt_list_[i].first);
    //auto weight_constant = relay::Constant(weight_data);
    //LOG_INFO << "weight " << AsText(weight_constant, false);
    //LOG_INFO << "weight " << weight_constant;
    //}
    LOG_INFO << "riscv_code_offset_ " << riscv_code_offset_.size();
    //for (unsigned int i = 0; i < riscv_code_offset_.size(); i++) {
    //  LOG_INFO << "i " << i;
    //  LOG_INFO << "riscv_code_offset_[i].first " << riscv_code_offset_[i].first;
    //  LOG_INFO << "riscv_code_offset_[i].second " << riscv_code_offset_[i].second;
    //}
  }

  Cpu_param* common_only_op(const CallNode* call) {
    Cpu_param* cpu_param = new Cpu_param();
    Common_parameters common;
    common_parameters_get(call, common);
    cpu_param->cpu_operation.common_only_op.common = common;

    // op_buffer_desc
    Op_buffer_desc src_data;
    Op_buffer_desc dst_data;
    Expr src_expr = GetNotQuantizedExpr(call->args[0]);  // yuanyue 20220630
    Expr dst_expr = GetRef<Expr>(call);

    if (src_expr->checked_type().as<TupleTypeNode>())
      LOG(FATAL) << "common_only_op can't has tuple input";
    std::vector<Type> src_tensor_check_type;
    src_tensor_check_type = shape_type_get(src_expr);
    src_data = op_buffer_desc_get(src_expr, src_tensor_check_type[0], src_data, 0);

    std::vector<Type> dst_tensor_check_type;
    dst_tensor_check_type = shape_type_get(dst_expr);
    dst_data = op_buffer_desc_get(dst_expr, dst_tensor_check_type[0], dst_data, 2);

    // correction for op featuretoweight
    if(call->op == Op::Get("featuretoweight")) {
      std::vector<int> shape = GetShape(src_tensor_check_type[0]);
      if(shape.size() == 2) {
          src_data.batchsize = shape[0];
          src_data.channel = shape[1];
          src_data.height = 1;
          src_data.width = 1;

          dst_data.batchsize = shape[0];
          dst_data.channel = shape[1];
          dst_data.height = 1;
          dst_data.width = 1;
      }
    }

    cpu_param->cpu_operation_buffer.common_only_op_buffers.src_data = src_data;
    cpu_param->cpu_operation_buffer.common_only_op_buffers.dst_data = dst_data;

    if (debug_info)
      output_info(cpu_param->cpu_operation.common_only_op.common,
                  cpu_param->cpu_operation_buffer.common_only_op_buffers.src_data,
                  cpu_param->cpu_operation_buffer.common_only_op_buffers.dst_data);
    return cpu_param;
  }

  Cpu_param* with_weight_op(const CallNode* call) {
    Cpu_param* cpu_param = new Cpu_param();
    Common_parameters common;
    common_parameters_get(call, common);
    cpu_param->cpu_operation.with_weight_op.common = common;
    float weight_ioscale;
    uint32_t sub_op_type = 1;
    cpu_param->cpu_operation.with_weight_op.weight_ioscale = weight_ioscale;

    Op_buffer_desc src_data;
    Op_buffer_desc aux_data;
    Op_buffer_desc dst_data;
    Expr src_expr = GetNotQuantizedExpr(call->args[0]);
    Expr aux_expr = GetNotQuantizedExpr(call->args[1]);
    Expr dst_expr = GetRef<Expr>(call);

    //cpu_param->cpu_operation.with_weight_op.sub_op_type = sub_op_type;
    std::vector<Type> src_tensor_check_type;
    src_tensor_check_type = shape_type_get(src_expr);
    src_data = op_buffer_desc_get(src_expr, src_tensor_check_type[0], src_data, 1);

    bool broadcasting = false;
    if ( RISCV_BROADCAST_ON ){
      operator_enum op_type = cpu_param->cpu_operation.with_weight_op.common.op_type;
      if (op_type == operator_enum::ADD || op_type == operator_enum::SUBTRACT || op_type == operator_enum::MULTIPLY ||
          op_type == operator_enum::DIVIDE || op_type == operator_enum::POWER || op_type == operator_enum::LESS){
        std::vector <int> src_shape;
        src_shape.push_back(src_data.batchsize);
        src_shape.push_back(src_data.height);
        src_shape.push_back(src_data.width);
        src_shape.push_back(src_data.channel);
        broadcasting = broadcast_weight_buffer_desc_get(aux_expr, src_shape, aux_data);
      }
    }
    if ( !broadcasting ){
      //LOG(INFO)<< "--------------------------NON BROADCASTING----------------------------------";
      std::vector<Type> aux_tensor_check_type;
      aux_tensor_check_type = shape_type_get(aux_expr);
      aux_data = op_buffer_desc_get(aux_expr, aux_tensor_check_type[0], aux_data, 1);
    }
    if (aux_data.batchsize ==1 && aux_data.height ==1 && aux_data.width ==1){
      if (aux_data.channel ==1){
        cpu_param->cpu_operation.with_weight_op.sub_op_type = 0;
      }
      else{
        cpu_param->cpu_operation.with_weight_op.sub_op_type = 1;
      }
    }
    else{
      cpu_param->cpu_operation.with_weight_op.sub_op_type = 2; 
    }
    std::vector<Type> dst_tensor_check_type;
    dst_tensor_check_type = shape_type_get(dst_expr);
    dst_data = op_buffer_desc_get(dst_expr, dst_tensor_check_type[0], dst_data, 2);
    cpu_param->cpu_operation.with_weight_op.weight = aux_data;
    cpu_param->cpu_operation_buffer.with_weight_op_buffers.src_data = src_data;
    cpu_param->cpu_operation_buffer.with_weight_op_buffers.dst_data = dst_data;
    if (debug_info)
      output_info(cpu_param->cpu_operation.with_weight_op.common,
                  cpu_param->cpu_operation_buffer.with_weight_op_buffers.src_data,
                  cpu_param->cpu_operation_buffer.with_weight_op_buffers.dst_data);
    if (debug_info) output_info_other(cpu_param->cpu_operation.with_weight_op.weight);
    if (debug_info) weight_info(cpu_param->cpu_operation.with_weight_op.weight);
    return cpu_param;
  }

  Cpu_param* transform_op(const CallNode* call) {
    Cpu_param* cpu_param = new Cpu_param();
    Common_parameters common;
    common_parameters_get(call, common);
    cpu_param->cpu_operation.transform_op.common = common;
    
    // op_buffer_desc
    Op_buffer_desc src_data;
    Op_buffer_desc dst_data;
    Expr src_expr = GetNotQuantizedExpr(call->args[0]);
    Expr dst_expr = GetRef<Expr>(call);
    std::vector<Type> src_tensor_check_type;
    src_tensor_check_type = shape_type_get(src_expr);
    src_data = op_buffer_desc_get(src_expr, src_tensor_check_type[0], src_data, 0);
    std::vector<Type> dst_tensor_check_type;
    dst_tensor_check_type = shape_type_get(dst_expr);
    dst_data = op_buffer_desc_get(dst_expr, dst_tensor_check_type[0], dst_data, 2);
    cpu_param->cpu_operation_buffer.transform_op_buffers.src_data = src_data;
    cpu_param->cpu_operation_buffer.transform_op_buffers.dst_data = dst_data;


    const int ndim = src_tensor_check_type[0].as<TensorTypeNode>()->shape.size();
    if (call->op == tvm::relay::Op::Get("squeeze")) {
      const auto* squeezeAttrs = call->attrs.as<tvm::relay::SqueezeAttrs>();
      size_t i = 0;
      if (squeezeAttrs->axis.defined()){
        for (; i < squeezeAttrs->axis.size(); i++){
          int64_t axis = squeezeAttrs->axis[i];
          ICHECK(-ndim <= axis && axis < ndim)
            << "transpose only allows each `axis` in `axes` in range [-data.ndim, data.ndim)"
            << ", but got axis = " << axis << ", and data.ndim = " << ndim;
          axis = axis < 0 ? axis + ndim : axis;
          cpu_param->cpu_operation.transform_op.axis[i] = axis + 4 - ndim;
        }
      }
      for (; i < 4 ; i++){
        cpu_param->cpu_operation.transform_op.axis[i] = -1;
      }
    } else if (call->op == tvm::relay::Op::Get("transpose")) {
      const auto* transposeAttrs = call->attrs.as<TransposeAttrs>();
      ICHECK(ndim == transposeAttrs->axes.size())  
        << "transpose only allows ndim equal to transposeAttrs->axes.size()"
        << ", but got data.ndim = " << ndim << ", and transposeAttrs->axes.size() = " << transposeAttrs->axes.size();
      size_t i = 0;
      for (;i< (4 - ndim);i++){
        cpu_param->cpu_operation.transform_op.axis[i] = i;
      }
      for (; i < 4; i++){
        int64_t axis = transposeAttrs->axes[i + ndim - 4];
        ICHECK(-ndim <= axis && axis < ndim)
          << "transpose only allows each `axis` in `axes` in range [-data.ndim, data.ndim)"
          << ", but got axis = " << axis << ", and data.ndim = " << ndim;
        axis = axis < 0 ? axis + ndim : axis;
        cpu_param->cpu_operation.transform_op.axis[i] = axis + 4 - ndim;
      }
    }

    if (debug_info)
      output_info(cpu_param->cpu_operation.transform_op.common,
                  cpu_param->cpu_operation_buffer.transform_op_buffers.src_data,
                  cpu_param->cpu_operation_buffer.transform_op_buffers.dst_data);
    return cpu_param;
  }

  Cpu_param* split_op(const CallNode* call) {
    Cpu_param* cpu_param = new Cpu_param();
    Common_parameters common;
    common_parameters_get(call, common);
    cpu_param->cpu_operation.split_op.common = common;
    
    // op_buffer_desc
    Op_buffer_desc src_data;
    Op_buffer_desc dst_data[INPUT_MAX];
    Expr src_expr = GetNotQuantizedExpr(call->args[0]);
    Expr dst_expr = GetRef<Expr>(call);
    std::vector<Type> src_tensor_check_type;
    src_tensor_check_type = shape_type_get(src_expr);
    src_data = op_buffer_desc_get(src_expr, src_tensor_check_type[0], src_data, 0);
    std::vector<Type> dst_tensor_check_type;
    dst_tensor_check_type = shape_type_get(dst_expr);

    const int ndim = src_tensor_check_type[0].as<TensorTypeNode>()->shape.size();
    const auto* SplitAttrs = call->attrs.as<tvm::relay::SplitAttrs>();
    int64_t axis = SplitAttrs->axis ;
    axis = axis < 0 ? axis + ndim : axis;
    cpu_param->cpu_operation.split_op.axis = axis + 4 - ndim;
    //LOG(INFO) <<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
    //LOG(INFO) << "SplitAttrs->axis: " << SplitAttrs->axis << " axis: "<<cpu_param->cpu_operation.split_op.axis << " ndim:" << ndim;

    int n = 0;
    if (SplitAttrs->indices_or_sections->IsInstance<ConstantNode>()) {
      //LOG(INFO) << "indices_or_sections AS ConstantNode";
      Constant splitconst = Downcast<Constant>(SplitAttrs->indices_or_sections);
      if (splitconst->is_scalar()) {
        cpu_param->cpu_operation.split_op.indices[n] =
            *(reinterpret_cast<int*>(splitconst->data->data));
        n++;
      } else {
        auto wshape = GetShape(splitconst->checked_type());
        //LOG(INFO) << "wshape.size()" << wshape.size();
        int weightsize = wshape[0];
        int* weightdata = reinterpret_cast<int*>(malloc(weightsize * sizeof(int)));
        memcpy(weightdata, reinterpret_cast<int*>((splitconst->data->data)),
               weightsize * sizeof(int));
        for (int i = 0; i < wshape[0]; i++) {
          cpu_param->cpu_operation.split_op.indices[n] = weightdata[i];
          n++;
        }
      }
    } else if (SplitAttrs->indices_or_sections->IsInstance<IntImmNode>()) {
      //LOG(INFO) << "indices_or_sections AS IntImmNode";
      cpu_param->cpu_operation.split_op.indices[n] =
          SplitAttrs->indices_or_sections.as<IntImmNode>()->value;
      n++;
    } else if (SplitAttrs->indices_or_sections->IsInstance<TupleNode>()) {
      //LOG(INFO) << "indices_or_sections AS TupleNode";
      auto tuple_node = SplitAttrs->indices_or_sections.as<TupleNode>();
      for (const auto& field : tuple_node->fields) {
        cpu_param->cpu_operation.split_op.indices[n] = field.as<IntImmNode>()->value;
        n++;
      }
    } else {
      LOG(FATAL) << "AIPU RISCV Split doesn't support: "
                 << SplitAttrs->indices_or_sections->GetTypeKey();
    }
    // yuanyue 20220708  invaild value set to -1
    for (; n< INPUT_MAX ; n++){
      cpu_param->cpu_operation.split_op.indices[n] = -1;
    }

    for (unsigned int i = 0; i < dst_tensor_check_type.size(); i++) {
      dst_data[i] = op_buffer_desc_get(dst_expr, dst_tensor_check_type[i], dst_data[i], 2, i);
      cpu_param->cpu_operation.split_op.dst_data[i] = dst_data[i];
    }
    cpu_param->cpu_operation_buffer.split_buffers.dst_data = dst_data[0];
    cpu_param->cpu_operation_buffer.split_buffers.src_data = src_data;

    if (debug_info)
      output_info(cpu_param->cpu_operation.split_op.common,
                  cpu_param->cpu_operation_buffer.split_buffers.src_data,
                  cpu_param->cpu_operation_buffer.split_buffers.dst_data);
    if (debug_info) output_info_other(cpu_param->cpu_operation.split_op.dst_data[0]);
    return cpu_param;
  }

  Cpu_param* expand_dims_op(const CallNode* call) {
    Cpu_param* cpu_param = new Cpu_param();
    Common_parameters common;
    common_parameters_get(call, common);
    cpu_param->cpu_operation.expand_dims_op.common = common;

     // op_buffer_desc
    Op_buffer_desc src_data;
    Op_buffer_desc dst_data;
    Expr src_expr = GetNotQuantizedExpr(call->args[0]);
    Expr dst_expr = GetRef<Expr>(call);
    std::vector<Type> src_tensor_check_type;
    src_tensor_check_type = shape_type_get(src_expr);
    src_data = op_buffer_desc_get(src_expr, src_tensor_check_type[0], src_data, 0);
    std::vector<Type> dst_tensor_check_type;
    dst_tensor_check_type = shape_type_get(dst_expr);
    dst_data = op_buffer_desc_get(dst_expr, dst_tensor_check_type[0], dst_data, 2);

    const int ndim = dst_tensor_check_type[0].as<TensorTypeNode>()->shape.size();
    const auto* expandDimsAttrs = call->attrs.as<ExpandDimsAttrs>();
    int64_t axis = expandDimsAttrs->axis ;
    axis = axis < 0 ? axis + ndim : axis;
    cpu_param->cpu_operation.expand_dims_op.axis = axis + 4 - ndim;
    cpu_param->cpu_operation.expand_dims_op.num_newaxis = expandDimsAttrs->num_newaxis;
    //LOG(INFO) <<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
    //LOG(INFO) << " expandDimsAttrs->axis: " << expandDimsAttrs->axis << " axis: " <<  cpu_param->cpu_operation.expand_dims_op.axis <<" ndim:" << ndim;
   
    cpu_param->cpu_operation_buffer.expand_dims_buffers.src_data = src_data;
    cpu_param->cpu_operation_buffer.expand_dims_buffers.dst_data = dst_data;

    if (debug_info)
      output_info(cpu_param->cpu_operation.expand_dims_op.common,
                  cpu_param->cpu_operation_buffer.expand_dims_buffers.src_data,
                  cpu_param->cpu_operation_buffer.expand_dims_buffers.dst_data);
    return cpu_param;
  }

  Cpu_param* concat_op(const CallNode* call) {
    Cpu_param* cpu_param = new Cpu_param();
    Common_parameters common;
    common_parameters_get(call, common);
    cpu_param->cpu_operation.concat_op.common = common;
    
    // op_buffer_desc
    Op_buffer_desc src_data[INPUT_MAX];
    Op_buffer_desc dst_data;
    Expr src_expr = GetNotQuantizedExpr(call->args[0]);
    Expr dst_expr = GetRef<Expr>(call);
    std::vector<Type> src_tensor_check_type;
    src_tensor_check_type = shape_type_get(src_expr);
    for (unsigned int i = 0; i < src_tensor_check_type.size(); i++) {
      src_data[i] = op_buffer_desc_get(src_expr, src_tensor_check_type[i], src_data[i], 0,i);
      cpu_param->cpu_operation.concat_op.src_data[i] = src_data[i];
    }
    std::vector<Type> dst_tensor_check_type;
    dst_tensor_check_type = shape_type_get(dst_expr);
    dst_data = op_buffer_desc_get(dst_expr, dst_tensor_check_type[0], dst_data, 2);

    const int ndim = src_tensor_check_type[0].as<TensorTypeNode>()->shape.size();
    const auto* concatenateAttrs = call->attrs.as<ConcatenateAttrs>();
    int64_t axis = concatenateAttrs->axis;
    axis = axis < 0 ? axis + ndim : axis;
    cpu_param->cpu_operation.concat_op.axis = axis + 4 - ndim;
    //LOG(INFO) << " concatenateAttrs->axis: " << concatenateAttrs->axis << " axis: " <<  cpu_param->cpu_operation.concat_op.axis <<" ndim:" << ndim;

    cpu_param->cpu_operation_buffer.concat_buffers.src_data = src_data[0];
    cpu_param->cpu_operation_buffer.concat_buffers.dst_data = dst_data;

    if (debug_info)
      output_info(cpu_param->cpu_operation.concat_op.common,
                  cpu_param->cpu_operation_buffer.concat_buffers.src_data,
                  cpu_param->cpu_operation_buffer.concat_buffers.dst_data);
    if (debug_info) output_info_other(cpu_param->cpu_operation.concat_op.src_data[0]);
    return cpu_param;
  }

  Cpu_param* reduce_op(const CallNode* call) {
    Cpu_param* cpu_param = new Cpu_param();
    Common_parameters common;
    common_parameters_get(call, common);
    cpu_param->cpu_operation.reduce_op.common = common;

    // op_buffer_desc
    Op_buffer_desc src_data;
    Op_buffer_desc dst_data;
    Expr src_expr = GetNotQuantizedExpr(call->args[0]);
    Expr dst_expr = GetRef<Expr>(call);
    std::vector<Type> src_tensor_check_type;
    src_tensor_check_type = shape_type_get(src_expr);
    src_data = op_buffer_desc_get(src_expr, src_tensor_check_type[0], src_data, 0);
    std::vector<Type> dst_tensor_check_type;
    dst_tensor_check_type = shape_type_get(dst_expr);
    dst_data = op_buffer_desc_get(dst_expr, dst_tensor_check_type[0], dst_data, 2);

    const auto* reduceAttrs = call->attrs.as<ReduceAttrs>();
    const int ndim = src_tensor_check_type[0].as<TensorTypeNode>()->shape.size();
    for (int n = 0; n < 4 ; n++){
      cpu_param->cpu_operation.reduce_op.axis[n] = -1;
    }
    if (!reduceAttrs->axis.empty()){
      for (int n = 0; n < reduceAttrs->axis.size(); n++){
        int64_t axis = reduceAttrs->axis[n];
        ICHECK(-ndim <= axis && axis < ndim)
          << "transpose only allows each `axis` in `axes` in range [-data.ndim, data.ndim)"
          << ", but got axis = " << axis << ", and data.ndim = " << ndim;
        axis = axis < 0 ? axis + ndim : axis;
        axis = axis + 4 - ndim;
        if (axis <4 && axis>-1){
          cpu_param->cpu_operation.reduce_op.axis[axis] = axis;
        }
        else {
          LOG(FATAL)<< "axis value is error !" << axis;
        }
        //LOG(INFO) <<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
        //LOG(INFO) << " reduceAttrs->axis: " << reduceAttrs->axis[n] << " axis: " <<  cpu_param->cpu_operation.reduce_op.axis[n] <<" ndim:" << ndim;
      }
    }
    
    cpu_param->cpu_operation.reduce_op.exclude = reduceAttrs->exclude;
    cpu_param->cpu_operation.reduce_op.keepdims = reduceAttrs->keepdims;
    
    cpu_param->cpu_operation_buffer.reduce_buffers.src_data = src_data;
    cpu_param->cpu_operation_buffer.reduce_buffers.dst_data = dst_data;

    if (debug_info)
      output_info(cpu_param->cpu_operation.reduce_op.common,
                  cpu_param->cpu_operation_buffer.reduce_buffers.src_data,
                  cpu_param->cpu_operation_buffer.reduce_buffers.dst_data);
    return cpu_param;
  }

  Cpu_param* cast_op(const CallNode* call) {
    Cpu_param* cpu_param = new Cpu_param();
    Common_parameters common;
    common_parameters_get(call, common);
    cpu_param->cpu_operation.cast_op.common = common;
    const auto* castAttrs = call->attrs.as<CastAttrs>();
    if (castAttrs->dtype.is_int())
      cpu_param->cpu_operation.cast_op.datatype = datatype_enum::RINT;
    else if (castAttrs->dtype.is_uint())
      cpu_param->cpu_operation.cast_op.datatype = datatype_enum::RUINT;
    else if (castAttrs->dtype.is_float())
      cpu_param->cpu_operation.cast_op.datatype = datatype_enum::RFLOAT;
    else if (castAttrs->dtype.is_bfloat16())
      cpu_param->cpu_operation.cast_op.datatype = datatype_enum::RBFLOAT;
    else
      cpu_param->cpu_operation.cast_op.datatype = datatype_enum::RINT8;

    // op_buffer_desc
    Op_buffer_desc src_data;
    Op_buffer_desc dst_data;
    Expr src_expr = GetNotQuantizedExpr(call->args[0]);
    Expr dst_expr = GetRef<Expr>(call);
    std::vector<Type> src_tensor_check_type;
    src_tensor_check_type = shape_type_get(src_expr);
    src_data = op_buffer_desc_get(src_expr, src_tensor_check_type[0], src_data, 0);
    std::vector<Type> dst_tensor_check_type;
    dst_tensor_check_type = shape_type_get(dst_expr);
    dst_data = op_buffer_desc_get(dst_expr, dst_tensor_check_type[0], dst_data, 2);
    cpu_param->cpu_operation_buffer.cast_buffers.src_data = src_data;
    cpu_param->cpu_operation_buffer.cast_buffers.dst_data = dst_data;

    if (debug_info)
      output_info(cpu_param->cpu_operation.cast_op.common,
                  cpu_param->cpu_operation_buffer.cast_buffers.src_data,
                  cpu_param->cpu_operation_buffer.cast_buffers.dst_data);
    return cpu_param;
  }

  Cpu_param* one_hot_op(const CallNode* call) {
    Cpu_param* cpu_param = new Cpu_param();
    Common_parameters common;
    common_parameters_get(call, common);
    // we make on_value and off_value internal, so the input num is 1
    common.input_num = 1;
    cpu_param->cpu_operation.one_hot_op.common = common;
    
    // op_buffer_desc
    Op_buffer_desc src_data;
    Op_buffer_desc dst_data;
    Expr src_expr = GetNotQuantizedExpr(call->args[0]);
    Expr dst_expr = GetRef<Expr>(call);
    std::vector<Type> src_tensor_check_type;
    src_tensor_check_type = shape_type_get(src_expr);
    src_data = op_buffer_desc_get(src_expr, src_tensor_check_type[0], src_data, 0);
    std::vector<Type> dst_tensor_check_type;
    dst_tensor_check_type = shape_type_get(dst_expr);
    dst_data = op_buffer_desc_get(dst_expr, dst_tensor_check_type[0], dst_data, 2);
    cpu_param->cpu_operation_buffer.one_hot_buffers.src_data = src_data;
    cpu_param->cpu_operation_buffer.one_hot_buffers.dst_data = dst_data;

    
    const int ndim = src_tensor_check_type[0].as<TensorTypeNode>()->shape.size();
    const auto* oneHotAttrs = call->attrs.as<OneHotAttrs>();
    int64_t axis = oneHotAttrs->axis ;
    //shape size from n to n+1 //yuanyue 
    int true_axis = (axis == -1) ? ndim : axis;
    cpu_param->cpu_operation.one_hot_op.axis = true_axis + 4 - (ndim + 1); 
    cpu_param->cpu_operation.one_hot_op.depth = oneHotAttrs->depth;
    Constant on_value = Downcast<Constant>(GetNotQuantizedExpr(call->args[1]));
    Constant off_value = Downcast<Constant>(GetNotQuantizedExpr(call->args[2]));
    cpu_param->cpu_operation.one_hot_op.on_value =
        (reinterpret_cast<float*>(on_value->data->data))[0];
    cpu_param->cpu_operation.one_hot_op.off_value =
        (reinterpret_cast<float*>(off_value->data->data))[0];  //yuanyue


    if (debug_info)
      output_info(cpu_param->cpu_operation.one_hot_op.common,
                  cpu_param->cpu_operation_buffer.one_hot_buffers.src_data,
                  cpu_param->cpu_operation_buffer.one_hot_buffers.dst_data);
    return cpu_param;
  }

  Cpu_param* take_op(const CallNode* call) {
    Cpu_param* cpu_param = new Cpu_param();
    Common_parameters common;
    common_parameters_get(call, common);
    cpu_param->cpu_operation.take_op.common = common;
    
    Op_buffer_desc src_data_take;
    Expr src_expr_take;
    src_expr_take = GetNotQuantizedExpr(call->args[1]);
    std::vector<Type> src2_tensor_check_type;
    src2_tensor_check_type = shape_type_get(src_expr_take);
    src_data_take = op_buffer_desc_get(src_expr_take, src2_tensor_check_type[0], src_data_take, 1);
    cpu_param->cpu_operation.take_op.indices = src_data_take;

    // op_buffer_desc
    Op_buffer_desc src_data;
    Op_buffer_desc dst_data;
    Expr src_expr = GetNotQuantizedExpr(call->args[0]);
    Expr dst_expr = GetRef<Expr>(call);
    std::vector<Type> src_tensor_check_type;
    src_tensor_check_type = shape_type_get(src_expr);
    src_data = op_buffer_desc_get(src_expr, src_tensor_check_type[0], src_data, 1);
    std::vector<Type> dst_tensor_check_type;
    dst_tensor_check_type = shape_type_get(dst_expr);
    dst_data = op_buffer_desc_get(dst_expr, dst_tensor_check_type[0], dst_data, 2);
    cpu_param->cpu_operation_buffer.take_buffers.src_data = src_data;
    cpu_param->cpu_operation_buffer.take_buffers.dst_data = dst_data;

    const auto* takeAttrs = call->attrs.as<TakeAttrs>();
    if (!takeAttrs->axis.defined()) {
      cpu_param->cpu_operation.take_op.axis = -1;
    }
    else{
      const int ndim = src_tensor_check_type[0].as<TensorTypeNode>()->shape.size();
      int64_t axis = takeAttrs->axis;
      axis = axis < 0 ? axis + ndim : axis;
      cpu_param->cpu_operation.take_op.axis =  axis + 4 - ndim;
    }

    auto s_name = takeAttrs->mode;
    if (s_name =="clip"){
      cpu_param->cpu_operation.take_op.take_mode = take_mode_enum::CLIP;
    }
    else if (s_name =="fast"){
      cpu_param->cpu_operation.take_op.take_mode = take_mode_enum::FAST;
    }
    else if (s_name =="wrap"){
      cpu_param->cpu_operation.take_op.take_mode = take_mode_enum::WRAP;
    }
    else {
      LOG(FATAL) <<"the take mode is not listed, its name is: "  <<s_name <<"!";
    }

    if (debug_info)
      output_info(cpu_param->cpu_operation.take_op.common,
                  cpu_param->cpu_operation_buffer.take_buffers.src_data,
                  cpu_param->cpu_operation_buffer.take_buffers.dst_data);
    if (debug_info) output_info_other(cpu_param->cpu_operation.take_op.indices);
    if (debug_info) weight_info(cpu_param->cpu_operation_buffer.take_buffers.src_data);

    return cpu_param;
  }

  Cpu_param* slice_op(const CallNode* call) {
    Cpu_param* cpu_param = new Cpu_param();
    Common_parameters common;
    common_parameters_get(call, common);
    cpu_param->cpu_operation.slice_op.common = common;
    const auto* stridedSliceAttrs = call->attrs.as<StridedSliceAttrs>();
    
    // op_buffer_desc
    Op_buffer_desc src_data;
    Op_buffer_desc dst_data;
    Expr src_expr = GetNotQuantizedExpr(call->args[0]);
    Expr dst_expr = GetRef<Expr>(call);
    std::vector<Type> src_tensor_check_type;
    src_tensor_check_type = shape_type_get(src_expr);
    src_data = op_buffer_desc_get(src_expr, src_tensor_check_type[0], src_data, 0);
    std::vector<Type> dst_tensor_check_type;
    dst_tensor_check_type = shape_type_get(dst_expr);
    dst_data = op_buffer_desc_get(dst_expr, dst_tensor_check_type[0], dst_data, 2);
    
    const int ndim = src_tensor_check_type[0].as<TensorTypeNode>()->shape.size();
	
    for (int n = 0; n < 4 ; n++){
      cpu_param->cpu_operation.slice_op.begin[n] = -1;
      cpu_param->cpu_operation.slice_op.end[n] = -1;
      cpu_param->cpu_operation.slice_op.stride[n] = -1;
    }
	
    if (stridedSliceAttrs->begin != nullptr){
      for (int n = 0; n < 4; n++){
        if (n < 4 - ndim && n > -1){
          cpu_param->cpu_operation.slice_op.begin[n] = 0;
        }
        else if (n < 4 && n >= 4 - ndim){
          int64_t begin = (stridedSliceAttrs->begin.value())[n - (4 - ndim)];
          cpu_param->cpu_operation.slice_op.begin[n] = begin;
        }
        else {
          LOG(FATAL)<< "begin value is error !";
        }
      }
    }
	
    if (stridedSliceAttrs->end != nullptr){
      for (int n = 0; n < 4; n++){
        if (n < 4 - ndim && n > -1){
          cpu_param->cpu_operation.slice_op.end[n] = 1;
        }
        else if (n < 4 && n >= 4 - ndim){
          int64_t end = (stridedSliceAttrs->end.value())[n - (4 - ndim)];
          cpu_param->cpu_operation.slice_op.end[n] = end;
        }
        else {
          LOG(FATAL)<< "end value is error !";
        }
      }
    }
	
    if (stridedSliceAttrs->strides != nullptr){
      for (int n = 0; n < 4; n++){
        if (n < 4 - ndim && n > -1){
          cpu_param->cpu_operation.slice_op.stride[n] = 1;
        }
        else if (n < 4 && n >= 4 - ndim){
          int64_t strides = (stridedSliceAttrs->strides.value())[n - (4 - ndim)];
          cpu_param->cpu_operation.slice_op.stride[n] = strides;
        }
        else {
          LOG(FATAL)<< "strides value is error !";
        }
      }
    }
	
    if (stridedSliceAttrs->slice_mode == "end") {
      cpu_param->cpu_operation.slice_op.slice_mode = stride_slice_mode_enum::END;
    }
	
    cpu_param->cpu_operation_buffer.slice_buffers.src_data = src_data;
    cpu_param->cpu_operation_buffer.slice_buffers.dst_data = dst_data;

    if (debug_info)
      output_info(cpu_param->cpu_operation.slice_op.common,
                  cpu_param->cpu_operation_buffer.slice_buffers.src_data,
                  cpu_param->cpu_operation_buffer.slice_buffers.dst_data);
    return cpu_param;
  }

  Cpu_param* resize_op(const CallNode* call) {
    Cpu_param* cpu_param = new Cpu_param();
    Common_parameters common;
    common_parameters_get(call, common);
    cpu_param->cpu_operation.resize_op.common = common;
    const auto* resizeAttrs = call->attrs.as<ResizeAttrs>();
    if (resizeAttrs->layout == "NCHW") {
      cpu_param->cpu_operation.resize_op.layout = resize_layout_enum::NCHW;
    } else if (resizeAttrs->layout == "NHWC") {
      cpu_param->cpu_operation.resize_op.layout = resize_layout_enum::NHWC;
    } else {
      LOG_FATAL << "unsupported layout " << resizeAttrs->layout;
    }
    if (resizeAttrs->method == "nearest_neighbor") {
      cpu_param->cpu_operation.resize_op.method = resize_method_enum::NEAREST;
    } else if (resizeAttrs->method == "bilinear") {
      cpu_param->cpu_operation.resize_op.method = resize_method_enum::BILINEAR;
    } else if (resizeAttrs->method == "bicubic") {
      cpu_param->cpu_operation.resize_op.method = resize_method_enum::BICUBIC;
    } else {
      LOG_FATAL << "unsupported method " << resizeAttrs->method;
    }
    if (resizeAttrs->coordinate_transformation_mode == "half_pixel") {
      cpu_param->cpu_operation.resize_op.coordinate_transf_mode =
          resize_coordinate_transformation_mode_enum::HALF_PIXEL;
    } else if (resizeAttrs->coordinate_transformation_mode == "align_corners") {
      cpu_param->cpu_operation.resize_op.coordinate_transf_mode =
          resize_coordinate_transformation_mode_enum::ALIGN_CORNERS;
    } else if (resizeAttrs->coordinate_transformation_mode == "asymmetric") {
      cpu_param->cpu_operation.resize_op.coordinate_transf_mode =
          resize_coordinate_transformation_mode_enum::ASYMMETRIC;
    } else {
      LOG_FATAL << "unsupported coordinate_transformation_mode "
                << resizeAttrs->coordinate_transformation_mode;
    }
    if (resizeAttrs->rounding_method == "round" || resizeAttrs->rounding_method == "") {
      cpu_param->cpu_operation.resize_op.rounding_method = resize_rounding_method_enum::ROUND;
    } else if (resizeAttrs->rounding_method == "floor") {
      cpu_param->cpu_operation.resize_op.rounding_method = resize_rounding_method_enum::FLOOR;
    } else if (resizeAttrs->rounding_method == "ceil") {
      cpu_param->cpu_operation.resize_op.rounding_method = resize_rounding_method_enum::CEIL;
    } else {
      LOG_FATAL << "unsupported rounding_method " << resizeAttrs->rounding_method;
    }
    cpu_param->cpu_operation.resize_op.bicubic_alpha = resizeAttrs->bicubic_alpha;
    cpu_param->cpu_operation.resize_op.bicubic_exclude = resizeAttrs->bicubic_exclude;

    // op_buffer_desc
    Op_buffer_desc src_data;
    Op_buffer_desc dst_data;
    Expr src_expr = GetNotQuantizedExpr(call->args[0]);
    Expr dst_expr = GetRef<Expr>(call);
    std::vector<Type> src_tensor_check_type;
    src_tensor_check_type = shape_type_get(src_expr);
    src_data = op_buffer_desc_get(src_expr, src_tensor_check_type[0], src_data, 0);
    std::vector<Type> dst_tensor_check_type;
    dst_tensor_check_type = shape_type_get(dst_expr);
    dst_data = op_buffer_desc_get(dst_expr, dst_tensor_check_type[0], dst_data, 2);
    cpu_param->cpu_operation_buffer.resize_buffers.src_data = src_data;
    cpu_param->cpu_operation_buffer.resize_buffers.dst_data = dst_data;

    if (debug_info)
      output_info(cpu_param->cpu_operation.resize_op.common,
                  cpu_param->cpu_operation_buffer.resize_buffers.src_data,
                  cpu_param->cpu_operation_buffer.resize_buffers.dst_data);
    return cpu_param;
  }

  Cpu_param* pool2d_op(const CallNode* call) {
    Cpu_param* cpu_param = new Cpu_param();
    Common_parameters common;
    common_parameters_get(call, common);
    cpu_param->cpu_operation.pool2d_op.common = common;

    const auto* op_node = call->op.as<OpNode>();
    const auto op_name = GetRef<Op>(op_node)->name;
    if (op_name == "nn.max_pool2d") {
      cpu_param->cpu_operation.pool2d_op.pool_type = 0;
      const auto* max_attr = call->attrs.as<MaxPool2DAttrs>();
      if (max_attr->layout == "NCHW") {
        cpu_param->cpu_operation.pool2d_op.layout = resize_layout_enum::NCHW;
      } else if (max_attr->layout == "NHWC") {
        cpu_param->cpu_operation.pool2d_op.layout = resize_layout_enum::NHWC;
      } else {
        LOG_FATAL << "unsupported layout " << max_attr->layout;
      }
      for (size_t i = 0; i < max_attr->padding.size(); i++) {
        cpu_param->cpu_operation.pool2d_op.padding[i] =
            max_attr->padding[i].as<IntImmNode>()->value;
      }
      cpu_param->cpu_operation.pool2d_op.kernel[0] = max_attr->pool_size[0].as<IntImmNode>()->value;
      cpu_param->cpu_operation.pool2d_op.kernel[1] = max_attr->pool_size[1].as<IntImmNode>()->value;
      cpu_param->cpu_operation.pool2d_op.strides[0] = max_attr->strides[0].as<IntImmNode>()->value;
      cpu_param->cpu_operation.pool2d_op.strides[1] = max_attr->strides[1].as<IntImmNode>()->value;
      if (max_attr->ceil_mode == true)
        cpu_param->cpu_operation.pool2d_op.ceil_mode = 1;
      else if (max_attr->ceil_mode == false)
        cpu_param->cpu_operation.pool2d_op.ceil_mode = 0;
      else
        LOG_ERROR << "bad max_attr->ceil_mode " << max_attr->ceil_mode;
    } else if (op_name == "nn.avg_pool2d") {
      cpu_param->cpu_operation.pool2d_op.pool_type = 1;
      const auto* avg_attr = call->attrs.as<AvgPool2DAttrs>();
      if (avg_attr->layout == "NCHW") {
        cpu_param->cpu_operation.pool2d_op.layout = resize_layout_enum::NCHW;
      } else if (avg_attr->layout == "NHWC") {
        cpu_param->cpu_operation.pool2d_op.layout = resize_layout_enum::NHWC;
      } else {
        LOG_FATAL << "unsupported layout " << avg_attr->layout;
      }
      for (size_t i = 0; i < avg_attr->padding.size(); i++) {
        cpu_param->cpu_operation.pool2d_op.padding[i] =
            avg_attr->padding[i].as<IntImmNode>()->value;
      }
      cpu_param->cpu_operation.pool2d_op.kernel[0] = avg_attr->pool_size[0].as<IntImmNode>()->value;
      cpu_param->cpu_operation.pool2d_op.kernel[1] = avg_attr->pool_size[1].as<IntImmNode>()->value;
      cpu_param->cpu_operation.pool2d_op.strides[0] = avg_attr->strides[0].as<IntImmNode>()->value;
      cpu_param->cpu_operation.pool2d_op.strides[1] = avg_attr->strides[1].as<IntImmNode>()->value;
      if (avg_attr->ceil_mode == true)
        cpu_param->cpu_operation.pool2d_op.ceil_mode = 1;
      else if (avg_attr->ceil_mode == false)
        cpu_param->cpu_operation.pool2d_op.ceil_mode = 0;
      else
        LOG_ERROR << "bad avg_attr->ceil_mode " << avg_attr->ceil_mode;
      if (avg_attr->count_include_pad == true)
        cpu_param->cpu_operation.pool2d_op.count_include_pad = 1;
      else if (avg_attr->count_include_pad == false)
        cpu_param->cpu_operation.pool2d_op.count_include_pad = 0;
      else
        LOG_ERROR << "bad avg_attr->count_include_pad " << avg_attr->count_include_pad;
    }

    // op_buffer_desc
    Op_buffer_desc src_data;
    Op_buffer_desc dst_data;

    Expr src_expr = GetNotQuantizedExpr(call->args[0]);
    Expr dst_expr = GetRef<Expr>(call);
    std::vector<Type> src_tensor_check_type;
    src_tensor_check_type = shape_type_get(src_expr);
    src_data = op_buffer_desc_get(src_expr, src_tensor_check_type[0], src_data, 0);
    std::vector<Type> dst_tensor_check_type;
    dst_tensor_check_type = shape_type_get(dst_expr);
    dst_data = op_buffer_desc_get(dst_expr, dst_tensor_check_type[0], dst_data, 2);
    cpu_param->cpu_operation_buffer.pool2d_buffers.src_data = src_data;
    cpu_param->cpu_operation_buffer.pool2d_buffers.dst_data = dst_data;
    if (debug_info)
      output_info(cpu_param->cpu_operation.pool2d_op.common,
                  cpu_param->cpu_operation_buffer.pool2d_buffers.src_data,
                  cpu_param->cpu_operation_buffer.pool2d_buffers.dst_data);
    return cpu_param;
  }

  Cpu_param* pad_op(const CallNode* call) {
    Cpu_param* cpu_param = new Cpu_param();
    Common_parameters common;
    common_parameters_get(call, common);
    cpu_param->cpu_operation.pad_op.common = common;

    const auto* padAttrs = call->attrs.as<PadAttrs>();
    if (padAttrs->pad_mode == "constant") {
      cpu_param->cpu_operation.pad_op.pad_mode = pad_mode_enum::CONSTANT;
    } else if (padAttrs->pad_mode == "edge") {
      cpu_param->cpu_operation.pad_op.pad_mode = pad_mode_enum::EDGE;
    } else if (padAttrs->pad_mode == "reflect") {
      cpu_param->cpu_operation.pad_op.pad_mode = pad_mode_enum::REFLECT;
    } else {
      LOG_FATAL << "unsupported pad_mode " << padAttrs->pad_mode;
    }
    Op_buffer_desc src_data_pad;
    Expr src_expr_pad;
    src_expr_pad = GetNotQuantizedExpr(call->args[1]);
    std::vector<Type> src2_tensor_check_type;
    src2_tensor_check_type = shape_type_get(src_expr_pad);
    src_data_pad = op_buffer_desc_get(src_expr_pad, src2_tensor_check_type[0], src_data_pad, 0);
    cpu_param->cpu_operation.pad_op.src_data = src_data_pad;

    // op_buffer_desc
    Op_buffer_desc src_data;
    Op_buffer_desc dst_data;
    Expr src_expr = GetNotQuantizedExpr(call->args[0]);
    Expr dst_expr = GetRef<Expr>(call);
    std::vector<Type> src_tensor_check_type;
    src_tensor_check_type = shape_type_get(src_expr);
    src_data = op_buffer_desc_get(src_expr, src_tensor_check_type[0], src_data, 0);
    std::vector<Type> dst_tensor_check_type;
    dst_tensor_check_type = shape_type_get(dst_expr);
    dst_data = op_buffer_desc_get(dst_expr, dst_tensor_check_type[0], dst_data, 2);
    cpu_param->cpu_operation_buffer.pad_buffers.src_data = src_data;
    cpu_param->cpu_operation_buffer.pad_buffers.dst_data = dst_data;

    if (debug_info)
      output_info(cpu_param->cpu_operation.pad_op.common,
                  cpu_param->cpu_operation_buffer.pad_buffers.src_data,
                  cpu_param->cpu_operation_buffer.pad_buffers.dst_data);
    return cpu_param;
  }

  // yuanyue 20220516 broadcast
  // Weightshape broadcast to outshape.
  // This function implements two shape broadcasting,
  // but, at present, in this class it can only be used to broadcasting weightshape as outshape
  // when the dimensions of the weightshape and outshape are identical.
  // to be advised(TBA)
  std::vector<int> Broadcast_Shape(std::vector<int> &weightshape, std::vector<int> outshape){
    int s1_size=weightshape.size();
    int s2_size=outshape.size();
    int i;
    std::vector<int> common_shape;
    //Need to revisit this part
    for (i = 1; i <= std::min(s1_size, s2_size); ++i) {
      int static_size1 = weightshape[s1_size - i];
      int static_size2 = outshape[s2_size - i];
      if (static_size1==static_size2) {
        common_shape.insert(common_shape.begin(),weightshape[s1_size - i]);
      }
      else if (static_size1==1) {
        common_shape.insert(common_shape.begin(),outshape[s2_size - i]);
      }
      else if (static_size2==1) {
        common_shape.insert(common_shape.begin(),weightshape[s1_size - i]);
      }
      else {
        LOG(FATAL) << "AIPU RISCV Incompatible broadcast dims:" << static_size1 << static_size2;
      }
      }
    // Remaining dimensions whether on shape1 or shape2 can always be completed
    int max_size = std::max(s1_size, s2_size);
    auto& shape = (s1_size > s2_size) ? weightshape : outshape;
    for (; i <= max_size; ++i) {
      common_shape.insert(common_shape.begin(),shape[max_size - i]);
    }
    //LOG(INFO)<<"weightshape: "<< weightshape[0] <<", "<< weightshape[1] <<", "<< weightshape[2] <<", "<< weightshape[3] ;
    //LOG(INFO)<<"outshape: "<< outshape[0] <<", "<< outshape[1] <<", "<< outshape[2] <<", "<< outshape[3] ;
    //LOG(INFO)<<"common_shape: "<< common_shape[0] <<", "<< common_shape[1] <<", "<< common_shape[2] <<", "<< common_shape[3] ;
    return common_shape;
  } 
  
  // yuanyue 20220516 broadcast
  // broadcast Weightdata.
  void Broadcast_weightdata(void* weightdata, void* broadcast_weights,std::vector<int>wshape, 
  std::vector<int>broadcast_wshape, const TensorTypeNode *wttype){
    int D0=broadcast_wshape[0];
    int D1=broadcast_wshape[1];
    int D2=broadcast_wshape[2];
    int D3=broadcast_wshape[3];
    //int times=(D0/wshape[0])*(D1/wshape[1])*(D2/wshape[2])*(D3/wshape[3]);
    //void *tmp_weightdata = (void*)malloc(times*sizeof(weightdata));
    //LOG(INFO)<<"wttype->dtype.code(): "<< wttype->dtype.code();
    for(int i=0;i<D0;i++){
      int i0 = i < wshape[0] ? i: wshape[0]-1;
      for(int j=0;j<D1;j++){
        int j0 = j < wshape[1] ? j: wshape[1]-1;
        for(int k=0;k<D2;k++){
          int k0 = k < wshape[2] ? k: wshape[2]-1;
          for(int l=0;l<D3;l++){
            int l0 = l < wshape[3] ? l: wshape[3]-1;
            if (wttype->dtype.code() == DataType::kInt ){
              //LOG(INFO) << "DataType::kInt";
              ((int*)broadcast_weights)[i*D1*D2*D3+j*D2*D3+k*D3+l]=((int*)weightdata)[i0*D1*D2*D3+j0*D2*D3+k0*D3+l0];
            }
            else if (wttype->dtype.code() == DataType::kUInt){
              //LOG(INFO) << "DataType::kUInt";
              ((unsigned int*)broadcast_weights)[i*D1*D2*D3+j*D2*D3+k*D3+l]=((unsigned int*)weightdata)[i0*D1*D2*D3+j0*D2*D3+k0*D3+l0];
            }
            else if (wttype->dtype.code() == DataType::kFloat){
              //LOG(INFO) << "DataType::kFloat";
              ((float*)broadcast_weights)[i*D1*D2*D3+j*D2*D3+k*D3+l]=((float*)weightdata)[i0*D1*D2*D3+j0*D2*D3+k0*D3+l0];
            }
            else if (wttype->dtype.code() == DataType::kBFloat && wttype->dtype.bits() == 16){
              //LOG(INFO) << "DataType::kBFloat";
              ((char*)broadcast_weights)[i*D1*D2*D3*2+j*D2*D3*2+k*D3*2+l]=((char*) weightdata)[i0*D1*D2*D3*2+j0*D2*D3*2+k0*D3*2+l0];
              ((char*)broadcast_weights)[i*D1*D2*D3*2+j*D2*D3*2+k*D3*2+l+1]=((char*) weightdata)[i0*D1*D2*D3*2+j0*D2*D3*2+k0*D3*2+l0+1];
            }
            else{
              LOG(FATAL) << "Broadcast_weightdata doesn't support: " << wttype->dtype.code() << "TypeCode";
            }
          }
        }
      }
    }
  }


  typedef Cpu_param* (CodegenAIPU::*operator2riscv)(const CallNode*);
  std::map<std::string, operator2riscv> relayParseTable = {
      {"nn.softmax", &CodegenAIPU::common_only_op},
      {"exp", &CodegenAIPU::common_only_op},
      {"sigmoid", &CodegenAIPU::common_only_op},
      {"reshape", &CodegenAIPU::common_only_op},
      {"sqrt", &CodegenAIPU::common_only_op},
      {"erf", &CodegenAIPU::common_only_op},
      {"tanh", &CodegenAIPU::common_only_op},
      {"nn.relu", &CodegenAIPU::common_only_op},
      {"nn.leaky_relu", &CodegenAIPU::common_only_op},
      {"featuretoweight", &CodegenAIPU::common_only_op},

      {"add", &CodegenAIPU::with_weight_op},
      {"less", &CodegenAIPU::with_weight_op},
      {"nn.batch_matmul", &CodegenAIPU::with_weight_op},
      {"power", &CodegenAIPU::with_weight_op},
      {"divide", &CodegenAIPU::with_weight_op},
      {"subtract", &CodegenAIPU::with_weight_op},
      {"nn.dense", &CodegenAIPU::with_weight_op},
      {"multiply", &CodegenAIPU::with_weight_op},

      {"transpose", &CodegenAIPU::transform_op},
      {"squeeze", &CodegenAIPU::transform_op},
      {"split", &CodegenAIPU::split_op},
      {"expand_dims", &CodegenAIPU::expand_dims_op},
      {"concatenate", &CodegenAIPU::concat_op},
      {"max", &CodegenAIPU::reduce_op},
      {"min", &CodegenAIPU::reduce_op},
      {"sum", &CodegenAIPU::reduce_op},
      {"mean", &CodegenAIPU::reduce_op},
      {"all", &CodegenAIPU::reduce_op},
      {"any", &CodegenAIPU::reduce_op},
      {"argmax", &CodegenAIPU::reduce_op},
      {"argmin", &CodegenAIPU::reduce_op},
      {"cast", &CodegenAIPU::cast_op},
      {"one_hot", &CodegenAIPU::one_hot_op},
      {"take", &CodegenAIPU::take_op},
      {"strided_slice", &CodegenAIPU::slice_op},
      {"image.resize", &CodegenAIPU::resize_op},
      {"nn.max_pool2d", &CodegenAIPU::pool2d_op},
      {"nn.avg_pool2d", &CodegenAIPU::pool2d_op},
      {"nn.pad", &CodegenAIPU::pad_op},
  };

  // void VisitExpr_(const VarNode* node) final {}

  // void VisitExpr_(const ConstantNode* cn) final {}

  // need to deal with in the future
  // void VisitExpr_(const TupleNode* n) final {}
  void VisitExpr_(const TupleNode* op) final {
    for (auto field : op->fields) {
      VisitExpr(field);
    }
  }

  void VisitExpr_(const CallNode* call) final {
    for (auto arg : call->args) {
      VisitExpr(arg);
    }
    //LOG_INFO << "visit expr call->op " << call->op;
    const auto* op_node = call->op.as<OpNode>();
    const auto op_name = GetRef<Op>(op_node)->name;

    if (op_name == "relay.op.annotation.simulated_quantize") {
      auto argconst = Downcast<Constant>(call->args[1]);
      float scale = *((float*)argconst->data->data);
      op_in_scale_map_.insert(std::pair<Expr, float>(GetRef<Expr>(call), scale));
      return;
    }

    std::map<std::string, operator2riscv>::iterator it = relayParseTable.find(op_name);
    if (it == relayParseTable.end())
      LOG(FATAL) << "AIPU RISCV codegen doesn't support: " << op_name;
    if(debug_info) {
      LOG(INFO) << "RISC-V op_name: " << op_name;
    }
    for (size_t i=0; i<call->args.size(); i++) {
      if(debug_info) {
        LOG(INFO) << "RISC-V VarNode: " << call->args[i]->IsInstance<tvm::relay::VarNode>();
      }
      if (call->args[i]->IsInstance<tvm::relay::VarNode>() && !subfunc_in_scale_map_.empty()) {
        if (subfunc_in_scale_map_.find(call->args[i]) != subfunc_in_scale_map_.end()) {
          iscale_[i] = subfunc_in_scale_map_[call->args[i]][0];
          if (subfunc_in_scale_map_[call->args[i]].size() > 1) { // TupleNode
            for (size_t j=0; j<subfunc_in_scale_map_[call->args[i]].size(); j++)
              iscale_[j] = subfunc_in_scale_map_[call->args[i]][j];
          }
        } else {
          iscale_[i] = -1.0f;
        }
      } else if (op_in_scale_map_.find(call->args[i]) != op_in_scale_map_.end()) {
        iscale_[i] = op_in_scale_map_[call->args[i]];
      } else {
        iscale_[i] = -1.0f;
      }
    }

    if (op_out_scale_map_.find(GetRef<Expr>(call)) != op_out_scale_map_.end()) {
      oscale_ = op_out_scale_map_[GetRef<Expr>(call)];
    } else {
      oscale_ = -1.0f;
    }

    riscv_code.push_back((this->*it->second)(call));
    expr_execute_order_.insert(std::pair<Expr, size_t>(GetRef<Expr>(call), riscv_code.size()));

  }

  void preprocess(Riscv_wt_list riscv_wt_list, size_t weight_offset,
                  std::map<Expr, float> op_out_scale_map, std::map<Expr, std::vector<float>>subfunc_in_scale_map) {
    // input and output offset
    for (auto mp : storage_device_map_) {
      // bool flag_input_output = false;
      Expr mp_expr = GetNotQuantizedExpr(mp.first);
      for (auto it : storage_input_output_map_) {
        Expr it_expr = GetNotQuantizedExpr(it.first);
        if (it_expr == mp_expr) {
          // flag_input_output = true;
          // LOG(INFO) << AsText(mp.first,false);
          /*
          if(it.second[0].size() != mp.second[0].size()){
            LOG(INFO)<<"it.second[0].size(): " << it.second[0].size() << " mp.second[0].size(): "<< mp.second[0].size();
            LOG(INFO) << " mp.first: " << AsText(mp.first,false);
            LOG(INFO) << " mp_expr: " << AsText(mp_expr,false);
            LOG(INFO) << " it.first: " << AsText(it.first,false);
            LOG(INFO) << " it_expr: " << AsText(it_expr,false);
            for (int s =0 ; s < it.second[0].size() ; s++){
              size_t offset = it.second[1][s];
              int storage_id = it.second[0][s];
              LOG(INFO)<<"storage_id: "<< storage_id <<" offset: "<< offset ;
            }
            LOG(FATAL)<< "it.second[0].size(): " << it.second[0].size() << " mp.second[0].size(): "<< mp.second[0].size();
          }
          */
          for (int i = 0; i < it.second[0].size(); i++) {
            size_t offset = it.second[1][i];
            int storage_id = mp.second[0][i];
            //LOG(INFO)<<"it.second[0].size(): "<< it.second[0].size() <<" storage_id: "<< storage_id <<" offset: "<< offset ;
            temporary_data_offset_[storage_id] = offset;
          }
        }
      }
    }
    //LOG(INFO) << "data_memory_used_: " << data_memory_used_;
    //LOG(INFO) << "temporary_data_offset_: " << temporary_data_offset_.size();

    weight_base_size_ = riscv_wt_list.size();
    weight_offset_ = weight_offset;
    op_out_scale_map_ = op_out_scale_map;
    subfunc_in_scale_map_ = subfunc_in_scale_map;
  }

  void postprocess(Riscv_addr_list& riscv_addr_list, Riscv_wt_list& riscv_wt_list,
                   size_t& weight_offset) {
    riscv_addr_list.push_back(riscv_code_offset_);
    for (size_t i = 0; i < riscv_wt_list_.size(); i++) riscv_wt_list.push_back(riscv_wt_list_[i]);

    weight_offset = weight_offset_;
  }

  Expr GetNotQuantizedExpr(Expr expr) {
    if (expr->IsInstance<CallNode>()) {
      auto op = Downcast<Call>(expr);
      if (const auto* op_node = op->op.as<OpNode>()) {
        std::string op_name = GetRef<Op>(op_node)->name;
        if (op_name == "relay.op.annotation.simulated_quantize" ||
            op_name.substr(0, 10) == "annotation")
          return GetNotQuantizedExpr(op->args[0]);
      }
    }
    return expr;
  }

  std::vector<Cpu_param*> GetRiscvSource() {
    // LOG(INFO) << "pass riscv_vector_ sise(): " << riscv_vector_.size();
    return riscv_code;
  }

  void debug() {
    for (auto it : riscv_code_offset_)
      LOG(INFO) << "index " << it.first << " offset " << it.second;
    for (auto it : temporary_data_offset_)
      LOG(INFO) << "storage_id " << it.first << " offset " << it.second;
    for (auto it : riscv_wt_list_)
      LOG(INFO) << "index " << it.first << " address " << it.second;
  }

  /*
  void debug(std::vector<Cpu_param *> test_riscv_code) {
    for (auto it : temporary_data_storage_)
      LOG(INFO) << "storage_id " << it.first << " size " << it.second;
    for (auto it : temporary_data_offset_)
      LOG(INFO) << "storage_id " << it.first << " offset " << it.second;
    LOG(INFO) << "!!!!!!!!!!!!!!!!!!!*****************************!!!!!!!!!!!!!!!!!!!!!!!";
    for (int i =0 ; i < test_riscv_code.size() ; i++ ){
      if (test_riscv_code[i]->op_type == tvm::runtime::contrib::SOFTMAX ){
        LOG(INFO) << "************SOFTMAX************";
        output_info(test_riscv_code[i]->cpu_operation.common_only_op.common,
                test_riscv_code[i]->cpu_operation_buffer.common_only_op_buffers.src_data,
                test_riscv_code[i]->cpu_operation_buffer.common_only_op_buffers.dst_data);

      }else if (test_riscv_code[i]->op_type == tvm::runtime::contrib::CONCAT){
        LOG(INFO) << "************CONCAT************";
        output_info(test_riscv_code[i]->cpu_operation.concat_op.common,
                test_riscv_code[i]->cpu_operation_buffer.concat_buffers.src_data,
                test_riscv_code[i]->cpu_operation_buffer.concat_buffers.dst_data);
        LOG(INFO) << "concat_param axis: " << test_riscv_code[i]->cpu_operation.concat_op.axis;
        LOG(INFO) << "common.input_num: " <<
  test_riscv_code[i]->cpu_operation.concat_op.common.input_num; for (int j = 0; j <
  test_riscv_code[i]->cpu_operation.concat_op.common.input_num; j++)
          output_info_other(test_riscv_code[i]->cpu_operation.concat_op.src_data[j]);

      }else if (test_riscv_code[i]->op_type == tvm::runtime::contrib::SPLIT){
        LOG(INFO) << "************SPLIT************";
        output_info(test_riscv_code[i]->cpu_operation.split_op.common,
                test_riscv_code[i]->cpu_operation_buffer.split_buffers.src_data,
                test_riscv_code[i]->cpu_operation_buffer.split_buffers.dst_data);
        LOG(INFO) << "split_param axis: " << test_riscv_code[i]->cpu_operation.split_op.axis;
        LOG(INFO) << "split_param indices[INPUT_MAX]: " ;
        for (int j=0;j<INPUT_MAX;j++){
          LOG(INFO) << test_riscv_code[j]->cpu_operation.split_op.indices[j];
        }
        LOG(INFO) << "common.output_num: " <<
  test_riscv_code[i]->cpu_operation.split_op.common.output_num; for (int j = 0; j <
  test_riscv_code[i]->cpu_operation.split_op.common.output_num; j++)
          output_info_other(test_riscv_code[i]->cpu_operation.split_op.dst_data[j]);

      }else if (test_riscv_code[i]->op_type == tvm::runtime::contrib::SUM){
        LOG(INFO) << "************SUM************";
        output_info(test_riscv_code[i]->cpu_operation.reduce_op.common,
                test_riscv_code[i]->cpu_operation_buffer.reduce_buffers.src_data,
                test_riscv_code[i]->cpu_operation_buffer.reduce_buffers.dst_data);
        int32_t * tmp = test_riscv_code[i]->cpu_operation.reduce_op.axis;
        LOG(INFO) << "split_param axis: " << tmp[0] <<", "<< tmp[1] <<", "<< tmp[2]<<", " << tmp[3]
  ; LOG(INFO) << "split_param keepdims: " << test_riscv_code[i]->cpu_operation.reduce_op.keepdims ;
        LOG(INFO) << "split_param exclude: " << test_riscv_code[i]->cpu_operation.reduce_op.exclude
  ;

      }else if (test_riscv_code[i]->op_type == tvm::runtime::contrib::EXPAND_DIMS){
        LOG(INFO) << "************EXPAND_DIMS************";
        output_info(test_riscv_code[i]->cpu_operation.expand_dims_op.common,
                test_riscv_code[i]->cpu_operation_buffer.expand_dims_buffers.src_data,
                test_riscv_code[i]->cpu_operation_buffer.expand_dims_buffers.dst_data);
        LOG(INFO) << "split_param axis: " << test_riscv_code[i]->cpu_operation.expand_dims_op.axis ;
        LOG(INFO) << "split_param num_newaxis: " <<
  test_riscv_code[i]->cpu_operation.expand_dims_op.num_newaxis ;

      }else if (test_riscv_code[i]->op_type == tvm::runtime::contrib::RESHAPE){
        LOG(INFO) << "************RESHAPE************";
        output_info(test_riscv_code[i]->cpu_operation.common_only_op.common,
                test_riscv_code[i]->cpu_operation_buffer.common_only_op_buffers.src_data,
                test_riscv_code[i]->cpu_operation_buffer.common_only_op_buffers.dst_data);
      }
      else{
        LOG(INFO) << "!!!!!!!!!!!!!!!!!!!! op_type: " << test_riscv_code[i]->op_type;
      }
    }
  }
  */

 protected:
  // plan memory of operators and weights
  Map<Expr, Array<IntegerArray>> storage_device_map_;

  Map<Expr, Array<IntegerArray>> storage_input_output_map_;

  // used for accumulated operator input and output, these memory blocks can be reused
  size_t data_memory_used_;

  size_t weight_offset_;

  size_t weight_base_size_;

  // key: expr , 
  // value: two array are singly ref_dtypes, dtype_names
  Map<Expr, Array<IntegerArray>> aid_dtype_map_;
  // map between memory block and its' size
  std::map<int, size_t> temporary_data_storage_;
  // map between memory block and its' offset
  std::map<int, size_t> temporary_data_offset_;
  // final riscv code
  std::vector<Cpu_param*> riscv_code;
  std::vector<std::pair<int32_t, size_t>> riscv_code_offset_;
  Riscv_wt_list riscv_wt_list_;
  std::map<Expr, size_t> expr_execute_order_;

  //std::map<Expr, uint32_t> expr_index_map_;

  std::map<Expr, float> op_in_scale_map_;
  std::map<Expr, float> op_out_scale_map_;
  Expr func_body;
  float* iscale_ = new float[10]();
  float oscale_ = -1.0f;
  std::map<Expr, std::vector<float>> subfunc_in_scale_map_;
  bool debug_info = true;
};

std::vector<Cpu_param*> CompileFunc4Riscv(const Function& func, Riscv_addr_list& riscv_addr_list,
                                          Riscv_wt_list& riscv_wt_list,
                                          Map<Expr, Array<IntegerArray>> aid_dtype_map,
                                          Map<Expr, Array<IntegerArray>> storage_input_output_map,
                                          size_t total_memory_used, size_t& weight_offset,
                                          std::map<Expr, float> op_out_scale_map, std::map<Expr, std::vector<float>> subfunc_in_scale_map) {
  //Map<Expr, IntegerArray> aid_dtype = aid_dtype_;
  std::string c_name = "riscv";
  Map<Expr, Array<IntegerArray>> storage_device_map;
  StorageAllocator storage_allocator;
  storage_device_map = storage_allocator.Plan(func, aid_dtype_map, total_memory_used, c_name);
  std::map<int, size_t> temporary_data_storage = storage_allocator.GetDataStorage();
  // temporary_data_offset = storage_allocator.GetDataOffset();

  CodegenAIPU codegen_riscv(func, storage_device_map, storage_input_output_map, aid_dtype_map,
                            temporary_data_storage, total_memory_used);

  codegen_riscv.preprocess(riscv_wt_list, weight_offset, op_out_scale_map, subfunc_in_scale_map);
  codegen_riscv.VisitExpr(func->body);
  //codegen_riscv.debug();
  // std::vector<Cpu_param *> test_riscv_code = codegen_riscv.GetRiscvSource();
  // codegen_riscv.debug(test_riscv_code);

  codegen_riscv.postprocess(riscv_addr_list, riscv_wt_list, weight_offset);
  return codegen_riscv.GetRiscvSource();
}
}  // namespace aipu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_AIPU_CODEGEN_RISCV_H_
