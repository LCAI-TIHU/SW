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
 * \file AIPU_batch_matmul_transform.cc
 * \brief transform batch_matmul operator to a set of dense operators. Execute this pass after type checking
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include "./pattern_utils.h"
#include <tvm/relay/expr.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/runtime/logging.h>

namespace tvm {
namespace relay {

class BatchMatMulTransform_rewriter : public ExprRewriter {
 public:
  explicit BatchMatMulTransform_rewriter()
      :batch_matmul_op_(Op::Get("nn.batch_matmul")),
       quantize_op_(relay::Op::Get("relay.op.annotation.simulated_quantize")),
       reshape_op_(relay::Op::Get("reshape")),
       split_op_(relay::Op::Get("split")),
       dense_op_(relay::Op::Get("nn.dense")),
       cat_op_(relay::Op::Get("concatenate")),
       add_op_(relay::Op::Get("add"))
  {}

  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    /*
    const Op softmax_op_ = relay::Op::Get("nn.softmax");
    if(pre->op == softmax_op_) {
      std::cout << "nn.softmax " << std::endl;
      const CallNode* softmax_call = pre;
      int the_axis = softmax_call->attrs.as<tvm::relay::SoftmaxAttrs>()->axis;
      std::cout << "axis: " << the_axis << std::endl;
      int aa;
      std::cin >> aa;
    }
    */
    // we assume pre has the pattern: simulated_quantize ---> batch_matmul ---> simulated_quantize. So we first check if pre has this pattern
    if (!(pre->op == quantize_op_)) {
      return post;
    }
    if (!(pre->args[0]->IsInstance<tvm::relay::CallNode>())) {
      return post;
    }
    const CallNode* batch_matmul_call = pre->args[0].as<CallNode>();
    if(batch_matmul_call->op != batch_matmul_op_) {
      return post;
    }
    // LOG(INFO) << "Deal with a batch_matmul op" << std::endl;
    if(batch_matmul_call->args[0].as<CallNode>()->op != quantize_op_ || batch_matmul_call->args[1].as<CallNode>()->op != quantize_op_) {
      LOG(FATAL) << "The inputs of batch_matmul should be quantized!!!" << std::endl;
      exit(0);
    }
    //const auto param = batch_matmul_call->attrs.as<BatchMatmulAttrs>();
    if(!( (batch_matmul_call->checked_type_).defined() && (batch_matmul_call->args[0]->checked_type_).defined() && (batch_matmul_call->args[1]->checked_type_).defined() )){
      LOG(FATAL) << "The checked_type_ field is not available. Consider do InferType before this pass." << std::endl;
      exit(0);
    }

    Expr tensor_out = post.as<CallNode>()->args[0];
    Expr tensor_a = tensor_out.as<CallNode>()->args[0];
    Expr tensor_b = tensor_out.as<CallNode>()->args[1];
    
    // these are used to extract quantization informations
    const CallNode* call_out = post.as<CallNode>();
    const CallNode* call_a = tensor_a.as<CallNode>();
    const CallNode* call_b = tensor_b.as<CallNode>();

    auto checked_type = batch_matmul_call->checked_type();
    const auto* tensor_type = checked_type.as<TensorTypeNode>();
    // LOG(INFO) << "Type information: " << checked_type->GetTypeKey() << " " << tensor_type->shape << " " << tensor_type->dtype << std::endl;

    auto checked_type_a = batch_matmul_call->args[0]->checked_type();
    const auto* tensor_type_a = checked_type_a.as<TensorTypeNode>();
    if(!tensor_type_a){
      return post;
    }
    //LOG(INFO) << "Type information of a: " << checked_type_a->GetTypeKey() << " " << tensor_type_a->shape << " " << tensor_type_a->dtype << std::endl;

    auto checked_type_b = batch_matmul_call->args[1]->checked_type();
    const auto* tensor_type_b = checked_type_b.as<TensorTypeNode>();
    if(!tensor_type_b){
      return post;
    }
    // LOG(INFO) << "Type information of b: " << checked_type_b->GetTypeKey() << " " << tensor_type_b->shape << " " << tensor_type_b->dtype << std::endl;
    
    const auto* batch_size_tmp = (tensor_type->shape[0]).as<IntImmNode>();
    int batch_size = batch_size_tmp->value;
    int a_shape_1 = (tensor_type_a->shape[1]).as<IntImmNode>()->value;
    int a_shape_2 = (tensor_type_a->shape[2]).as<IntImmNode>()->value;
    int b_shape_1 = (tensor_type_b->shape[1]).as<IntImmNode>()->value;
    int b_shape_2 = (tensor_type_b->shape[2]).as<IntImmNode>()->value;

    Expr tensor_a_pre = call_a->args[0];
    Expr tensor_b_pre = call_b->args[0];

    auto attrs_a_reshape = make_object<ReshapeAttrs>();
    attrs_a_reshape->newshape = Array<Integer>({a_shape_1, a_shape_2});

    auto attrs_b_reshape = make_object<ReshapeAttrs>();
    attrs_b_reshape->newshape = Array<Integer>({b_shape_1, b_shape_2});

    auto attrs_mult_reshape = make_object<ReshapeAttrs>();
    attrs_mult_reshape->newshape = Array<Integer>({1, a_shape_1, b_shape_1});
      
    // dense attrs for each dense op
    auto attrs_dense = make_object<DenseAttrs>();
    attrs_dense->units = b_shape_1;
    attrs_dense->out_dtype = DataType::Float(32U);

    Array<Expr> results_array;
    for(int i = 0; i < batch_size; ++i){
      Expr component_a;
      Expr component_b;
      // use the slice op approach
      Array<Integer> begin_a({i, 0, 0});
      Array<Integer> end_a({i + 1, a_shape_1, a_shape_2});
      Array<Integer> strides_a({1, 1, 1});
      component_a = MakeStridedSlice(tensor_a_pre, begin_a, end_a, strides_a, "end");

      Array<Integer> begin_b({i, 0, 0});
      Array<Integer> end_b({i + 1, b_shape_1, b_shape_2});
      Array<Integer> strides_b({1, 1, 1});
      component_b = MakeStridedSlice(tensor_b_pre, begin_b, end_b, strides_b, "end");
      Expr component_a_reshaped = relay::Call(reshape_op_, {component_a}, Attrs(attrs_a_reshape), {});
      Expr component_b_reshaped = relay::Call(reshape_op_, {component_b}, Attrs(attrs_b_reshape), {});
      
      // quantization
      auto quantize_attrs_a = make_object<tvm::relay::qnn::SimulatedQuantizeAttrs>();
      quantize_attrs_a->axis = -1;
      Expr component_a_reshaped_quantized = relay::Call(quantize_op_,
              //{component_a_reshaped, call_a->args[1], call_a->args[2], call_a->args[3]}, 
              {component_a_reshaped, Constant(call_a->args[1].as<ConstantNode>()->data), Constant(call_a->args[2].as<ConstantNode>()->data), Constant(call_a->args[3].as<ConstantNode>()->data)}, 
              Attrs(quantize_attrs_a),
              {});

      auto quantize_attrs_b = make_object<tvm::relay::qnn::SimulatedQuantizeAttrs>();
      quantize_attrs_b->axis = -1;
      Expr component_b_reshaped_quantized = relay::Call(quantize_op_,
              {component_b_reshaped, Constant(call_b->args[1].as<ConstantNode>()->data), Constant(call_b->args[2].as<ConstantNode>()->data), Constant(call_b->args[3].as<ConstantNode>()->data)}, 
              Attrs(quantize_attrs_b),
              {});
      //

      Expr component_mult_reshaped_quantized = relay::Call(dense_op_, {component_a_reshaped_quantized, component_b_reshaped_quantized}, Attrs(attrs_dense), {});

      // append an add op (with all elements equal to 0) after dense op
      int dim_c = b_shape_1;
      runtime::NDArray zero_array = runtime::NDArray::Empty({dim_c}, DataType::Float(32U), {kDLCPU, 0});
      // define a DLTensor with all 0 elements
      DLTensor zero_dltensor;
      zero_dltensor.data = (void*)(new float_t[dim_c]);
      for(int __i = 0; __i < dim_c; ++__i) {
          ((float*)(zero_dltensor.data))[__i] = 0;
      }
      zero_dltensor.device.device_type = kDLCPU;
      zero_dltensor.device.device_id = 0;
      zero_dltensor.ndim = 1;
      zero_dltensor.dtype.code = kDLFloat;
      zero_dltensor.dtype.bits = 32U;
      zero_dltensor.dtype.lanes = 1U;
      zero_dltensor.shape = new int64_t[1];
      zero_dltensor.shape[0] = dim_c;
      zero_dltensor.strides = nullptr;
      zero_dltensor.byte_offset = 0;
      zero_array.CopyFrom(&zero_dltensor);
      Constant zeros = Constant(zero_array);

      Expr component_mult_add_reshaped_quantized = relay::Call(add_op_, {component_mult_reshaped_quantized, zeros});

      // quantization
      auto quantize_attrs = make_object<tvm::relay::qnn::SimulatedQuantizeAttrs>();
      quantize_attrs->axis = -1;
      Expr component_mult_add_reshaped = relay::Call(quantize_op_,
              {component_mult_add_reshaped_quantized, Constant(call_out->args[1].as<ConstantNode>()->data), Constant(call_out->args[2].as<ConstantNode>()->data), Constant(call_out->args[3].as<ConstantNode>()->data)},
              Attrs(quantize_attrs),
              {});

      Expr component_mult_add = relay::Call(reshape_op_, {component_mult_add_reshaped}, Attrs(attrs_mult_reshape), {});
      results_array.push_back(component_mult_add);
    }
    Expr results_Tuple = Tuple(results_array);

    auto attrs_cat = make_object<ConcatenateAttrs>();
    attrs_cat->axis = 0;
    Expr result_quantized = relay::Call(cat_op_, {results_Tuple}, Attrs(attrs_cat), {});

    return result_quantized;
    //auto attrs_final_reshape = make_object<ReshapeAttrs>();
    //attrs_final_reshape->newshape = Array<Integer>({ batch_size, a_shape_1, b_shape_1});
    //Expr result_quantized_reshaped = relay::Call(reshape_op_, {result_quantized}, Attrs(attrs_final_reshape), {});

    //return result_quantized_reshaped;

  }

 private:
  const Op& batch_matmul_op_;
  const Op& quantize_op_;
  const Op& reshape_op_;
  const Op& split_op_;
  const Op& dense_op_;
  const Op& cat_op_;
  const Op& add_op_;
};

Expr BatchMatMulTransform(const Expr& expr) {
  auto rewriter = BatchMatMulTransform_rewriter();
  return PostOrderRewrite(expr, &rewriter);
}

namespace transform {

Pass BatchMatMulTransform() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto r = Downcast<Function>(BatchMatMulTransform(f));
        return r;
      };
  return CreateFunctionPass(pass_func, 1000, "BatchMatMulTransform", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.BatchMatMulTransform").set_body_typed(BatchMatMulTransform);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
