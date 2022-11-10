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
 * \file AIPU_add_after_dense.cc
 * \brief To make DENSE op execute on NVDLA, we need an ADD op after DENSE op.
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include "./pattern_utils.h"
#include <tvm/relay/expr.h>
#include <tvm/runtime/ndarray.h>

namespace tvm {
namespace relay {

class AddAfterDense_rewritter : public ExprRewriter {
 public:
  explicit AddAfterDense_rewritter()
      :dense_op_(Op::Get("nn.dense")) {}

  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    if (pre->op == dense_op_) {
      auto tensor_a = post.as<CallNode>()->args[0];
      auto tensor_b = post.as<CallNode>()->args[1];
      if(!( (pre->checked_type_).defined() && (pre->args[0]->checked_type_).defined() && (pre->args[1]->checked_type_).defined() )) {
        std::cout << "FATAL!!! The checked_type_ field is not available. Consider do InferType before this pass." << std::endl;
        exit(0);
      }

      auto checked_type = pre->checked_type();
      const auto* tensor_type = checked_type.as<TensorTypeNode>();
      std::cout << "Type information: " << checked_type->GetTypeKey() << " " << tensor_type->shape << " " << tensor_type->dtype << std::endl;

      int dim_n = (tensor_type->shape[0]).as<IntImmNode>()->value;
      int dim_c = (tensor_type->shape[1]).as<IntImmNode>()->value;
      // int dim_h = 1;
      // int dim_w = 1;
      auto attrs = make_object<ReshapeAttrs>();
      attrs->newshape = Array<Integer>({dim_n, dim_c});
      
      runtime::NDArray zero_array = runtime::NDArray::Empty({dim_c}, DataType::Float(32U), {kDLCPU, 0});
      // define a DLTensor with all zero elements
      DLTensor zero_dltensor;
      zero_dltensor.data = (void*)(new float_t[dim_c]);
      for(int __i = 0; __i < dim_c; ++__i) {
        ((float_t*)(zero_dltensor.data))[__i] = 0;
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
      Expr result = relay::Call(relay::Op::Get("add"), {post, zeros});

      return result;
    }
    return post;
  }

 private:
  const Op& dense_op_;
};

Expr AddAfterDense(const Expr& expr) {
  auto rewriter = AddAfterDense_rewritter();
  return PostOrderRewrite(expr, &rewriter);
}

namespace transform {

Pass AddAfterDense() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto r = Downcast<Function>(AddAfterDense(f));
        return r;
      };
  return CreateFunctionPass(pass_func, 1000, "AddAfterDense", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.AddAfterDenseTransform").set_body_typed(AddAfterDense);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
