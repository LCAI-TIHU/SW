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
 * \file AIPU_fuse_GELU.cc
 * \brief GELU is used in bert model. But GELU was implemented as a combination of commonly used operators, which is inefficient to compute. This pass fuse these operators into a new operator "AIPU_GELU".
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "./pattern_utils.h"

#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

class AIPU_fuse_GELU_rewritter : public ExprRewriter {
 public:
  explicit AIPU_fuse_GELU_rewritter() :
      power_op_(Op::Get("power")),
      multiply_op_(Op::Get("multiply")),
      add_op_(Op::Get("add")),
      tanh_op_(Op::Get("tanh")),
      AIPU_GELU_op_(Op::Get("AIPU_GELU")) {}

  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    if (post.as<CallNode>()->op != multiply_op_) return post;

    Expr tensor_0 = post.as<CallNode>()->args[0];
    Expr tensor_1 = post.as<CallNode>()->args[1];
    if (! (tensor_0.as<CallNode>() || tensor_0.as<VarNode>())) return post;
    if (! (tensor_1.as<CallNode>() || tensor_1.as<VarNode>())) return post;
    if (tensor_1.as<CallNode>() && tensor_1.as<CallNode>()->op == multiply_op_ ) {
    } else if (tensor_0.as<CallNode>() && tensor_0.as<CallNode>()->op == multiply_op_ ) {
        Expr tmp = tensor_0;
        tensor_0 = tensor_1;
        tensor_1 = tmp;
    } else {
      return post;
    }

    Expr tensor_2 = tensor_1.as<CallNode>()->args[0];
    if (!tensor_2.as<CallNode>()) {
      tensor_2 = tensor_1.as<CallNode>()->args[1];
      if (!tensor_2.as<CallNode>()) return post;
    }
    if (tensor_2.as<CallNode>()->op != add_op_) return post;

    Expr tensor_3 = tensor_2.as<CallNode>()->args[0];
    if (!tensor_3.as<CallNode>()) {
      tensor_3 = tensor_2.as<CallNode>()->args[1];
      if (!tensor_3.as<CallNode>()) return post;
    }
    if (tensor_3.as<CallNode>()->op != tanh_op_) return post;

    Expr tensor_4 = tensor_3.as<CallNode>()->args[0];
    if (!tensor_4.as<CallNode>()) return post;
    if (tensor_4.as<CallNode>()->op != multiply_op_) return post;

    Expr tensor_5 = tensor_4.as<CallNode>()->args[0];
    if (!tensor_5.as<CallNode>()) {
      tensor_5 = tensor_4.as<CallNode>()->args[1];
      if (!tensor_5.as<CallNode>()) return post;
    }
    if (tensor_5.as<CallNode>()->op != add_op_) return post;

    Expr tensor_6 = tensor_5.as<CallNode>()->args[0];
    Expr tensor_7 = tensor_5.as<CallNode>()->args[1];
    if (tensor_6 != tensor_0) {
      Expr tmp = tensor_6;
      tensor_6 = tensor_7;
      tensor_7 = tmp;
    }

    if (tensor_6 != tensor_0) return post;
    if (!tensor_7.as<CallNode>()) return post;
    if (tensor_7.as<CallNode>()->op != multiply_op_) return post;

    Expr tensor_8 = tensor_7.as<CallNode>()->args[0];
    if (!tensor_8.as<CallNode>()) {
      tensor_8 = tensor_7.as<CallNode>()->args[1];
      if (!tensor_8.as<CallNode>()) return post;
    }
    if (tensor_8.as<CallNode>()->op != power_op_) return post;

    Expr tensor_9 = tensor_8.as<CallNode>()->args[0];
    if (tensor_9 != tensor_0) return post;

    bool fast = false;
    if(fast) {
    // x * sigmoid(1.702 * x)
      runtime::NDArray the_array = runtime::NDArray::Empty({1}, DataType::Float(32U), {kDLCPU, 0});
      DLTensor the_dltensor;
      the_dltensor.data = (void*)(new float_t[1]);
      ((float*)(the_dltensor.data))[0] = 1.702;
      the_dltensor.device.device_type = kDLCPU;
      the_dltensor.device.device_id = 0;
      the_dltensor.ndim = 1;
      the_dltensor.dtype.code = kDLFloat;
      the_dltensor.dtype.bits = 32U;
      the_dltensor.dtype.lanes = 1U;
      the_dltensor.shape = new int64_t[1];
      the_dltensor.shape[0] = 1;
      the_dltensor.strides = nullptr;
      the_dltensor.byte_offset = 0;
      the_array.CopyFrom(&the_dltensor);
      Constant the_const = Constant(the_array);
      Expr multiplied = relay::Call(multiply_op_, {tensor_0, the_const});

      Expr sigmoided = relay::Call(Op::Get("sigmoid"), {multiplied});

      return relay::Call(multiply_op_, {tensor_0, sigmoided});
    } else {
      auto attrs = make_object<ReshapeAttrs>();
      auto tensor_type_input = pre->checked_type().as<TensorTypeNode>();
      int dim_number = tensor_type_input->shape.size();
      Array<Integer> the_shape;
      for(int i = 0; i < dim_number; ++i) {
        the_shape.push_back(Integer(tensor_type_input->shape[i].as<IntImmNode>()->value));
      }
      attrs->newshape = the_shape;
      Expr after_GELU = relay::Call(AIPU_GELU_op_, {tensor_0}, Attrs(attrs), {});
      return after_GELU;
    }

  }

 private:
  const Op& power_op_;
  const Op& multiply_op_;
  const Op& add_op_;
  const Op& tanh_op_;
  const Op& AIPU_GELU_op_ ;
};

Expr AIPU_fuse_GELU(const Expr& expr) {
  auto rewriter = AIPU_fuse_GELU_rewritter();
  return PostOrderRewrite(expr, &rewriter);
}

namespace transform {

Pass AIPU_fuse_GELU() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto r = Downcast<Function>(AIPU_fuse_GELU(f));
        return r;
      };
  return CreateFunctionPass(pass_func, 1000, "AIPU_fuse_GELU", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.AIPU_fuse_GELU").set_body_typed(AIPU_fuse_GELU);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
