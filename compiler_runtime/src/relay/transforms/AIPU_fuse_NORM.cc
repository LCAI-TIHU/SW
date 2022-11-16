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
 * \file AIPU_fuse_NORM.cc
 * This pass fuse operators to "AIPU_NORM" operator.
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "./pattern_utils.h"

#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

class AIPU_fuse_NORM_rewritter : public ExprRewriter {
 public:
  explicit AIPU_fuse_NORM_rewritter() :
      multiply_op_(Op::Get("multiply")),
      add_op_(Op::Get("add")),
      subtract_op_(Op::Get("subtract")),
      AIPU_NORM_op_(Op::Get("AIPU_NORM")) {}

  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    if (post.as<CallNode>()->op != add_op_) return post;

    Expr tensor_0 = post.as<CallNode>()->args[0];
    Expr tensor_1 = post.as<CallNode>()->args[1];
    if (! (tensor_0.as<CallNode>())) return post;
    if (! (tensor_1.as<CallNode>())) return post;

    if (tensor_0.as<CallNode>()->op != multiply_op_ ) return post;
    if (tensor_1.as<CallNode>()->op != subtract_op_ ) return post;
    Expr add_weight = tensor_1.as<CallNode>()->args[0];

    Expr tensor_2 = tensor_0.as<CallNode>()->args[1];
    if (tensor_2.as<CallNode>()->op != multiply_op_ ) return post;
    Expr multiply_weight = tensor_2.as<CallNode>()->args[1];

    Expr tensor_3 = tensor_0.as<CallNode>()->args[0];

    auto attrs = make_object<LayerNormAttrs>();
    attrs->axis = -1;
    Expr after_NORM = relay::Call(AIPU_NORM_op_, {tensor_3}, Attrs(attrs), {});

    Expr after_multiply = relay::Call(multiply_op_, {after_NORM, multiply_weight});
    Expr after_add = relay::Call(add_op_, {after_multiply, add_weight});
    return after_add;
  }

 private:
  const Op& multiply_op_;
  const Op& add_op_;
  const Op& subtract_op_;
  const Op& AIPU_NORM_op_ ;
};

Expr AIPU_fuse_NORM(const Expr& expr) {
  auto rewriter = AIPU_fuse_NORM_rewritter();
  return PostOrderRewrite(expr, &rewriter);
}

namespace transform {

Pass AIPU_fuse_NORM() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto r = Downcast<Function>(AIPU_fuse_NORM(f));
        return r;
      };
  return CreateFunctionPass(pass_func, 1000, "AIPU_fuse_NORM", {});
}

TVM_REGISTER_GLOBAL("relay._transform.AIPU_fuse_NORM").set_body_typed(AIPU_fuse_NORM);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
