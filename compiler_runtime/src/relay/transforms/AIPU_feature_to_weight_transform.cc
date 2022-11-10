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
 * \file AIPU_feature_to_weight_transform.cc
 * \brief DENSE is executed on NVDLA. If the weight of DENSE is a feature data, we need to convert feature data to weight data. This pass does this job.
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "./pattern_utils.h"

#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

class FeatureToWeightTransform_rewritter : public ExprRewriter {
 public:
  explicit FeatureToWeightTransform_rewritter()
      :dense_op_(Op::Get("nn.dense")), featuretoweight_op_(Op::Get("featuretoweight")) {}

  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    if (pre->op == dense_op_) {
      std::cout << "This is an nn.dense CallNode" << std::endl;
      auto tensor_a = post.as<CallNode>()->args[0];
      auto tensor_b = post.as<CallNode>()->args[1];
      if(!( (pre->checked_type_).defined() && (pre->args[0]->checked_type_).defined() && (pre->args[1]->checked_type_).defined() )){
        std::cout << "FATAL!!! The checked_type_ field is not available. Consider do InferType before this pass." << std::endl;
        exit(0);
      }

      if(pre->args[1].as<ConstantNode>()) {
        std::cout << "This is a classical dense, do nothing" << std::endl;
        return post;
      }

      auto checked_type = pre->checked_type();
      const auto* tensor_type = checked_type.as<TensorTypeNode>();
      std::cout << "Type information: " << checked_type->GetTypeKey() << " " << tensor_type->shape << " " << tensor_type->dtype << std::endl;

      auto checked_type_a = pre->args[0]->checked_type();
      const auto* tensor_type_a = checked_type_a.as<TensorTypeNode>();
      if(!tensor_type_a){
        return post;
      }
      std::cout << "Type information of data: " << checked_type_a->GetTypeKey() << " " << tensor_type_a->shape << " " << tensor_type_a->dtype << std::endl;

      auto checked_type_b = pre->args[1]->checked_type();
      const auto* tensor_type_b = checked_type_b.as<TensorTypeNode>();
      if(!tensor_type_b){
        return post;
      }
      std::cout << "Type information of weight (as feature): " << checked_type_b->GetTypeKey() << " " << tensor_type_b->shape << " " << tensor_type_b->dtype << std::endl;

      int dim_n = (tensor_type_b->shape[0]).as<IntImmNode>()->value;
      int dim_c = (tensor_type_b->shape[1]).as<IntImmNode>()->value;
      // int dim_h = 1;
      // int dim_w = 1;
      auto attrs = make_object<ReshapeAttrs>();
      attrs->newshape = Array<Integer>({dim_n, dim_c});
      Call tensor_b_transformed = relay::Call(featuretoweight_op_, {tensor_b}, Attrs(attrs), {});

      // auto attrs_dense = make_object<DenseAttrs>();
      Expr result = relay::Call(dense_op_, {tensor_a, tensor_b_transformed}, Attrs(pre->attrs), {});
      return result;
    }
    return post;
  }

 private:
  const Op& dense_op_;
  const Op& featuretoweight_op_;
};

Expr FeatureToWeightTransform(const Expr& expr) {
  auto rewriter = FeatureToWeightTransform_rewritter();
  return PostOrderRewrite(expr, &rewriter);
}

namespace transform {

Pass FeatureToWeightTransform() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto r = Downcast<Function>(FeatureToWeightTransform(f));
        return r;
      };
  return CreateFunctionPass(pass_func, 1000, "FeatureToWeightTransform", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FeatureToWeightTransform").set_body_typed(FeatureToWeightTransform);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
