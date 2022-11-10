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
 * \file AIPU_extract_subfuctions.cc
 * \brief extract subfunctions
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include "./pattern_utils.h"
#include <tvm/relay/expr.h>
#include <tvm/relay/qnn/attrs.h>

#include <string>
#include <sstream>
#include <fstream>

namespace tvm {
namespace relay {

class subfunctions_writer : public ExprRewriter {
 public:
  explicit subfunctions_writer()
      :batch_matmul_op_(Op::Get("nn.batch_matmul")),
       quantize_op_(relay::Op::Get("relay.op.annotation.simulated_quantize")),
       reshape_op_(relay::Op::Get("reshape")),
       split_op_(relay::Op::Get("split")),
       dense_op_(relay::Op::Get("nn.dense")),
       cat_op_(relay::Op::Get("concatenate")),
       add_op_(relay::Op::Get("add")),
       global_counter(0)
  {}

  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    if(const auto* func = pre->op.as<FunctionNode>()) {
      std::string compiler = func->GetAttr<String>(tvm::relay::attr::kCompiler).value();
      if(compiler == "riscv") {
        auto func_ = tvm::runtime::GetRef<Function>(func);
        std::stringstream ss;
        ss.str("");
        ss << "model" << global_counter << ".txt";
        std::string name = ss.str();
        std::fstream fs(name, std::ios::in|std::ios::out|std::ios::trunc);
        fs << AsText(func_, true); 
        global_counter += 1;
      }
    }

    return post;
  }

 private:
  const Op& batch_matmul_op_;
  const Op& quantize_op_;
  const Op& reshape_op_;
  const Op& split_op_;
  const Op& dense_op_;
  const Op& cat_op_;
  const Op& add_op_;
  int global_counter;
};

Expr subfunctionsWriter(const Expr& expr) {
  auto rewriter = subfunctions_writer();
  return PostOrderRewrite(expr, &rewriter);
}

namespace transform {

Pass subfunctionsWriterTransform() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto r = Downcast<Function>(subfunctionsWriter(f));
        return r;
      };
  return CreateFunctionPass(pass_func, 1000, "subfunctionsWriterTransform", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.subfunctionsWriterTransform").set_body_typed(subfunctionsWriterTransform);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
