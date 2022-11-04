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

/*!
 * \file transform.cc
 * \brief Transform operators.
 */
#include "transform.h"

#include <tvm/ir/error.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/data_layout.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/detail/constant_utils.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/nn.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/transform.h>

#include <vector>

#include "../../transforms/infer_layout_utils.h"
#include "../../transforms/pass_utils.h"
#include "../../transforms/pattern_utils.h"
#include "../make_op.h"
#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {
using tir::IntImmNode;

bool FeaturetoweightRel(const Array<Type>& types, int num_inputs, const Attrs& attrs, const TypeReporter& reporter) {
  // types: [data, result]
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "reshape: expect input type to be TensorType but get " << types[0];
    return false;
  }

  // This function is in transform.cc
  const auto& oshape = InferNewShape(data->shape, attrs, false);

  Array<IndexExpr> data_shape;
  data_shape = data->shape;

  //ICHECK_EQ(oshape.size(), 4) << "Input tensor shape and reshaped shape are not compatible";
  //ICHECK_EQ(data_shape.size(), 4) << "Input tensor shape and reshaped shape are not compatible";
  ICHECK_EQ(oshape.size(), data_shape.size()) << "Input tensor shape and reshaped shape are not compatible";

  int64_t oshape_ele = 1;
  int64_t data_shape_ele = 1;
  for(size_t i = 0; i < oshape.size(); ++ i) {
    oshape_ele = Downcast<tvm::Integer>(oshape[i])->value;
    data_shape_ele = Downcast<tvm::Integer>(data_shape[i])->value;
    ICHECK_EQ(oshape_ele, data_shape_ele) << "Input tensor shape and reshaped shape are not compatible";
  }

  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> featuretoweightCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                 const Type& out_type) {

  const auto* out_ttype = out_type.as<TensorTypeNode>();
  ICHECK(out_ttype != nullptr);
  Array<IndexExpr> newshape;
  bool newshape_has_any = false;
  for (auto val : out_ttype->shape) {
    if (val->IsInstance<tir::AnyNode>() || val->IsInstance<tir::VarNode>()) {
      newshape_has_any = true;
      break;
    } else {
      newshape.push_back(val);
    }
  }

  if (newshape_has_any) {
    newshape = InferNewShape(inputs[0]->shape, attrs, false);
  }
  // TODO give a correct implementation
  return {topi::reshape(inputs[0], newshape)};
}

Expr MakeFeaturetoweight(Expr data, Array<Integer> newshape) {
  auto attrs = make_object<ReshapeAttrs>();
  attrs->newshape = std::move(newshape);
  static const Op& op = Op::Get("featuretoweight");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.featuretoweight").set_body_typed(MakeFeaturetoweight);

RELAY_REGISTER_OP("featuretoweight")
    .describe(R"code(convert an NVDLA feature data (N{C/32}HW{32}) to an NVDLA weight data (N{C/64}HW{64}).
)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<ReshapeAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Featuretoweight", FeaturetoweightRel)
    .set_attr<FTVMCompute>("FTVMCompute", featuretoweightCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace relay
}  // namespace tvm
