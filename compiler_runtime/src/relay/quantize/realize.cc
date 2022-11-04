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
 *
 * \file realize.cc
 *
 * \brief Realizing the simulated graph into real low-precision
 *   graph.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/transform.h>

#include "../qnn/utils.h"
#include "../transforms/pattern_utils.h"
#include "./quantize.h"

namespace tvm {
namespace relay {
namespace quantize {

using namespace relay::transform;

class QRealizeExpr;
class QRealizeIntExpr;

class QRealizeExprNode : public TempExprNode {
 public:
  Expr data;
  static constexpr const char* _type_key = "relay.quantize.QRealizeExpr";
  TVM_DECLARE_BASE_OBJECT_INFO(QRealizeExprNode, TempExprNode);
};

class QRealizeExpr : public TempExpr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(QRealizeExpr, TempExpr, QRealizeExprNode);
};

class QRealizeIntExprNode : public QRealizeExprNode {
 public:
  Expr dom_scale;
  DataType dtype;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("data", &data);
    v->Visit("dom_scale", &dom_scale);
    v->Visit("dtype", &dtype);
  }

  Expr Realize() const final;

  static constexpr const char* _type_key = "relay.quantize.QRealizeIntExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(QRealizeIntExprNode, QRealizeExprNode);
};

class QRealizeIntExpr : public QRealizeExpr {
 public:
  TVM_DLL QRealizeIntExpr(Expr data, Expr dom_scale, DataType dtype);

  TVM_DEFINE_OBJECT_REF_METHODS(QRealizeIntExpr, QRealizeExpr, QRealizeIntExprNode);
};

Expr QRealizeIntExprNode::Realize() const {
  Expr data = this->data;
  // dequantize
  data = Cast(data, DataType::Float(32));
  data = Multiply(data, this->dom_scale);
  return data;
}

QRealizeIntExpr::QRealizeIntExpr(Expr data, Expr dom_scale, DataType dtype) {
  ObjectPtr<QRealizeIntExprNode> n = make_object<QRealizeIntExprNode>();
  n->data = std::move(data);
  n->dom_scale = std::move(dom_scale);
  n->dtype = std::move(dtype);
  data_ = std::move(n);
}

inline Expr ForwardOp(const Call& ref_call, const Array<Expr>& args) {
  return Call(ref_call->op, args, ref_call->attrs, ref_call->type_args);
}

/* calculate `data * s1 / s2`, use shift if possible */
inline Expr MulAndDiv(Expr data, Expr s1, Expr s2, DataType dtype,
                      const Array<IndexExpr>& data_shape) {
  const QConfig& cfg = QConfig::Current();
  // here we assume the dtype of data is dtype activation
  // if (s1 == s2) return data;
  
  // if (!IsConstScalar(s1) || !IsConstScalar(s2)) {
  if (cfg->weight_scale == "channel_max") {
    // This is per-channel (per=axis) MulAndDiv.
    std::vector<double> double_multipliers;
    auto s1_vector = qnn::GetFloatVectorFromConstant(s1);
    auto s2_vector = qnn::GetFloatVectorFromConstant(s2);
    for (int i = 0; i < s1_vector.size(); i++) {
      double multiplier =
          static_cast<double>(s1_vector[i]) / static_cast<double>(s2_vector[i]);
      double_multipliers.push_back(multiplier);
    }

    int axis = data_shape.size() - 1; //TODO Abraham: Can it be compatible with Depthwise Convolution?
    // TODO Jasper: can do it, s1_vector.size() == s2_vector.size()
    data = qnn::FixedPointMultiplyPerChannel(data, double_multipliers, data_shape,
                                                      axis, cfg->rounding);
    return Cast(data, dtype);
  } else {
    float s1_float = GetScalarFromConstant<float>(s1);
    float s2_float = GetScalarFromConstant<float>(s2);
    if (s1_float == s2_float) return data;

    float factor = s1_float / s2_float;
    float shift_factor = std::log2(factor);
    // ICHECK_GT(shift_factor, 0);
    if (static_cast<int>(shift_factor) == shift_factor) {
      return LeftShift(data, MakeConstantScalar(dtype, static_cast<int>(shift_factor)));
    } else if (static_cast<int>(factor) == factor) {
      return Multiply(data, MakeConstantScalar(dtype, factor));
    } else {
      if (cfg->rounding == "UPWARD") {
        int32_t fixed_point_multiplier, shift;
        std::tie(fixed_point_multiplier, shift) = qnn::GetFixedPointMultiplierShift(factor);
        data = relay::FixedPointMultiply(data, fixed_point_multiplier, shift);
      } else {
        data = qnn::FixedPointMultiplyToNearest(data, factor, data_shape);
      }
      return Cast(data, dtype);
    }
  }
}

Expr QuantizeRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  // do not handle data type cast
  const auto param = ref_call->attrs.as<SimulatedQuantizeAttrs>();
  ICHECK_EQ(param->rounding, "round");

  Expr dom_scale = new_args[1];
  Expr clip_min = new_args[2];
  Expr clip_max = new_args[3];

  float clip_min_imm = GetScalarFromConstant<float>(clip_min);
  float clip_max_imm = GetScalarFromConstant<float>(clip_max);

  int axis = param->axis;
  axis = (axis == -1) ? ref_call->type_as<TensorTypeNode>()->shape.size() - 1 : axis;

  // x * idom_scale = y * odom_scale
  // => y = x * idom_scale / odom_scale
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    // int32->int8
    Expr data = n->data;
    float odom_scale_imm = GetScalarFromConstant<float>(dom_scale);
    if (IsConstScalar(n->dom_scale)) {
      float idom_scale_imm = GetScalarFromConstant<float>(n->dom_scale);
      if (idom_scale_imm == odom_scale_imm) {
        // same domain scale, only clip
        data = Clip(data, clip_min_imm, clip_max_imm);
        return QRealizeIntExpr(data, dom_scale, n->dtype);
      }

      float shift_nbit = std::log2(odom_scale_imm / idom_scale_imm);
      ICHECK_NE(shift_nbit, 0);
      if (static_cast<int>(shift_nbit) == shift_nbit) {
        if (shift_nbit > 0) {
          // use right shift
          if (cfg->round_for_shift) {
            float round_bias = std::pow(2.0, shift_nbit - 1);
            data = Add(data, MakeConstantScalar(cfg->dtype_activation, static_cast<int>(round_bias)));
          }
          data = RightShift(data,
                            MakeConstantScalar(cfg->dtype_activation, static_cast<int>(shift_nbit)));
        } else {
          data = LeftShift(data,
                          MakeConstantScalar(cfg->dtype_activation, static_cast<int>(-shift_nbit)));
        }
        data = Clip(data, clip_min_imm, clip_max_imm);
        return QRealizeIntExpr(data, dom_scale, n->dtype);
      } else {
        data = Cast(data, DataType::Int(64));
        if (cfg->rounding == "UPWARD") {
          int32_t fixed_point_multiplier, shift;
          std::tie(fixed_point_multiplier, shift) =
              qnn::GetFixedPointMultiplierShift(idom_scale_imm / odom_scale_imm);
          data = relay::FixedPointMultiply(data, fixed_point_multiplier, shift);
        } else {
          data = qnn::FixedPointMultiplyToNearest(data, idom_scale_imm / odom_scale_imm,
                                                  ref_call->type_as<TensorTypeNode>()->shape);
        }
        data = Cast(Clip(data, clip_min_imm, clip_max_imm), n->dtype);
        return QRealizeIntExpr(data, dom_scale, n->dtype);
      }
    } else {
      // This is per-channel (per=axis) quantization.
      std::vector<double> double_multipliers;
      auto input_axis_scales = qnn::GetFloatVectorFromConstant(n->dom_scale);
      for (auto input_axis_scale : input_axis_scales) {
        double multiplier =
            static_cast<double>(input_axis_scale) / static_cast<double>(odom_scale_imm);
        double_multipliers.push_back(multiplier);
      }
      //LOG(INFO) << "----------ref_call->type_as<TensorTypeNode>()->shape----------" << std::endl;
      //LOG(INFO) << ref_call->type_as<TensorTypeNode>()->shape << std::endl;
      data = qnn::FixedPointMultiplyPerChannel(data, double_multipliers, ref_call->type_as<TensorTypeNode>()->shape,
                                                    axis, cfg->rounding);
      data = Cast(Clip(data, clip_min_imm, clip_max_imm), n->dtype);
      return QRealizeIntExpr(data, dom_scale, n->dtype);
    }
  }
  // quantize from real
  ICHECK(!new_args[0]->IsInstance<TempExprNode>());
  Expr data = new_args[0];
  
  if (IsConstScalar(dom_scale)) {
    float dom_scale_imm = GetScalarFromConstant<float>(dom_scale);  
    Expr scaled_data = Multiply(data, MakeConstantScalar(DataType::Float(32), 1 / dom_scale_imm));
    Expr round_data = Clip(Round(scaled_data), clip_min_imm, clip_max_imm);
    return QRealizeIntExpr(round_data, dom_scale, DataType::Float(32));
  } else {
    auto input_axis_scales = qnn::GetFloatVectorFromConstant(dom_scale);
    std::vector<int64_t> shape;
    size_t n_dim = data->type_as<TensorTypeNode>()->shape.size();
    if (axis == 2) { // Jasper: depthwise op, [kernel_h, kernel_w, input_nums, output_nums]
      shape.push_back(ref_call->type_as<TensorTypeNode>()->shape[axis].as<IntImmNode>()->value);
      shape.push_back(ref_call->type_as<TensorTypeNode>()->shape[axis+1].as<IntImmNode>()->value);
    } else if (axis == 0 && n_dim == 2) { // Jasper: dense op, [input_nums, output_nums]
      shape.push_back(ref_call->type_as<TensorTypeNode>()->shape[axis].as<IntImmNode>()->value);
      shape.push_back(1);
    } else {
      shape.push_back(ref_call->type_as<TensorTypeNode>()->shape[axis].as<IntImmNode>()->value);
    }
    
    Expr expanded_scale = MakeConstantTensor(DataType::Float(32), shape, input_axis_scales);
    Expr scaled_data = Divide(data, expanded_scale);
    Expr round_data = Clip(Round(scaled_data), clip_min_imm, clip_max_imm);
    return QRealizeIntExpr(round_data, dom_scale, DataType::Float(32));                                      
  }
}

Expr FoldConstantOpt(const Expr& expr) {
  auto mod = IRModule::FromExpr(expr);
  mod = transform::FoldConstant()(mod);
  auto entry_func = Downcast<Function>(mod->Lookup("main"));
  return expr.as<FunctionNode>() == nullptr ? entry_func->body : entry_func;
}

RELAY_REGISTER_OP("relay.op.annotation.simulated_quantize")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", QuantizeRealize);

Expr Conv2dRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  ICHECK_EQ(new_args.size(), 2);
  if (!new_args[0]->IsInstance<TempExprNode>() && !new_args[1]->IsInstance<TempExprNode>()) {
    return Expr(nullptr);
  }
  const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
  ICHECK(lhs);
  const auto* rhs = new_args[1].as<QRealizeIntExprNode>();
  ICHECK(rhs);

  Expr ldata = lhs->data;
  if (lhs->dtype != cfg->dtype_input) {
    ldata = Cast(ldata, cfg->dtype_input);
  }
  Expr rdata = Cast(rhs->data, cfg->dtype_weight);

  const auto ref_attrs = ref_call->attrs.as<Conv2DAttrs>();
  auto attrs = make_object<Conv2DAttrs>();
  *attrs = *ref_attrs;
  DataType out_dtype = cfg->dtype_activation;
  attrs->out_dtype = out_dtype;

  Expr ret = Call(ref_call->op, {ldata, rdata}, Attrs(attrs), ref_call->type_args);
  Expr mul = Multiply(lhs->dom_scale, rhs->dom_scale);
  Expr dom_scale = FoldConstantOpt(mul);
  return QRealizeIntExpr(ret, dom_scale, out_dtype);
}

RELAY_REGISTER_OP("nn.conv2d").set_attr<FForwardRewrite>("FQRealizeRewrite", Conv2dRealize);

Expr Conv1dRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 2);
  if (!new_args[0]->IsInstance<TempExprNode>() && !new_args[1]->IsInstance<TempExprNode>()) {
    return Expr(nullptr);
  }
  const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
  CHECK(lhs);
  const auto* rhs = new_args[1].as<QRealizeIntExprNode>();
  CHECK(rhs);

  Expr ldata = lhs->data;
  if (lhs->dtype != cfg->dtype_input) {
    ldata = Cast(ldata, cfg->dtype_input);
  }
  Expr rdata = Cast(rhs->data, cfg->dtype_weight);

  const auto ref_attrs = ref_call->attrs.as<Conv1DAttrs>();
  auto attrs = make_object<Conv1DAttrs>();
  *attrs = *ref_attrs;
  DataType out_dtype = cfg->dtype_activation;
  attrs->out_dtype = out_dtype;

  Expr ret = Call(ref_call->op, {ldata, rdata}, Attrs(attrs), ref_call->type_args);
  Expr mul = Multiply(lhs->dom_scale, rhs->dom_scale);
  Expr dom_scale = FoldConstantOpt(mul);
  return QRealizeIntExpr(ret, dom_scale, out_dtype);
}

RELAY_REGISTER_OP("nn.conv1d").set_attr<FForwardRewrite>("FQRealizeRewrite", Conv1dRealize);

Expr DenseRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  ICHECK_EQ(new_args.size(), 2);
  if (!new_args[0]->IsInstance<TempExprNode>() || !new_args[1]->IsInstance<TempExprNode>()) {
    return Expr(nullptr);
  }
  const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
  const auto* rhs = new_args[1].as<QRealizeIntExprNode>();

  Expr ldata = lhs->data;
  if (lhs->dtype != cfg->dtype_input) {
    ldata = Cast(ldata, cfg->dtype_input);
  }
  Expr rdata = Cast(rhs->data, cfg->dtype_weight);

  const auto ref_attrs = ref_call->attrs.as<DenseAttrs>();
  auto attrs = make_object<DenseAttrs>();
  *attrs = *ref_attrs;
  DataType out_dtype = cfg->dtype_activation;
  attrs->out_dtype = out_dtype;

  Expr ret = Call(ref_call->op, {ldata, rdata}, Attrs(attrs), ref_call->type_args);
  Expr mul = Multiply(lhs->dom_scale, rhs->dom_scale);
  Expr dom_scale = FoldConstantOpt(mul);
  return QRealizeIntExpr(ret, dom_scale, out_dtype);
}

RELAY_REGISTER_OP("nn.dense").set_attr<FForwardRewrite>("FQRealizeRewrite", DenseRealize);

Expr MulRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  ICHECK_EQ(new_args.size(), 2);
  if (new_args[0].as<QRealizeIntExprNode>() && new_args[1].as<QRealizeIntExprNode>()) {
    // execute the operation with activation data type.
    const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
    const auto* rhs = new_args[1].as<QRealizeIntExprNode>();
    Expr ldata = lhs->data;
    Expr rdata = rhs->data;

    DataType dtype = cfg->dtype_activation;
    if (lhs->dtype != dtype) {
      ldata = Cast(ldata, dtype);
    }
    if (rhs->dtype != dtype) {
      rdata = Cast(rdata, dtype);
    }

    Expr ret = ForwardOp(ref_call, {ldata, rdata});
    Expr mul = Multiply(lhs->dom_scale, rhs->dom_scale);
    Expr dom_scale = FoldConstantOpt(mul);
    return QRealizeIntExpr(ret, dom_scale, dtype);
  }
  ICHECK(!new_args[0]->IsInstance<TempExprNode>() || !new_args[1]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("multiply").set_attr<FForwardRewrite>("FQRealizeRewrite", MulRealize);

float ChooseDomScale(const std::vector<const QRealizeIntExprNode*>& nptrs) {
  if (nptrs.size() == 2) {
    // x = a * s1, y = b * s2
    // x + y = (a * s1 / s2 + b) * s2, if s1 > s2
    //       = (a + b * s2 / s1) * s1, if s2 > s1
    float s1 = GetScalarFromConstant<float>(nptrs[0]->dom_scale);
    float s2 = GetScalarFromConstant<float>(nptrs[1]->dom_scale);
    return s1 > s2 ? s2 : s1;
  } else {
    const QConfig& cfg = QConfig::Current();
    float scale = cfg->global_scale;
    return scale / std::pow(2.0, cfg->nbit_activation - 1);
  }
}

Expr MergeDomScale(const std::vector<const QRealizeIntExprNode*>& nptrs) {
  if (nptrs.size() == 2) {
    // x = a * s1, y = b * s2
    // x + y = (a * s1 / s2 + b) * s2, if s1 > s2
    //       = (a + b * s2 / s1) * s1, if s2 > s1
    LOG(INFO) << "--------------MergeDomScale1-------------" << std::endl;
    LOG(INFO) << "nptrs[0]: " << nptrs[0] << std::endl;
    LOG(INFO) << "nptrs[1]: " << nptrs[1] << std::endl;
    LOG(INFO) << "IsConstScalar(nptrs[0]->dom_scale: " << IsConstScalar(nptrs[0]->dom_scale) << std::endl;
    LOG(INFO) << "IsConstScalar(nptrs[1]->dom_scale: " << IsConstScalar(nptrs[1]->dom_scale) << std::endl;
    if (IsConstScalar(nptrs[0]->dom_scale) && IsConstScalar(nptrs[1]->dom_scale)) {
      LOG(INFO) << "GetScalarFromConstant<float>(nptrs[0]->dom_scale): " << GetScalarFromConstant<float>(nptrs[0]->dom_scale);
      LOG(INFO) << "GetScalarFromConstant<float>(nptrs[1]->dom_scale): " << GetScalarFromConstant<float>(nptrs[1]->dom_scale);

      float s1 = GetScalarFromConstant<float>(nptrs[0]->dom_scale);
      float s2 = GetScalarFromConstant<float>(nptrs[1]->dom_scale);
      return MakeConstantScalar(DataType::Float(32), s1 > s2 ? s2 : s1);
      // return MakeConstantScalar(DataType::Float(32), s2);
    } else if (!IsConstScalar(nptrs[0]->dom_scale) && !IsConstScalar(nptrs[1]->dom_scale)) {
      if (nptrs[0]->dom_scale->type_as<TensorTypeNode>()->shape == nptrs[0]->dom_scale->type_as<TensorTypeNode>()->shape) {
        auto scales1 = qnn::GetFloatVectorFromConstant(nptrs[0]->dom_scale);
        auto scales2 = qnn::GetFloatVectorFromConstant(nptrs[1]->dom_scale);
        
        LOG(INFO) << "--------------MergeDomScale2-------------" << std::endl;
        // std::vector<float> merged_scales{scales2};
        std::vector<float> merged_scales;
        for (int i = 0; i < scales1.size(); i++) {
          // float scale2 = std::fabs(scales2[i]);
          merged_scales.push_back(scales1[i] > scales2[i] ? scales2[i] : scales1[i]);
          LOG(INFO) << "scales1 = " << scales1[i] << " scales2 = " << scales2[i] << " merged_scale = " << merged_scales[i] << std::endl;
        }
        
        return MakeConstantTensor(DataType::Float(32), {(long)scales1.size()}, merged_scales);
      } else {
        LOG(FATAL) << "Quantization realization cannot merge nptrs[0]->dom_scale and nptrs[1]->dom_scale with different shapes";
      }
    } else if (!IsConstScalar(nptrs[0]->dom_scale) && IsConstScalar(nptrs[1]->dom_scale)) {
      auto scales1 = qnn::GetFloatVectorFromConstant(nptrs[0]->dom_scale);
      float s2 = GetScalarFromConstant<float>(nptrs[1]->dom_scale);
      
      LOG(INFO) << "--------------MergeDomScale3-------------" << std::endl;
      // std::vector<float> merged_scales{scales1};
      std::vector<float> merged_scales;
      for (int i = 0; i < scales1.size(); i++) {
        merged_scales.push_back(scales1[i] > s2 ? s2 : scales1[i]);
        // merged_scales.push_back(scales1[i] + s2); //Abraham: Negative scale is okay? Where does the abs function take place?
        LOG(INFO) << "scales1 = " << scales1[i] << " s2 = " << s2 << " merged_scale = " << merged_scales[i] << std::endl;
      }
      
      return MakeConstantTensor(DataType::Float(32), {(long)scales1.size()}, merged_scales);
    } else if (IsConstScalar(nptrs[0]->dom_scale) && !IsConstScalar(nptrs[1]->dom_scale)) {
      float s1 = GetScalarFromConstant<float>(nptrs[0]->dom_scale);
      auto scales2 = qnn::GetFloatVectorFromConstant(nptrs[1]->dom_scale);
      
      LOG(INFO) << "--------------MergeDomScale3-------------" << std::endl;
      // std::vector<float> merged_scales{scales2};
      std::vector<float> merged_scales;
      for (int i = 0; i < scales2.size(); i++) {
        merged_scales.push_back(scales2[i] > s1 ? s1 : scales2[i]);
        // merged_scales.push_back(s1 + scales2[i]); //Abraham: Negative scale is okay? Where does the abs function take place?
        LOG(INFO) << "scales2 = " << scales2[i] << " s1 = " << s1 << " merged_scale = " << merged_scales[i] << std::endl;
      }
      
      return MakeConstantTensor(DataType::Float(32), {(long)scales2.size()}, merged_scales);
    } else {
      return nptrs[0]->dom_scale;
    }
  } else { //Abraham TODO: It should use a smarter way to handle the case that nptrs.size() != 2. 
    // const QConfig& cfg = QConfig::Current();
    // float scale = cfg->global_scale;
    // return MakeConstantScalar(DataType::Float(32), scale / std::pow(2.0, cfg->nbit_activation - 1));

    LOG(INFO) << "nptrs.size(): " << nptrs.size() << std::endl;
    float min_scale = std::numeric_limits<float>::max();
    for (int i=0; i < nptrs.size(); i++) {
      if (IsConstScalar(nptrs[i]->dom_scale)) {
        float s = GetScalarFromConstant<float>(nptrs[i]->dom_scale);
        LOG(INFO) << "s: " << s << " min_scale: " << min_scale << std::endl;
        min_scale = s > min_scale ? min_scale : s;
      } else {
        return nptrs[i]->dom_scale;
      }
    }
    return MakeConstantScalar(DataType::Float(32), min_scale);
  }
}

/* \brief Unify the dom scale of arguments */
Array<Expr> UnifyDTypeScale(const Array<Expr>& ref_args, const Array<Expr>& args,
                            DataType* dtype_ptr, Expr* scale_ptr,
                            DataType dtype = DataType::Void()) {
  static const Op& simulated_quantize = Op::Get("relay.op.annotation.simulated_quantize");
  const QConfig& cfg = QConfig::Current();

  std::vector<const QRealizeIntExprNode*> nptrs;
  Array<Expr> ret;
  for (auto arg : args) {
    const auto* nptr = arg.as<QRealizeIntExprNode>();
    ICHECK(nptr);
    nptrs.push_back(nptr);
    ret.push_back(nptr->data);
  }

  // unify the data type
  ICHECK_EQ(ref_args.size(), args.size());

  if (dtype.is_void()) {
    if (ret.size() == 2 && nptrs[1]->dtype == cfg->dtype_input) {
      dtype = cfg->dtype_input;
    } else {
      dtype = cfg->dtype_activation;
    }
  }

  for (size_t i = 0; i < ret.size(); ++i) {
    auto ref_arg = ref_args[i].as<CallNode>();
    if (nptrs[i]->dtype != dtype) {
      ret.Set(i, Cast(ret[i], dtype));
    } else if (ref_arg && ref_arg->op.same_as(simulated_quantize) &&
               ref_arg->attrs.as<SimulatedQuantizeAttrs>()->kind == kQInput) {
      auto new_arg = Cast(ret[i], cfg->dtype_input);
      new_arg = StopFusion(new_arg);
      ret.Set(i, Cast(new_arg, dtype));
    }
  }
  
  // Abraham's way:
  // 1. Replace ChooseDomScale with MergeDomScale.
  // 2. In case that the realized computing graph is too complicated, the following lines build a 
  //    complimentary method to optimize MergeDomScale function. (It is originally designed to simplify 
  //    BiasAdd node, but it may or may not work well for other similar situcations)
  // The details of following lines:
  // 1. Why don't we put it in the MergeDomScale function? Because we want to keep the MergeDomScale 
  //    interface consistent with ChooseDomScale and in the ChooseDomScale/MergeDomScale scope there is too
  //    few peripheral information.
  // 2. What does it do? When there are two inputs of add node and one is simulated quantize node one is not,
  //    we will use the scale of simulated quantize node as the scale of the other.
  // 3. Why do we do this? Without this, in the realization of BiasAdd node, the MergeDomScale function will 
  //    merge two multi-channel scale arrays into a new one. For BiasAdd parameters, it is okay to do this
  //    because fold constant pass will simplify the compute graph afterward. But for the input feature,
  //    it will add a bunch of MulAndDiv function to do the scale tranformation which is low-efficiency.
  //    Hence, in this case, this method use the input feature scale as the parameters' scale.
  // 4. Potential Risk: We use the judgements: (1) there are two inputs (2) one is simulated quantize node
  //    and one is not. This is too simple for a large scale of network models.
  Expr dom_scale;
  bool dom_scale_fold = false;
  if (ret.size() == 2) {
    auto ref_arg1 = ref_args[0].as<CallNode>();
    auto ref_arg2 = ref_args[1].as<CallNode>();
    if (ref_arg1 && ref_arg2) {
      if ((ref_arg1->op.same_as(simulated_quantize)) && !(ref_arg2->op.same_as(simulated_quantize))) {
        if (ref_arg1->attrs.as<SimulatedQuantizeAttrs>()->kind == kQWeight) {
          dom_scale = nptrs[1]->dom_scale;
          dom_scale_fold = true;
        }      
      } else if (!(ref_arg1->op.same_as(simulated_quantize)) && (ref_arg2->op.same_as(simulated_quantize))) {
        if (ref_arg2->attrs.as<SimulatedQuantizeAttrs>()->kind == kQWeight) {
          dom_scale = nptrs[0]->dom_scale;
          dom_scale_fold = true;
        }
      }
    }
  }

  // Expr dom_scale;
  // if (ref_args.size() == 2) { // Jasper: biasAdd ==> (kQActivation + kQWeight), the scale use kQActivation's scale.
  //   auto ref_arg1 = ref_args[0].as<CallNode>();
  //   auto ref_arg2 = ref_args[1].as<CallNode>();
  //   if (ref_arg1 && ref_arg2) {
  //     if (ref_arg1->attrs.as<SimulatedQuantizeAttrs>()->kind == kQActivation &&
  //        ref_arg2->attrs.as<SimulatedQuantizeAttrs>()->kind == kQWeight) {
  //          dom_scale = nptrs[0]->dom_scale;
  //     } else if (ref_arg1->attrs.as<SimulatedQuantizeAttrs>()->kind == kQWeight &&
  //        ref_arg2->attrs.as<SimulatedQuantizeAttrs>()->kind == kQActivation) {
  //          dom_scale = nptrs[1]->dom_scale;
  //     }
  //   }
  // } else {
  //   dom_scale = MergeDomScale(nptrs);
  // }

  if (!dom_scale_fold) 
    dom_scale = MergeDomScale(nptrs);
    
  for (size_t i = 0; i < ret.size(); ++i) {
    Expr cur_scale = nptrs[i]->dom_scale;
    // LOG(INFO) << "MulAndDiv s1: " << GetScalarFromConstant<float>(cur_scale);
    // LOG(INFO) << "MulAndDiv s2: " << GetScalarFromConstant<float>(dom_scale);
    ret.Set(i, MulAndDiv(ret[i], cur_scale, dom_scale, dtype, ref_args[i]->type_as<TensorTypeNode>()->shape));
  }

  *dtype_ptr = dtype;
  *scale_ptr = dom_scale;
  return ret;
}

Expr AddRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  ICHECK_EQ(new_args.size(), 2);
  if (new_args[0].as<QRealizeIntExprNode>() && new_args[1].as<QRealizeIntExprNode>()) {
    DataType dtype;
    Expr dom_scale;
    // execute the operation with activation data type.
    const QConfig& cfg = QConfig::Current();
    Array<Expr> ret_args =
        UnifyDTypeScale(ref_call->args, new_args, &dtype, &dom_scale, cfg->dtype_activation);
    for (size_t i = 0; i < ret_args.size(); ++i) {
      // do not fuse float32 arg
      if (new_args[i].as<QRealizeIntExprNode>()->dtype == DataType::Float(32)) {
        ret_args.Set(i, StopFusion(ret_args[i]));
      }
    }
    Expr ret = ForwardOp(ref_call, ret_args);
    return QRealizeIntExpr(ret, dom_scale, dtype);
  }

  // ICHECK(!new_args[0]->IsInstance<TempExprNode>() && !new_args[1]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("add").set_attr<FForwardRewrite>("FQRealizeRewrite", AddRealize);

Expr ClipRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  ICHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    const auto ref_attrs = ref_call->attrs.as<ClipAttrs>();
    auto attrs = make_object<ClipAttrs>();
    if (IsConstScalar(n->dom_scale)) {
      double dom_scale = GetScalarFromConstant<float>(n->dom_scale);
      attrs->a_min = ref_attrs->a_min / dom_scale;
      attrs->a_max = ref_attrs->a_max / dom_scale;
    } else {
      attrs->a_min = 0;
      attrs->a_max = INT32_MAX;
    }
    
    Expr ret = Call(ref_call->op, {n->data}, Attrs(attrs), ref_call->type_args);
    return QRealizeIntExpr(ret, n->dom_scale, n->dtype);
  }
  ICHECK(!new_args[0]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("clip").set_attr<FForwardRewrite>("FQRealizeRewrite", ClipRealize);

Expr ConcatenateRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  ICHECK_EQ(new_args.size(), 1);
  ICHECK_EQ(ref_call->args.size(), 1);

  const auto* tuple = new_args[0].as<TupleNode>();
  const auto* ref_tuple = ref_call->args[0].as<TupleNode>();
  ICHECK(tuple);
  ICHECK(ref_tuple);
  const Array<Expr>& arr = tuple->fields;
  const Array<Expr>& ref_arr = ref_tuple->fields;

  if (arr[0].as<QRealizeIntExprNode>()) {
    DataType dtype;
    Expr dom_scale;
    Array<Expr> ret_args = UnifyDTypeScale(ref_arr, arr, &dtype, &dom_scale);
    Expr ret = ForwardOp(ref_call, {Tuple(ret_args)});
    return QRealizeIntExpr(ret, dom_scale, dtype);
  } else {
    for (auto arg : new_args) {
      ICHECK(!arg->IsInstance<TempExprNode>());
    }
    return Expr(nullptr);
  }
}

RELAY_REGISTER_OP("concatenate").set_attr<FForwardRewrite>("FQRealizeRewrite", ConcatenateRealize);

/* \brief forward the original operator */
Expr IdentityRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  ICHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr ret = ForwardOp(ref_call, {n->data});
    return QRealizeIntExpr(ret, n->dom_scale, n->dtype);
  }
  ICHECK(!new_args[0]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("nn.relu").set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

RELAY_REGISTER_OP("reshape").set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

RELAY_REGISTER_OP("strided_slice").set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

RELAY_REGISTER_OP("nn.batch_flatten")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

RELAY_REGISTER_OP("transpose").set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

RELAY_REGISTER_OP("annotation.stop_fusion")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

/* \brief for unary operators which requantize its input to dtype_nbit */
Expr CastDtypeInputRealize(const Call& ref_call, const Array<Expr>& new_args,
                           const ObjectRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  ICHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr data = Cast(n->data, cfg->dtype_input);
    Expr ret = ForwardOp(ref_call, {data});
    return QRealizeIntExpr(ret, n->dom_scale, cfg->dtype_input);
  }
  ICHECK(!new_args[0]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("nn.max_pool2d")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", CastDtypeInputRealize);

RELAY_REGISTER_OP("nn.max_pool1d")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", CastDtypeInputRealize);

Expr AvgPoolRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  ICHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr data = n->data;
    if (n->dtype != cfg->dtype_activation) {
      data = Cast(n->data, cfg->dtype_activation);
    }
    Expr ret = ForwardOp(ref_call, {data});
    return QRealizeIntExpr(ret, n->dom_scale, cfg->dtype_activation);
  }
  ICHECK(!new_args[0]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("nn.avg_pool2d").set_attr<FForwardRewrite>("FQRealizeRewrite", AvgPoolRealize);

RELAY_REGISTER_OP("nn.global_avg_pool2d")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", AvgPoolRealize);

Expr CastHintRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  const auto param = ref_call->attrs.as<CastHintAttrs>();
  ICHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr ret = Cast(n->data, param->dtype);
    return QRealizeIntExpr(ret, n->dom_scale, param->dtype);
  }
  ICHECK(!new_args[0]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("annotation.cast_hint")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", CastHintRealize);

Expr BatchMatmulRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  ICHECK_EQ(new_args.size(), 2);
  if (!new_args[0]->IsInstance<TempExprNode>() || !new_args[1]->IsInstance<TempExprNode>()) {
    return Expr(nullptr);
  }
  const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
  const auto* rhs = new_args[1].as<QRealizeIntExprNode>();

  Expr ldata = lhs->data;
  Expr rdata = rhs->data;
  // DataType dtype = cfg->dtype_input;
  DataType dtype = cfg->dtype_activation;

  if (lhs->dtype != dtype) {
    ldata = Cast(ldata, dtype);
  }
  if (rhs->dtype != dtype) {
    rdata = Cast(rdata, dtype);
  }

  const auto ref_attrs = ref_call->attrs.as<BatchMatmulAttrs>();
  auto attrs = make_object<BatchMatmulAttrs>();
  *attrs = *ref_attrs;
  DataType out_dtype = cfg->dtype_activation;
  attrs->out_dtype = out_dtype;

  Expr ret = Call(ref_call->op, {ldata, rdata}, Attrs(attrs), ref_call->type_args);
  Expr mul = Multiply(lhs->dom_scale, rhs->dom_scale);
  Expr dom_scale = FoldConstantOpt(mul);
  return QRealizeIntExpr(ret, dom_scale, out_dtype);
}

RELAY_REGISTER_OP("nn.batch_matmul")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", BatchMatmulRealize);

Pass QuantizeRealizePass() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(ForwardRewrite(f, "FQRealizeRewrite", nullptr, nullptr));
      };
  return CreateFunctionPass(pass_func, 1, "QuantizeRealize", {});
}

TVM_REGISTER_GLOBAL("relay._quantize.QuantizeRealize").set_body_typed(QuantizeRealizePass);

}  // namespace quantize
}  // namespace relay
}  // namespace tvm
