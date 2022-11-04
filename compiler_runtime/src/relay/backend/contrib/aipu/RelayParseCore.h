
/*
 * Inspur.
 * This is a new or modified file.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_AIPU_RELAY_PARSE_CORE_H_
#define TVM_RELAY_BACKEND_CONTRIB_AIPU_RELAY_PARSE_CORE_H_

#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/module.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/relay/attrs/reduce.h>

#include <fstream>
#include <numeric>
#include <sstream>
#include <string>

#include "../../utils.h"
#include "codegen_aipu.h"
#include "RelayParser.h"
namespace tvm {
namespace relay {
namespace contrib {
namespace aipu{

using IntegerArray = Array<Integer>;
using namespace backend;
using ShapeVector = std::vector<int64_t>;

inline size_t GetShape1DSize(const Type& type) {
    const auto shape = GetShape(type);
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

class Dims4
{
  public:
    Dims4() : n(1), c(0), h(0), w(0) {};
    Dims4(int c, int h, int w) : n(1), c(c), h(h), w(w) {};
    Dims4(int n, int c, int h, int w) : n(n), c(c), h(h), w(w) {};
    int n;      //!< the number of images in the data or number of kernels in the weights (default = 1)
    int c;      //!< the number of channels in the data
    int h;      //!< the height of the data
    int w;      //!< the width of the data
    inline bool operator==(const Dims4& other) const
    {
        return (n == other.n && c == other.c && h == other.h && w == other.w);
    }
    inline bool operator!=(const Dims4& other) const
    {
        return !(n == other.n && c == other.c && h == other.h && w == other.w);
    }
};

static std::vector<std::string> Conv2d(const CallNode* call) {
  std::vector<std::string> args;
  const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
  ICHECK(conv2d_attr);

  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());
  //if(call->args[1].as<VarNode>())
  //{
    //LOG(INFO)<<"WEIGHT AS VARNODE";
  //}
  if(call->args[1].as<ConstantNode>())
  {
    //LOG(INFO)<<"Conv2d WEIGHT AS ConstantNode";
    Constant argconst = Downcast<Constant>(call->args[1]);
    //LOG(INFO)<<"Conv2d WEIGHT AS ConstantNode data:"<<*((float*)argconst->data->data);
  }
  // Args: N, C, H, W
  for (auto s : ishape) {
    // LOG(INFO) << s;
    args.push_back(std::to_string(s));
  }

  // Args: O, G, Ph, Pw, Kh, Kw, Sh, Sw
  args.push_back(std::to_string(wshape[0]));
  args.push_back(std::to_string(conv2d_attr->groups));
  args.push_back(std::to_string(conv2d_attr->padding[0].as<IntImmNode>()->value));
  args.push_back(std::to_string(conv2d_attr->padding[1].as<IntImmNode>()->value));
  args.push_back(std::to_string(wshape[2]));
  args.push_back(std::to_string(wshape[3]));
  args.push_back(std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value));
  args.push_back(std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value));

  //for (auto ar : args) {
  //  LOG(INFO) << ar;
  //}
  return args;
}

static std::vector<std::string> Dense(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());
  if(call->args[1].as<VarNode>())
  {
    //LOG(INFO)<<"WEIGHT AS VARNODE";
  }
  if(call->args[1].as<ConstantNode>())
  {
    //LOG(INFO)<<"WEIGHT AS ConstantNode";
    Constant argconst = Downcast<Constant>(call->args[1]);
    // LOG(INFO)<<"WEIGHT AS ConstantNode data:"<<*((float*)argconst->data->data);
  }
  // Args: N, C, O
  args.push_back(std::to_string(ishape[0]));
  args.push_back(std::to_string(ishape[1]));
  args.push_back(std::to_string(wshape[0]));

  return args;
}

static std::vector<std::string> Relu(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  return args;
}

static std::vector<std::string> LeakyRelu(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  return args;
}

static std::vector<std::string> aipuconcatenate(const CallNode* call) {return {};}

static std::vector<std::string> aipusoftmax(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  return args;
}

static std::vector<std::string> aipupool2d(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  return args;
}

static std::vector<std::string> aipupad(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  return args;
}

static std::vector<std::string> aipureshape(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  return args;
}

//squeeze
static std::vector<std::string> aipusqueeze(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  return args;
}

static std::vector<std::string> BatchNorm(const CallNode* call) {
  std::vector<std::string> args;
  const auto* bn_attr = call->attrs.as<BatchNormAttrs>();
  auto ishape = GetShape(call->args[0]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  // Args: epsilon
  args.push_back(std::to_string(bn_attr->epsilon));

  return args;
}


static std::vector<std::string> Add(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());
  if(call->args[1].as<VarNode>())
  {
    //LOG(INFO)<<"bias AS VARNODE";
  }
  if(call->args[1].as<ConstantNode>())
  {
    //LOG(INFO)<<"bias AS ConstantNode";
    Constant argconst = Downcast<Constant>(call->args[1]);
    // LOG(INFO)<<"bias AS ConstantNode data:"<<*((float*)argconst->data->data);
  }
  // Args: H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  return args;
}

class AIPURelay2NetworkCore : public MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {
 public:
  explicit AIPURelay2NetworkCore(const std::string& id,nvdla::INetwork* network,IBlobNameToTensor* blobname_to_tensor,std::vector<void*> *tmpallocs) {
    this->ext_func_id_ = id; 
    this->network_ = network;
    this->blobname_to_tensor_ = blobname_to_tensor;
    this->tmpallocs_ = tmpallocs;
    // FILE* fp= fopen("weight.bin","wb");
    // fclose(fp);
  }

  std::vector<Output> VisitExprDefault_(const Object* op) final {
    LOG(FATAL) << "AIPU codegen doesn't support: " << op->GetTypeKey();
    return {};
  }

  std::vector<Output> VisitExpr_(const VarNode* node) final {
    //LOG(INFO)<<"VarNode node name:"<<node->name_hint();
    ext_func_args_.push_back(GetRef<Var>(node));
    Output output;
    output.name = node->name_hint();
    return {output};
  }

  std::vector<Output> VisitExpr_(const TupleNode* node) final {
    std::vector<Output> outs;
    for (auto field : node->fields) {
      auto res = VisitExpr(field);
      ICHECK_EQ(res.size(), 1U) << "Do not support tuple nest";
      outs.push_back(res[0]);
    }
    return outs;
  }

  std::vector<Output> VisitExpr_(const TupleGetItemNode* op) final {
    auto res = VisitExpr(op->tuple);
    ICHECK_GT(res.size(), static_cast<size_t>(op->index));

    // Only keep the item we want for the child node.
    // FIXME(@comaniac): The other items should still be requried for the primary outputs.
    return {res[op->index]};
  }

  std::vector<Output> VisitExpr_(const ConstantNode* cn) final {

    Output output;
    //for (auto i : cn->data.Shape())
    //  LOG(INFO) << i;
    // Get const: static_cast<float*>(dnnl_0_consts[0]->data)
    output.name = CreateDataReference(ext_func_id_, const_idx_);
    output.dtype = "float";

    // Generate the global variable for needed ndarrays
    if (const_array_name_.empty()) {
      const_array_name_ = CreateNDArrayPool(ext_func_id_);
      std::string checker = CreateInitChecker(ext_func_id_);
      ext_func_body_.insert(ext_func_body_.begin(), checker);
    }

    // Give the ndarray a unique name to ease the initialization of it at
    // runtime.
    std::string const_var_name = CreateConstVar(ext_func_id_, const_idx_);
    const_vars_.push_back(const_var_name);
    const_idx_++;

    const auto* type_node = cn->checked_type().as<TensorTypeNode>();
    ICHECK(type_node);
    // ICHECK_EQ(GetDtypeString(type_node), "float") << "Only float is supported for now.";
    //LOG(INFO)<<"Const Node node name:"<<output.name;
    return {output};
  }

  std::vector<Output> VisitExpr_(const CallNode* call) final {
    GenerateBodyOutput ret;
    //LOG(INFO)<<"call node:";
    if (const auto* func = call->op.as<FunctionNode>()) {
      //LOG(INFO) << func;
      ret = GenerateCompositeFunctionCall(func, call);
    } else {
      const auto* op_node = call->op.as<OpNode>();
      const auto op_name = GetRef<Op>(op_node)->name;

      if (op_name == "relay.op.annotation.simulated_quantize" || op_name == "annotation.stop_fusion" || op_name == "annotation.cast_hint")
	return VisitExpr(call->args[0]);
      
      ret = GenerateOpCall(call);
    }
    buf_decl_.insert(buf_decl_.end(), ret.buffers.begin(), ret.buffers.end());
    ext_func_body_.push_back(ret.decl);
    return ret.outputs;
    //Output output;
    //output.name = "call";
    //return {output};
  }

  // based on JitImpl in tvm/src/relay/backend/contrib/codegen_c/codegen_c.h 
  std::string AIPUJitImpl(const std::string& ext_func_id, const Array<Var>& args,
                      const std::vector<std::string>& buf_decl,
                      const std::vector<std::string>& body, const std::string& const_arr_name,
                      const std::vector<Output>& outs) {
    // Create a declaration for global ndarrays that contain constant data.
    if (!const_arr_name.empty()) {
      code_stream_ << "#ifdef __cplusplus\n";
      code_stream_ << const_arr_name << "\n\n";
      code_stream_ << "#endif\n";
    }
    // Create the signature. For example, it could be:
    // void dnnl_0_(float* in0, float* in1, float* out0, float* out1) {}
    code_stream_ << "int " << ext_func_id << "(";


    // for (const auto& arg : args) {
    //   const auto& dtype_str = GetDtypeString(arg);
    //   code_stream_ << dtype_str << "* " << arg->name_hint() << ", ";
    // }
    // for (size_t i = 0; i < outs.size() - 1; ++i) {
    //   code_stream_ << outs[i].dtype << "* out" << i << ", ";
    // }
    // code_stream_ << outs.back().dtype << "* out" << outs.size() - 1 << ") {\n";
    code_stream_ << ") {\n";
    this->EnterScope();

    // Function body
    for (auto decl : buf_decl) {
      this->PrintIndents();
      code_stream_ << decl << "\n";
    }
    code_stream_ << "\n";
    for (auto stmt : body) {
      this->PrintIndents();
      code_stream_ << stmt << "\n";
    }

    // Copy output
    for (size_t i = 0; i < outs.size(); ++i) {
      if (!outs[i].need_copy) {
        continue;
      }
      this->PrintIndents();
      code_stream_ << "memcpy(out" << i << ", " << outs[i].name << ", 4 * " << outs[i].size
                   << ");\n";
    }

    // Free buffers
    for (size_t i = 0; i < buf_decl.size(); i++) {
      this->PrintIndents();
      code_stream_ << "free(buf_" << std::to_string(temporary_buffer_id_[i]) << ");\n";
    }

    this->PrintIndents();
    code_stream_ << "return 0;\n";

    this->ExitScope();
    code_stream_ << "}\n";

    // Create the wrapper to call the ext_func
    // this->GenerateBackendCFunc(ext_func_id, args, const_arr_name, outs);
    return code_stream_.str();
  }

  std::string JIT(const std::vector<Output>& out) {
    return AIPUJitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
  }

 private:
  struct GenerateBodyOutput {
    std::string decl;
    std::vector<std::string> buffers;
    std::vector<Output> outputs;
  };

  std::vector<std::string> GetArgumentNames(const CallNode* call) {
    std::vector<std::string> arg_names;
    //LOG(INFO)<<"GetArgumentNames(const CallNode* call):"<<call->args.size();
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto res = VisitExpr(call->args[i]);
      //LOG(INFO)<<"GetArgumentNames:"<<res.size();
      for (const auto& out : res) {
        arg_names.push_back(out.name);
        ///LOG(INFO)<<"GetArgumentNames:"<<out.name;
      }
    }
    return arg_names;
  }

  GenerateBodyOutput GenerateOpCall(const CallNode* call) {
    const auto* op_node = call->op.as<OpNode>();
    ICHECK(op_node) << "Expect OpNode, but got " << call->op->GetTypeKey();

    using ArgFunType = std::function<std::vector<std::string>(const CallNode*)>;
    static const std::map<std::string, std::pair<std::string, ArgFunType>> op_map = {
        {"nn.conv2d", {"aipu_conv2d", Conv2d}},
        {"nn.dense", {"aipu_dense", Dense}},
        {"nn.relu", {"aipu_relu", Relu}},
        {"clip", {"aipu_relu", Relu}},
        {"nn.batch_norm", {"aipu_bn", BatchNorm}},
        {"add", {"aipu_add", Add}},
        {"reshape",{"aipu_reshape",aipureshape}},
        {"nn.softmax",{"aipu_softmax",aipusoftmax}},
        {"nn.max_pool2d",{"aipu_max_pool2d",aipupool2d}},
        {"mean",{"aipu_global_pool2d",aipupool2d}},
        {"nn.avg_pool2d",{"aipu_global_pool2d",aipupool2d}},
        {"squeeze",{"aipu_squeeze",aipusqueeze}},
        {"nn.pad",{"aipu_pad",aipupad}},
        {"concatenate", {"aipu_concatenate", aipuconcatenate}},
        {"nn.leaky_relu", {"aipu_leakyRelu", LeakyRelu}},
    };

    const auto op_name = GetRef<Op>(op_node)->name;
    const auto iter = op_map.find(op_name);
    if (iter != op_map.end()) {
      return GenerateBody(call, iter->second.first, iter->second.second(call),call);
    }

    LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
    return {};
  }

  // 需要一个完整的列表来匹配融合函数，需要进一步完善
  GenerateBodyOutput GenerateCompositeFunctionCall(const FunctionNode* callee,
                                                   const CallNode* caller) {
    const auto pattern_name = callee->GetAttr<runtime::String>(attr::kComposite);
    //LOG(INFO)<<"pattern name:"<<pattern_name;
    ICHECK(pattern_name.defined()) << "Only functions with composite attribute supported";

    if (pattern_name == "aipu.conv2d_add") {
      const auto* conv_call =
          GetRootCall(callee->body.as<CallNode>(), 1, {"nn.conv2d", "add"});
      return GenerateBody(conv_call, "aipu_fused_conv2d_bias", GetArgumentNames(caller),
                          Conv2d(conv_call),caller);
      //GetArgumentNames传入的是caller 当weight和add为constant时args就只有一个。
    } else if (pattern_name == "aipu.dense_add") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, {"nn.dense", "add"});
      return GenerateBody(conv_call, "dnnl_fused_dense_bias", GetArgumentNames(caller),
                          Dense(conv_call),caller);
    } else if (pattern_name == "aipu.dense_add_extended") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, {"nn.dense", "add"});
      return GenerateBody(conv_call, "dnnl_fused_dense_bias", GetArgumentNames(caller),
                          Dense(conv_call),caller);
    }

    LOG(FATAL) << "Unknown composite function:" << pattern_name;
    return {};
  }

  GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                                  const std::vector<std::string>& attribute_args,
                                  const CallNode* caller) {
    return GenerateBody(root_call, func_name, GetArgumentNames(root_call), attribute_args/*useless*/,caller);
  }

  GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                                  const std::vector<std::string>& func_args,
                                  const std::vector<std::string>& attribute_args,
                                  const CallNode* caller) {
    std::string layername = "layer_" + std::to_string(layer_idx_++) + "_" + func_name ;
    //LOG(INFO)<<"conver "<<layername<<" begin";
    // Make function call with input buffers when visiting arguments
    ICHECK_GT(func_args.size(), 0);
    //LOG(INFO)<<layername<<" has :"<<func_args.size()<<" inputs";
    //for(auto arg_name : func_args){
    //  LOG(INFO)<<"arg name:"<<arg_name;
    //}
    // Analyze the output buffers
    std::vector<Type> out_types;
    if (root_call->checked_type()->IsInstance<TupleTypeNode>()) {
      auto type_node = root_call->checked_type().as<TupleTypeNode>();
      for (auto field : type_node->fields) {
        ICHECK(field->IsInstance<TensorTypeNode>());
        out_types.push_back(field);
      }
    } else if (root_call->checked_type()->IsInstance<TensorTypeNode>()) {
      ICHECK(root_call->checked_type()->IsInstance<TensorTypeNode>());
      out_types.push_back(root_call->checked_type());
    } else {
      LOG(FATAL) << "Unrecognized type node: " << AsText(root_call->checked_type(), false);
    }
    GenerateBodyOutput ret;
    for (const auto& out_type : out_types) {
      this->PrintIndents();
      const std::string out = "buf_" + std::to_string(buf_idx_++);
      const auto out_size = GetShape1DSize(out_type);
      Output output;
      output.name = out;
      output.size = out_size;
      output.dtype = GetDtypeString(out_type.as<TensorTypeNode>());
      output.need_copy = true;
      ret.buffers.push_back("float* " + out + " = (float*)std::malloc(4 * " +
                            std::to_string(out_size) + ");");
      ret.outputs.push_back(output);
    }
    //LOG(INFO)<<layername<<" has :"<<ret.outputs.size()<<" outputs";
    //for(auto output_content : ret.outputs){
      //LOG(INFO)<<"output name:"<<output_content.name;
    //}

    //parse 
    const auto* op_node = root_call->op.as<OpNode>();
    const auto op_name = GetRef<Op>(op_node)->name;
    nvdla::ILayer* layer = NULL;
    
    if(op_name == "nn.conv2d"){
      layer= ParseConvolution(root_call,caller,network_,blobname_to_tensor_,tmpallocs_,func_args);
      //LOG(INFO)<<"tmp call size:"<<tmpallocs_->size();
    }
    else if(op_name == "nn.relu" || op_name == "clip"){
      layer= ParseRelu(network_,blobname_to_tensor_,func_args);
      //LOG(INFO)<<"tmp call size:"<<tmpallocs_->size();
    }
    else if(op_name == "nn.max_pool2d"||op_name == "mean" ||op_name == "nn.avg_pool2d"){//get output?
      layer= parsePooling(root_call,network_,blobname_to_tensor_,func_args);
      //LOG(INFO)<<"tmp call size:"<<tmpallocs_->size();
    }
    else if(op_name == "reshape" || op_name == "squeeze"){ 
      nvdla::ITensor* tensor = (*blobname_to_tensor_)[func_args[0]];
      (*blobname_to_tensor_)[ret.outputs[0].name] = tensor;
      std::cout << "Warning: Flatten layer ignored." << std::endl;
      return ret;
    }
    else if(op_name == "nn.dense"){
      layer= ParseDense(root_call,caller,network_,blobname_to_tensor_,tmpallocs_,func_args);
      //LOG(INFO)<<"tmp call size:"<<tmpallocs_->size();
    }
    else if(op_name == "nn.softmax"){
      layer= parseSoftMax(network_,blobname_to_tensor_,func_args);
      //LOG(INFO)<<"tmp call size:"<<tmpallocs_->size();
    }
    else if(op_name == "add"){
      nvdla::ElementWiseOperation op = nvdla::ElementWiseOperation::kSUM;
      layer = network_->addElementWise((*blobname_to_tensor_)[func_args[0]], (*blobname_to_tensor_)[func_args[1]], op);
    }
    else if(op_name == "nn.leaky_relu"){
      const auto* leaky_relu_attr = root_call->attrs.as<LeakyReluAttrs>();
      double alpha = leaky_relu_attr->alpha;
      layer = network_->addActivation((*blobname_to_tensor_)[func_args[0]], /*ActivationType::*/nvdla::ActivationType::kPRELU, alpha);
    }
    else if(op_name == "concatenate"){
      layer= parseConcatenate(network_, blobname_to_tensor_, func_args);
      // LOG(INFO)<<"tmp call size:"<<tmpallocs_->size();
    }
    else{
      LOG(FATAL)<<"error unsupport layertype:"<<layername;
    }
    if (layer == 0)
    {
        LOG(FATAL) <<"error: parsing layer "<< layername;
    }
    else
    {
        layer->setName((const char*)layername.c_str());
        blobname_to_tensor_->add(ret.outputs[0].name, layer->getOutput(0));
        //LOG(INFO) << ret.outputs[0].name;
    }
    return ret;
  }
  static nvdla::ILayer* ParseConvolution(const CallNode* root_call,const CallNode* caller,
                                         nvdla::INetwork* network,IBlobNameToTensor* tensor,
                                         std::vector<void*>* tmp_weight_vector,
                                        const std::vector<std::string>& func_args){
    nvdla::ILayer* layer = NULL;
    nvdla::Weights kernelWeights;
    nvdla::Weights biasWeights;
    const auto* conv2d_attr = root_call->attrs.as<Conv2DAttrs>();
    ICHECK(conv2d_attr);
    auto ishape = GetShape(root_call->args[0]->checked_type());
    auto wshape = GetShape(root_call->args[1]->checked_type());
    //LOG(INFO)<<"input ishape:"<<ishape[0]<<
    //  ","<<ishape[1]<<
    //  ","<<ishape[2]<<
    //  ","<<ishape[3];
    if(root_call->args[1].as<VarNode>())
    {
      LOG(FATAL)<<"WEIGHT AS VARNODE";
    }
    if(root_call->args[1].as<ConstantNode>())
    {
      //LOG(INFO)<<"Conv2d WEIGHT AS ConstantNode";
      int weightsize = wshape[0]*wshape[1]*wshape[2]*wshape[3];
      //LOG(INFO)<<"wshape:"<<wshape[0]<<
      //            ","<<wshape[1]<<
      //            ","<<wshape[2]<<
      //            ","<<wshape[3]<<",kernel_layout:"<<conv2d_attr->kernel_layout;
      //LOG(INFO)<<"weightsize："<<weightsize;
      Constant argconst = Downcast<Constant>(root_call->args[1]);

      // std::ofstream origin_weight_file;
      // origin_weight_file.open ("origin_weight_file.txt" + func_args[0]);
      // for (int i = 0; i < weightsize; i++)
      //   origin_weight_file << ((float*)(argconst->data->data))[i] << std::endl;
      // origin_weight_file.close();

      float * weightdata = (float*)malloc(weightsize*sizeof(float));
      memcpy(weightdata,(float*)(argconst->data->data),weightsize*sizeof(float));

      // std::ofstream first_weightdata_file;
      // first_weightdata_file.open("first_weightdata_file.txt" + func_args[0]);
      // for (int i = 0; i < weightsize; i++)
      //   first_weightdata_file << weightdata[i] << std::endl;
      // first_weightdata_file.close();

      hwcn2nchw(weightdata,wshape,conv2d_attr->kernel_layout, false);

      // std::ofstream second_weightdata_file;
      // second_weightdata_file.open("second_weightdata_file.txt" + func_args[0]);
      // for (int i = 0; i < weightsize; i++)
      //   second_weightdata_file << weightdata[i] << std::endl;
      // second_weightdata_file.close();

      // FILE* fp= fopen("weight.bin","a+");
      // fwrite(weightdata,1,weightsize*sizeof(float),fp);
      // fclose(fp);

      kernelWeights = nvdla::Weights(nvdla::DataType::FLOAT, weightdata, weightsize);
      tmp_weight_vector->push_back((void*)weightdata);
      // LOG(INFO)<<"Conv2d WEIGHT AS ConstantNode data: "<<*((float*)argconst->data->data)
      // <<" weight size:"<<weightsize;
    }
    //const dc::ConvolutionParameter& p = msg.convolution_param();
    //LOG(INFO)<<"numGroups："<<conv2d_attr->groups;
    //LOG(INFO)<<"strides:"<<conv2d_attr->strides;
    //LOG(INFO)<<"padding："<<conv2d_attr->padding;
    //LOG(INFO)<<"dilation："<<conv2d_attr->dilation;


    int numOutputs = 0;
    if(conv2d_attr->kernel_layout=="HWOI"){
      numOutputs = wshape[2];//w shape h,w,in,out
    }
    if(conv2d_attr->kernel_layout=="HWIO"){
      numOutputs = wshape[3];//w shape h,w,in,out
    }

    int numGroups  = conv2d_attr->groups;
  
    int kernelW = wshape[0];
    int kernelH = wshape[1];

    int strideW = conv2d_attr->strides[0].as<IntImmNode>()->value;
    int strideH = conv2d_attr->strides[1].as<IntImmNode>()->value;

    int padTop = conv2d_attr->padding[0].as<IntImmNode>()->value;
    int padLeft = conv2d_attr->padding[1].as<IntImmNode>()->value;
    int padBottom = conv2d_attr->padding[2].as<IntImmNode>()->value;
    int padRight = conv2d_attr->padding[3].as<IntImmNode>()->value;

    int dilationW = conv2d_attr->dilation[0].as<IntImmNode>()->value;
    int dilationH = conv2d_attr->dilation[1].as<IntImmNode>()->value;

    //panduan shifouyou nn.pad
    if(auto *padcallnode = root_call->args[0].as<CallNode>()) {
      const auto* op_node = padcallnode->op.as<OpNode>();
      const auto op_name = GetRef<Op>(op_node)->name;
      if(op_name == "nn.pad"){
        //LOG(INFO)<<"conv has nn.pad";
        const auto* pad_attr = padcallnode->attrs.as<PadAttrs>();
        padTop = pad_attr->pad_width[1][0];//need know pad hw or wh 
        padLeft = pad_attr->pad_width[2][0];
        padBottom = pad_attr->pad_width[1][1];
        padRight = pad_attr->pad_width[2][1];
        //LOG(INFO)<<"conv has nn.pad: h w "<<padLeft<<","<<padRight;
      }
      else if (op_name == "relay.op.annotation.simulated_quantize") {
        if (const auto *real_pad = padcallnode->args[0].as<CallNode>()) {
        const auto* opnode = real_pad->op.as<OpNode>();
        const auto opname = GetRef<Op>(opnode)->name;
        if(opname == "nn.pad"){
          //LOG(INFO)<<"conv has nn.pad";
          const auto* pad_attr = real_pad->attrs.as<PadAttrs>();
          padTop = pad_attr->pad_width[1][0];
          padLeft = pad_attr->pad_width[2][0];
          padBottom = pad_attr->pad_width[1][1];
          padRight = pad_attr->pad_width[2][1];
          //LOG(INFO)<<"conv has nn.pad: h w "<<padLeft<<","<<padRight;
        }
        }
      }
    }
    //panduan shifouyou add
    if(const auto* func = caller->op.as<FunctionNode>()){
      auto biascallnode = func->body.as<CallNode>();
      const auto* op_node = biascallnode->op.as<OpNode>();
      const auto op_name = GetRef<Op>(op_node)->name;
      if(op_name == "add"){//GetRef<Op>(op_node)->name
        if(biascallnode->args[1].as<VarNode>()){
          //LOG(FATAL)<<"bias Must ConstantNode";
        }
        if(biascallnode->args[1].as<ConstantNode>()){
          auto bshape = GetShape(biascallnode->args[1]->checked_type());
          //LOG(INFO)<<"bias AS ConstantNode";
	  int biassize = 1;
	  for (size_t i = 0; i < bshape.size(); i++)
	    biassize = biassize * bshape[i];
          //LOG(INFO) << bshape[0] << " " << bshape[1] << " " << bshape[2] << " " << bshape[3];
          Constant argconst = Downcast<Constant>(biascallnode->args[1]);
          float * biasdata = (float*)malloc(biassize*sizeof(float));
          memcpy(biasdata,(float*)argconst->data->data,biassize*sizeof(float));
          // FILE* fp= fopen("weight.bin","a+");
          // fwrite(biasdata,1,biassize*sizeof(float),fp);
          // fclose(fp);
          biasWeights = nvdla::Weights(nvdla::DataType::FLOAT, biasdata, biassize);
          tmp_weight_vector->push_back((void*)biasdata);
          // LOG(INFO)<<"bias AS ConstantNode data:"<<*((float*)argconst->data->data)
          // <<",bias size:"<<biassize;
        }
      }
      else{
        biasWeights =nvdla::Weights(nvdla::DataType::FLOAT, NULL, 0);
      }
    }

    nvdla::BiasMode biasMode = nvdla::BiasMode::bNONE;

    // TODO: cross-correlation vs convolution

    //LOG(INFO) << "biasWeights.count " << biasWeights.count;

    if ( biasWeights.count == 0 )
    {
        biasMode = nvdla::BiasMode::bNONE;
    }
    else if ( biasWeights.count == 1 )
    {
        biasMode = nvdla::BiasMode::bUNIFORM;
    }
    else if ( biasWeights.count == numOutputs )
    {
      //LOG(INFO) << "nvdla::BiasMode::bCHANNEL";
        biasMode = nvdla::BiasMode::bCHANNEL;
    }
    else
    {
        biasMode = nvdla::BiasMode::bm_ELEMENTWISE;
    }

    // LOG(INFO) << "padTop " << padTop << " padLeft " << padLeft << " padBottom " << padBottom << " padRight " << padRight;
    // LOG(INFO) << "strideH " << strideH << " strideW " << strideW;
    // LOG(INFO) << "dilationH " << dilationH << " dilationW " << dilationH;
    // LOG(INFO) << "kernelH " << kernelH << " kernelW " << kernelW;
    // LOG(INFO) << "Groups " << numGroups;
    nvdla::Dims2 tlPadding = nvdla::Dims2(padTop, padLeft);
    nvdla::Dims2 brPadding = nvdla::Dims2(padBottom, padRight);
    nvdla::Dims2 stride    = nvdla::Dims2(strideH, strideW);
    nvdla::Dims2 dilation  = nvdla::Dims2(dilationH, dilationW);
    nvdla::Dims2 kernelSize= nvdla::Dims2(kernelH, kernelW);

    // TODO: cross-correlation vs convolution
    layer = network->addConvolution((*tensor)[func_args[0]], numOutputs, 0,
                                    kernelSize, tlPadding, brPadding, stride, dilation,
                                    kernelWeights, biasWeights, biasMode, numGroups);
    //LOG(INFO)<<"layer conver end output:"<<func_args[0];
    return layer;
  }

  static nvdla::ILayer* ParseDense(const CallNode* root_call, const CallNode* caller,
				   nvdla::INetwork* network, IBlobNameToTensor* tensor,
				   std::vector<void*>* tmp_weight_vector,
				   const std::vector<std::string>& func_args) {
    nvdla::Weights kernelWeights;
    nvdla::Weights biasWeights;
    // true: root_call->args[1] is ConstantNode; false: root_call->args[1] is VarNode
    bool weight_flag = true;
    auto wshape = GetShape(root_call->args[1]->checked_type());
    //LOG(INFO)<<"input ishape:"<<ishape[0]<<
    //  ","<<ishape[1]<<
    //  ","<<ishape[2]<<
    //  ","<<ishape[3];
    nvdla::Dims4 dims = (*tensor)[func_args[0]]->getDimensions();
    wshape.push_back(dims.h);
    wshape.push_back(dims.w);
    wshape.push_back(dims.c);
    int weightsize = wshape[0]*wshape[1];
    if(root_call->args[1].as<VarNode>())
    {
      weight_flag = false;
      LOG(WARNING)<<"WEIGHT AS VARNODE";
      float * weightdata = (float*)malloc(weightsize*sizeof(float));
      for(int _i =0; _i < weightsize; ++_i){
        weightdata[_i] = 0.;
      }

      kernelWeights = nvdla::Weights(nvdla::DataType::FLOAT, weightdata, weightsize);
      tmp_weight_vector->push_back((void*)weightdata);

    }
    if(root_call->args[1].as<ConstantNode>())
    {
      weight_flag = true;
      //LOG(INFO)<<"Conv2d WEIGHT AS ConstantNode";
      // LOG(INFO)<<"wshape:"<<wshape[0]<<
      // 	","<<wshape[1]<<
      // 	","<<wshape[2]<<
      // 	","<<wshape[3];
      //LOG(INFO)<<"weightsize："<<weightsize;
      Constant argconst = Downcast<Constant>(root_call->args[1]);

      // std::ofstream origin_weight_file;
      // origin_weight_file.open ("origin_weight_file.txt" + func_args[0]);
      // for (int i = 0; i < weightsize; i++)
      //   origin_weight_file << ((float*)(argconst->data->data))[i] << std::endl;
      // origin_weight_file.close();

      float * weightdata = (float*)malloc(weightsize*sizeof(float));
      memcpy(weightdata,(float*)(argconst->data->data),weightsize*sizeof(float));

      // std::ofstream first_weightdata_file;
      // first_weightdata_file.open("first_weightdata_file.txt" + func_args[0]);
      // for (int i = 0; i < weightsize; i++)
      //   first_weightdata_file << weightdata[i] << std::endl;
      // first_weightdata_file.close();

      hwcn2nchw(weightdata, wshape, "HWIO", true);

      // std::ofstream second_weightdata_file;
      // second_weightdata_file.open("second_weightdata_file.txt" + func_args[0]);
      // for (int i = 0; i < weightsize; i++)
      //   second_weightdata_file << weightdata[i] << std::endl;
      // second_weightdata_file.close();

      // FILE* fp= fopen("weight.bin","a+");
      // fwrite(weightdata,1,weightsize*sizeof(float),fp);
      // fclose(fp);

      kernelWeights = nvdla::Weights(nvdla::DataType::FLOAT, weightdata, weightsize);
      tmp_weight_vector->push_back((void*)weightdata);
      // LOG(INFO)<<"Conv2d WEIGHT AS ConstantNode data: "<<*((float*)argconst->data->data)
      // <<" weight size:"<<weightsize;
    }
    //const dc::ConvolutionParameter& p = msg.convolution_param();
    //LOG(INFO)<<"numGroups："<<conv2d_attr->groups;
    //LOG(INFO)<<"strides:"<<conv2d_attr->strides;
    //LOG(INFO)<<"padding："<<conv2d_attr->padding;
    //LOG(INFO)<<"dilation："<<conv2d_attr->dilation;



    int numOutputs = wshape[0];


    int numGroups  = 1;
    //LOG(INFO) << numOutputs;

    int kernelW = 1;
    int kernelH = 1;
    
    if (dims.c != wshape[1]) {
      kernelW = dims.w;
      kernelH = dims.h;
    }
    
    // if (wshape[1] == 3136) {
    //   kernelW = 7;
    //   kernelH = 7;
    // }
      


    int strideW = 1;
    int strideH = 1;

    int padTop = 0;
    int padLeft = 0;
    int padBottom = 0;
    int padRight = 0;

    int dilationW = 1;
    int dilationH = 1;

    //panduan shifouyou nn.pad
    if(auto *padcallnode = root_call->args[0].as<CallNode>()) {
      const auto* op_node = padcallnode->op.as<OpNode>();
      const auto op_name = GetRef<Op>(op_node)->name;
      if(op_name == "nn.pad"){
        //LOG(INFO)<<"conv has nn.pad";
        const auto* pad_attr = padcallnode->attrs.as<PadAttrs>();
        padTop = pad_attr->pad_width[1][0];//need know pad hw or wh 
        padLeft = pad_attr->pad_width[2][0];
        padBottom = pad_attr->pad_width[1][1];
        padRight = pad_attr->pad_width[2][1];
        //LOG(INFO)<<"conv has nn.pad: h w "<<padLeft<<","<<padRight;
      }
      else if (op_name == "relay.op.annotation.simulated_quantize") {
        if (const auto *real_pad = padcallnode->args[0].as<CallNode>()) {
        const auto* opnode = real_pad->op.as<OpNode>();
        const auto opname = GetRef<Op>(opnode)->name;
        if(opname == "nn.pad"){
          //LOG(INFO)<<"conv has nn.pad";
          const auto* pad_attr = real_pad->attrs.as<PadAttrs>();
          padTop = pad_attr->pad_width[1][0];
          padLeft = pad_attr->pad_width[2][0];
          padBottom = pad_attr->pad_width[1][1];
          padRight = pad_attr->pad_width[2][1];
          //LOG(INFO)<<"conv has nn.pad: h w "<<padLeft<<","<<padRight;
        }
        }
      }
    }
    //panduan shifouyou add
    if(const auto* func = caller->op.as<FunctionNode>()){
      auto biascallnode = func->body.as<CallNode>();
      const auto* op_node = biascallnode->op.as<OpNode>();
      const auto op_name = GetRef<Op>(op_node)->name;
      if(op_name == "add"){//GetRef<Op>(op_node)->name
        if(biascallnode->args[1].as<VarNode>()){
          //LOG(FATAL)<<"bias Must ConstantNode";
        }
        if(biascallnode->args[1].as<ConstantNode>()){
          auto bshape = GetShape(biascallnode->args[1]->checked_type());
          //LOG(INFO)<<"bias AS ConstantNode";
          int biassize = bshape[0];
          //LOG(INFO) << bshape[0] << " " << bshape[1] << " " << bshape[2] << " " << bshape[3];
          Constant argconst = Downcast<Constant>(biascallnode->args[1]);
          float * biasdata = (float*)malloc(biassize*sizeof(float));
          memcpy(biasdata,(float*)argconst->data->data,biassize*sizeof(float));
          // FILE* fp= fopen("weight.bin","a+");
          // fwrite(biasdata,1,biassize*sizeof(float),fp);
          // fclose(fp);
          biasWeights = nvdla::Weights(nvdla::DataType::FLOAT, biasdata, biassize);
          tmp_weight_vector->push_back((void*)biasdata);
          // LOG(INFO)<<"bias AS ConstantNode data:"<<*((float*)argconst->data->data)
          // <<",bias size:"<<biassize;
        }
      }
      else{
        biasWeights =nvdla::Weights(nvdla::DataType::FLOAT, NULL, 0);
      }
    }

    nvdla::BiasMode biasMode = nvdla::BiasMode::bNONE;

    // TODO: cross-correlation vs convolution

    //LOG(INFO) << "biasWeights.count " << biasWeights.count;

    if ( biasWeights.count == 0 )
    {
        biasMode = nvdla::BiasMode::bNONE;
    }
    else if ( biasWeights.count == 1 )
    {
        biasMode = nvdla::BiasMode::bUNIFORM;
    }
    else if ( biasWeights.count == numOutputs )
    {
      //LOG(INFO) << "nvdla::BiasMode::bCHANNEL";
        biasMode = nvdla::BiasMode::bCHANNEL;
    }
    else
    {
        biasMode = nvdla::BiasMode::bm_ELEMENTWISE;
    }

    //LOG(INFO) << "padTop " << padTop << " padLeft " << padLeft << " padBottom " << padBottom << " padRight " << padRight;
    //LOG(INFO) << "strideH " << strideH << " strideW " << strideW;
    //LOG(INFO) << "dilationH " << dilationH << " dilationW " << dilationH;
    //LOG(INFO) << "kernelH " << kernelH << " kernelW " << kernelW;
    nvdla::Dims2 tlPadding = nvdla::Dims2(padTop, padLeft);
    nvdla::Dims2 brPadding = nvdla::Dims2(padBottom, padRight);
    nvdla::Dims2 stride    = nvdla::Dims2(strideH, strideW);
    nvdla::Dims2 dilation  = nvdla::Dims2(dilationH, dilationW);
    nvdla::Dims2 kernelSize= nvdla::Dims2(kernelH, kernelW);

    // TODO: cross-correlation vs convolution
    if(weight_flag){
        return network->addConvolution((*tensor)[func_args[0]], numOutputs, 0,
                       kernelSize, tlPadding, brPadding, stride, dilation,
                       kernelWeights, biasWeights, biasMode, numGroups);
    }else{
        // the dimension of weights should be dealt separately
        auto theShape = GetShape(root_call->args[1]->checked_type());
        nvdla::Dims4 theDims;
        theDims.n = theShape[0];
        theDims.c = theShape[1];
        theDims.h = 1;
        theDims.w = 1;
        ((*tensor)[func_args[1]])->setDimensions(theDims);
        return network->addConvolution((*tensor)[func_args[0]], numOutputs, 0,
                       kernelSize, tlPadding, brPadding, stride, dilation,
                       (*tensor)[func_args[1]], biasWeights, biasMode, numGroups, weightsize);
    }
  }

  /*
    static nvdla::ILayer* ParseDense(const CallNode* root_call,const CallNode* caller,nvdla::INetwork* network,
    IBlobNameToTensor* tensor,std::vector<void*>* tmp_weight_vector,
    const std::vector<std::string>& func_args){
    nvdla::Weights kernelWeights;
    nvdla::Weights biasWeights;
    const auto* dense_attr = root_call->attrs.as<DenseAttrs>();
    ICHECK(dense_attr);
    auto wshape = GetShape(root_call->args[1]->checked_type());
    if(root_call->args[1].as<VarNode>())
    {
      LOG(FATAL)<<"WEIGHT AS VARNODE";
    }
    //bool falg = 0;
    if(root_call->args[1].as<ConstantNode>())
    {
      //LOG(INFO)<<"Conv2d WEIGHT AS ConstantNode";
      int weightsize = wshape[0]*wshape[1];
      //LOG(INFO)<<"wshape:"<<wshape[0]
      //             <<","<<wshape[1];
      Constant argconst = Downcast<Constant>(root_call->args[1]);
      float * weightdata = (float*)malloc(weightsize*sizeof(float));
      memcpy(weightdata,(float*)argconst->data->data,weightsize*sizeof(float));
      //nc2cn(weightdata,wshape);
      // FILE* fp= fopen("weight.bin","a+");
      // fwrite(weightdata,1,weightsize*sizeof(float),fp);
      // fclose(fp);
      // 需要获取FC的输入Shape，如果Shape不是1x1（H x W）的，则需要对权重做转换
      //if(wshape[1]==3136){
      //  nc2cn(weightdata,wshape);
      //}
      // 384 768 
      //LOG(INFO)<< "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
      kernelWeights = nvdla::Weights(nvdla::DataType::FLOAT, weightdata, weightsize);
      tmp_weight_vector->push_back((void*)weightdata);
      // LOG(INFO)<<"Conv2d WEIGHT AS ConstantNode data: "<<*((float*)argconst->data->data)
      // <<" weight size:"<<weightsize;
    }
    //const dc::ConvolutionParameter& p = msg.convolution_param();
    int numOutputs = wshape[0];//w shape h,w,in,out
    //LOG(INFO)<<"numGroups："<<numOutputs;

    if(const auto* func = caller->op.as<FunctionNode>()){
      auto biascallnode = func->body.as<CallNode>();
      //if(GetRef<Op>(biascallnode->op.as<OpNode>)->name == "add"){//GetRef<Op>(op_node)->name
        if(biascallnode->args[1].as<VarNode>()){
          LOG(FATAL)<<"bias Must ConstantNode";
        }
        if(biascallnode->args[1].as<ConstantNode>()){
          auto bshape = GetShape(biascallnode->args[1]->checked_type());
          //LOG(INFO)<<"bias AS ConstantNode shape:"<<bshape[0]
          //         <<","<<bshape[1]<<","<<bshape[2]<<","<<bshape[3];
          int biassize = bshape[0];
          Constant argconst = Downcast<Constant>(biascallnode->args[1]);
          float * biasdata = (float*)malloc(biassize*sizeof(float));
          memcpy(biasdata,(float*)argconst->data->data,biassize*sizeof(float));
          // FILE* fp= fopen("weight.bin","a+");
          // fwrite(biasdata,1,biassize*sizeof(float),fp);
          // fclose(fp);
          biasWeights = nvdla::Weights(nvdla::DataType::FLOAT, biasdata, biassize);
          tmp_weight_vector->push_back((void*)biasdata);
          // LOG(INFO)<<"bias AS ConstantNode data:"<<*((float*)argconst->data->data)
          // <<",bias size:"<<biassize;
        }
      //}
    }
    else{
      biasWeights =nvdla::Weights(nvdla::DataType::FLOAT, NULL, 0);
    }


    nvdla::BiasMode biasMode = nvdla::BiasMode::bNONE;

    // TODO: cross-correlation vs convolution


    if ( biasWeights.count == 0 )
    {
        biasMode = nvdla::BiasMode::bNONE;
    }
    else if ( biasWeights.count == 1 )
    {
        biasMode = nvdla::BiasMode::bUNIFORM;
    }
    else if ( biasWeights.count == numOutputs )
    {
        biasMode = nvdla::BiasMode::bCHANNEL;
    }
    else
    {
        biasMode = nvdla::BiasMode::bm_ELEMENTWISE;
    }
    // TODO: cross-correlation vs convolution
    gLogInfo<<"layer conver end output:"<<func_args[0];
    return network->addFullyConnected((*tensor)[func_args[0]], numOutputs,
                                      kernelWeights, biasWeights, biasMode);
  }
  */

  static nvdla::ILayer* ParseRelu(nvdla::INetwork* network,IBlobNameToTensor* tensor,
                                        const std::vector<std::string>& func_args){
    //LOG(INFO) << func_args[0];
    return network->addActivation((*tensor)[func_args[0]], /*ActivationType::*/nvdla::ActivationType::kRELU);
  }

  static nvdla::ILayer* parsePooling(const CallNode* call,nvdla::INetwork* network,IBlobNameToTensor* tensor,
                                        const std::vector<std::string>& func_args)
  {
    const auto* op_node = call->op.as<OpNode>();
    const auto op_name = GetRef<Op>(op_node)->name;
    auto ishape = GetShape(call->args[0]->checked_type());
    //LOG(INFO)<<"pool ishape:"<<ishape[0]<<
    //  ","<<ishape[1]<<
    //  ","<<ishape[2]<<
    //  ","<<ishape[3];
    int kernelH, kernelW;
    int strideH, strideW;
    int padH,padW;
    nvdla::PoolingType type; 
    if(op_name == "mean") {
      // const auto* reduce_attr = call->attrs.as<ReduceAttrs>();
      // ICHECK(reduce_attr);

      // only support specific mean that can be converted to average pool2d
      const auto* reduceAttrs = call->attrs.as<ReduceAttrs>();
      int axis1, axis2;
      nvdla::Dims4 dims = (*tensor)[func_args[0]]->getDimensions();
      //LOG(INFO)<<"gloabal pool kernel:"<<dims.h<<","<<dims.w << "," << dims.n << "," << dims.c;
      if (reduceAttrs->axis.size() == 2) {
	axis1 = reduceAttrs->axis[0];
	axis2 = reduceAttrs->axis[1];
	if ((axis1 == 1 && axis2 == 2) || (axis1 == 2 && axis2 == 1)) {
	  kernelH = dims.h;
	  kernelW = dims.w;
	}
	else {
	  LOG(FATAL) << "not supported mean";
	}
      }
      else if (reduceAttrs->axis.size() == 1) {
	axis1 = reduceAttrs->axis[0];
	if (axis1 == 1) {
	  kernelH = dims.h;
	  kernelW = 1;
	}
	else if (axis1 == 2) {
	  kernelH = 1;
	  kernelW = dims.w;
	}
	else {
	  LOG(FATAL) << "not supported mean";
	}
      }
      else {
	LOG(FATAL) << "not supported mean";
      }
      strideH = 1;
      strideW = 1;
      padH = 0;
      padW = 0;
      type = nvdla::PoolingType::kAVERAGE;
    }
    else if (op_name == "nn.avg_pool2d") {
      type = nvdla::PoolingType::kAVERAGE;

      const auto* pool2d_attr = call->attrs.as<AvgPool2DAttrs>();
      ICHECK(pool2d_attr);
      //LOG(INFO)<<"pool kernel:"<<pool2d_attr->pool_size;
      //LOG(INFO)<<"pool strides:"<<pool2d_attr->strides;
      //LOG(INFO)<<"pool padding:"<<pool2d_attr->padding;
      // mandatory

      kernelH = pool2d_attr->pool_size[0].as<IntImmNode>()->value;
      kernelW = pool2d_attr->pool_size[1].as<IntImmNode>()->value;

      strideH = pool2d_attr->strides[0].as<IntImmNode>()->value;
      strideW = pool2d_attr->strides[1].as<IntImmNode>()->value;

      padH = pool2d_attr->padding[0].as<IntImmNode>()->value;//0
      padW = pool2d_attr->padding[2].as<IntImmNode>()->value;//1

      //LOG(INFO) << "kernelH " << kernelH << " kernelW " << kernelW;
      //LOG(INFO) << "strideH " << strideH << " strideW " << strideW;
      //LOG(INFO) << "padH " << padH << " padW " << padW;
    }
    else if(op_name == "nn.max_pool2d"){
      const auto* pool2d_attr = call->attrs.as<MaxPool2DAttrs>();
      ICHECK(pool2d_attr);
      //LOG(INFO)<<"pool kernel:"<<pool2d_attr->pool_size;
      //LOG(INFO)<<"pool strides:"<<pool2d_attr->strides;
      //LOG(INFO)<<"pool padding:"<<pool2d_attr->padding;
      // mandatory

      kernelH = pool2d_attr->pool_size[0].as<IntImmNode>()->value;
      kernelW = pool2d_attr->pool_size[1].as<IntImmNode>()->value;

      strideH = pool2d_attr->strides[0].as<IntImmNode>()->value;
      strideW = pool2d_attr->strides[1].as<IntImmNode>()->value;

      padH = pool2d_attr->padding[0].as<IntImmNode>()->value;//0
      padW = pool2d_attr->padding[2].as<IntImmNode>()->value;//1
      type = nvdla::PoolingType::kMAX;
      //LOG(INFO) << "kernelH " << kernelH << " kernelW " << kernelW;
      //LOG(INFO) << "strideH " << strideH << " strideW " << strideW;
      //LOG(INFO) << "padH " << padH << " padW " << padW;
    }
    else {
      LOG(FATAL)<<"unsupport pool type";
    }
    nvdla::Dims2 windowSize = nvdla::Dims2(kernelH, kernelW);
    nvdla::Dims2 stride     = nvdla::Dims2(strideH, strideW);
    nvdla::Dims2 tlPadding  = nvdla::Dims2(padH, padH);
    nvdla::Dims2 brPadding  = nvdla::Dims2(padW, padW);//pad:0,0,1,1
    nvdla::ILayer *layer = network->addPooling((*tensor)[func_args[0]], type,
                                        windowSize, stride, tlPadding, brPadding);
    //LOG(INFO) << func_args[0];
    return layer;
  }
  static nvdla::ILayer* parseSoftMax(nvdla::INetwork* network,IBlobNameToTensor* tensor,
                                        const std::vector<std::string>& func_args)
  {
    return network->addSoftMax((*tensor)[func_args[0]]);
  }
  static nvdla::ILayer* parseConcatenate(nvdla::INetwork* network,IBlobNameToTensor* tensor,
                                         const std::vector<std::string>& func_args)
  {
    std::vector<nvdla::ITensor*> ptrs;
    for (unsigned int i = 0, n = func_args.size(); i < n; i++) {
      ptrs.push_back((*tensor)[func_args[i]]);
    }

    return network->addConcatenation(&ptrs[0], func_args.size());
  }
  static void hwcn2nchw(float *weightdata,std::vector<int>wshape,tvm::String kernel_layout, bool for_dense)
  {
    int shapesize;
    int N, C, H, W;
    if (for_dense) {
      H = 1;
      W = 1;
      C = wshape[1];
      // when input channel is not the same as weight channel, we use input channel
      if (wshape[1] != wshape[4]) {
	H = wshape[2];
	W = wshape[3];
	C = wshape[4];
      }
      shapesize = wshape[0]*wshape[1];
      N = wshape[0];
    }
    else {
      //LOG(INFO)<<"convert kernel_layout begin";
      shapesize = wshape[0]*wshape[1]*wshape[2]*wshape[3];
      if (kernel_layout == "HWIO") {
	N = wshape[3];
	C = wshape[2];
	H = wshape[0];
	W = wshape[1];
      }
      else if (kernel_layout == "HWOI") {
	N = wshape[2];
	C = wshape[3];
	H = wshape[0];
	W = wshape[1];
      }
      else {
	LOG(FATAL) << "kernel layout not supported " << kernel_layout;
      }
    }
    float *tmp_weightdata = (float*)malloc(shapesize*sizeof(float));
    memcpy(tmp_weightdata,weightdata,shapesize*sizeof(float));
    
    if (for_dense) {
      for (int i = 0; i < wshape[1]; i++) {
	for (int j = 0; j < N; j++) {
	  tmp_weightdata[i * N + j] = weightdata[j * wshape[1] + i];
	}
      }
    }
    // HWIO or HWOI -> OIHW
    for(int i=0;i<N/*N*/;i++){
      for(int j=0;j<C/*c*/;j++){
        for(int k=0;k<H/*H*/;k++){
          for(int l=0;l<W/*W*/;l++){
            weightdata[i*C*H*W+j*H*W+k*W+l]=tmp_weightdata[k*W*C*N+l*C*N+j*N+i];
          }
        }
      }
    }
    free(tmp_weightdata);
  }

  /*! \brief The id of the external dnnl ext_func. */
  std::string ext_func_id_{""};
  /*!
   * \brief The index to track the output buffer. Each kernel will redirect the
   * output to a buffer that may be consumed by other kernels.
   */
  int buf_idx_{0};
  /*! \brief The index of layer. */
  int layer_idx_{0};
  /*! \brief The index of global constants. */
  int const_idx_{0};
  /*! \brief The arguments used by a wrapped function that calls DNNL kernels. */
  Array<Var> ext_func_args_;
  /*! \brief Statement of the function that will be compiled using DNNL kernels. */
  std::vector<std::string> ext_func_body_;
  /*! \brief The array declared to store the constant values. */
  std::string const_array_name_;
  /*! \brief The declaration of intermeidate buffers. */
  std::vector<std::string> buf_decl_;
  /*! \brief The variable name to constant mapping. */
  Array<String> const_vars_;

  nvdla::INetwork *network_;

  IBlobNameToTensor* blobname_to_tensor_;

  std::vector<void *>* tmpallocs_;
  protected://guosy add
  /*! \brief plan memory of device result */
  Map<Expr, Array<IntegerArray>> storage_device_map_;
  std::vector<int64_t> storage_info_;
  std::vector<int64_t> temporary_buffer_id_;

};

/*class AIPURelay2Network{
public:
  // Create a corresponding AIPU function for the given relay Function.
  std::pair<std::string, Array<String>> GenNetWork(const Function& func,nvdla::INetwork * network) {

    ICHECK(func.defined()) << "Input error: expect a Relay function.";
    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);

    
    AIPURelay2NetworkCore builder(sid,network);
    auto out = builder.VisitExpr(func->body);
    LOG(INFO)<<"OUT END";
    //code_stream_ << builder.JIT(out);

    return {sid, {}};
  }
};*/



}
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_AIPU_RELAY_PARSE_CORE_H_
