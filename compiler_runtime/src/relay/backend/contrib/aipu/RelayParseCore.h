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

struct Output {
  std::string name;
  std::string dtype;
  int size;
};

inline size_t GetShape1DSize(const Type& type) {
  const auto shape = GetShape(type);
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

class AIPURelay2NetworkCore : public MemoizedExprTranslator<std::vector<Output>> {
 public:
  explicit AIPURelay2NetworkCore(const std::string& id, nvdla::INetwork* network, BlobNameToTensor* blobname_to_tensor)
    : ext_func_id_(id), network_(network), blobname_to_tensor_(blobname_to_tensor) { }

  std::vector<Output> VisitExprDefault_(const Object* op) final {
    LOG(FATAL) << "AIPU codegen doesn't support: " << op->GetTypeKey();
    return {};
  }

  std::vector<Output> VisitExpr_(const VarNode* node) final {
    LOG(INFO)<<"VarNode node name: "<<node->name_hint();
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
    output.name = "(float*)(" + ext_func_id_ + "_consts[" + std::to_string(const_idx_) + "]->data)";
    const_idx_++;
    output.dtype = "float";
    LOG(INFO)<<"ConstantNode node name: "<<output.name;
    return {output};
  }

  std::vector<Output> VisitExpr_(const CallNode* call) final {
    // quantization OPs are all skipped
    if(call->op.as<OpNode>()) {
      const auto op_name = call->op.as<OpNode>()->name;
      if (op_name == "relay.op.annotation.simulated_quantize" || op_name == "annotation.stop_fusion" || op_name == "annotation.cast_hint")
	      return VisitExpr(call->args[0]);
    }
    // recursively visit the inputs, and obtain the names of the inputs
    std::vector<std::string> arg_names;
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto res = VisitExpr(call->args[i]);
      for (const auto& out : res) {
        arg_names.push_back(out.name);
      }
    }
    std::vector<Output> outputs;
    if (const auto* func = call->op.as<FunctionNode>()) {
      outputs = GenerateCompositeFunctionCall(func, call, arg_names);
    } else {
      outputs = GenerateOpCall(call, arg_names);
    }
    return outputs;
  }

 private:

  /*!
   * \brief compile a relay op to Network. TODO: Can GenerateOpCall and GenerateCompositeFunctionCall be unified?
   *
   * \param call The relay op
   * \param arg_names The names of the inputs of the call
   *
   * \return The descriptions of output tensors
   */
  std::vector<Output> GenerateOpCall(const CallNode* call, const std::vector<std::string>& arg_names) {
    const auto* op_node = call->op.as<OpNode>();
    ICHECK(op_node) << "Expect OpNode, but got " << call->op->GetTypeKey();
    static const std::map<std::string, std::string> op_map = {
        {"nn.conv2d", "aipu_conv2d"},
        {"nn.dense", "aipu_dense"},
        {"nn.relu", "aipu_relu"},
        {"clip", "aipu_relu"},
        {"nn.batch_norm", "aipu_bn"},
        {"add", "aipu_add"},
        {"reshape", "aipu_reshape"},
        {"nn.softmax", "aipu_softmax"},
        {"nn.max_pool2d", "aipu_max_pool2d"},
        {"mean", "aipu_global_pool2d"},
        {"nn.avg_pool2d", "aipu_global_pool2d"},
        {"squeeze", "aipu_squeeze"},
        {"nn.pad", "aipu_pad"},
        {"concatenate", "aipu_concatenate"},
        {"nn.leaky_relu", "aipu_leakyRelu"},
    };
    const auto op_name = GetRef<Op>(op_node)->name;
    const auto iter = op_map.find(op_name);
    if (iter != op_map.end()) {
      return GenerateBody(call, iter->second, arg_names, call);
    }
    LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
    return {};
  }

  /*!
   * \brief compile a composed relay function to Network
   *
   * \param callee The FunctionNode* to be compiled
   * \param caller The CallNode*. caller->op is callee
   * \param arg_names The names of the inputs of the function
   *
   * \return The descriptions of output tensors
   */
  std::vector<Output> GenerateCompositeFunctionCall(const FunctionNode* callee, const CallNode* caller, const std::vector<std::string>& arg_names) {
    const auto pattern_name = callee->GetAttr<runtime::String>(attr::kComposite);
    LOG(INFO) << "pattern name: " << pattern_name;
    ICHECK(pattern_name.defined()) << "Only functions with composite attribute are supported";
    if (pattern_name == "aipu.conv2d_add") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, {"nn.conv2d", "add"});
      return GenerateBody(conv_call, "aipu_fused_conv2d_bias", arg_names, caller);
    } else if (pattern_name == "aipu.dense_add") {
      const auto* dense_call = GetRootCall(callee->body.as<CallNode>(), 1, {"nn.dense", "add"});
      return GenerateBody(dense_call, "aipu_fused_dense_bias", arg_names, caller);
    } else if (pattern_name == "aipu.dense_add_extended") {
      const auto* dense_call = GetRootCall(callee->body.as<CallNode>(), 1, {"nn.dense", "add"});
      return GenerateBody(dense_call, "aipu_fused_dense_bias_extended", arg_names, caller);
    }
    LOG(FATAL) << "Unknown composite function:" << pattern_name;
    return {};
  }

  /*!
   * \brief Parse the relay op or relay function into an operator of Network
   *
   * \param root_call The first relay op in the relay function
   * \param func_name function name
   * \param func_args The names of function arguments
   * \param caller The relay function call
   *
   * \return The descriptions of output tensors
   */
  std::vector<Output> GenerateBody(const CallNode* root_call, const std::string& func_name, const std::vector<std::string>& func_args, const CallNode* caller) {
    std::string layername = "layer_" + std::to_string(layer_idx_++) + "_" + func_name;
    ICHECK_GT(func_args.size(), 0);
    LOG(INFO) << layername <<" has " << func_args.size() << " inputs";
    for(auto arg_name : func_args) {
      LOG(INFO)<<"arg name: "<<arg_name;
    }
    // Analyze the output. TODO: The following code actually analyzes "root_call". This causes no problem if root_call and output have the same shape. But what if root_call and output have different shapes?
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
    std::vector<Output> outputs;
    for (const auto& out_type : out_types) {
      const std::string out = "buf_" + std::to_string(buf_idx_++);
      const auto out_size = GetShape1DSize(out_type);
      Output output;
      output.name = out;
      output.size = out_size;
      output.dtype = "float";
      outputs.push_back(output);
    }
    LOG(INFO) << layername << " has " << outputs.size() << " outputs";
    for(auto output_content : outputs) {
      LOG(INFO)<<"output name: "<<output_content.name;
    }
    // parse 
    const auto op_name = root_call->op.as<OpNode>()->name;
    nvdla::ILayer* layer = nullptr;
    if(op_name == "nn.conv2d") {
      layer= ParseConvolution(root_call, caller, network_, blobname_to_tensor_, func_args);
    } else if(op_name == "nn.relu" || op_name == "clip") {
      layer= ParseRelu(network_, blobname_to_tensor_, func_args);
    } else if(op_name == "nn.max_pool2d"||op_name == "mean" ||op_name == "nn.avg_pool2d") {//get output?
      layer= parsePooling(root_call, network_, blobname_to_tensor_, func_args);
    } else if(op_name == "reshape" || op_name == "squeeze") { 
      nvdla::ITensor* tensor = (*blobname_to_tensor_)[func_args[0]];
      (*blobname_to_tensor_)[outputs[0].name] = tensor;
      std::cout << "Warning: " << op_name << " layer ignored." << std::endl;
      return outputs;
    } else if(op_name == "nn.dense") {
      layer= ParseDense(root_call, caller, network_, blobname_to_tensor_, func_args);
    } else if(op_name == "nn.softmax") {
      layer= parseSoftMax(network_, blobname_to_tensor_, func_args);
    } else if(op_name == "add") {
      nvdla::ElementWiseOperation op = nvdla::ElementWiseOperation::kSUM;
      layer = network_->addElementWise((*blobname_to_tensor_)[func_args[0]], (*blobname_to_tensor_)[func_args[1]], op);
    } else if(op_name == "nn.leaky_relu") {
      const auto* leaky_relu_attr = root_call->attrs.as<LeakyReluAttrs>();
      double alpha = leaky_relu_attr->alpha;
      layer = network_->addActivation((*blobname_to_tensor_)[func_args[0]], /*ActivationType::*/nvdla::ActivationType::kPRELU, alpha);
    } else if(op_name == "concatenate") {
      layer= parseConcatenate(network_, blobname_to_tensor_, func_args);
    } else {
      LOG(FATAL) << "Error: unsupport layertype: " << layername;
    }
    if (!layer) {
      LOG(FATAL) << "Error: parsing layer: " << layername;
    }
    layer->setName(layername.c_str());
    // TODO. The following only add one output to blobname_to_tensor_. What if there are multiple outputs?
    blobname_to_tensor_->add(outputs[0].name, layer->getOutput(0));
    return outputs;
  }

  /*!
   * \brief Parse convlution op. From a relay function to INetwork
   *
   * \param
   *
   * \return The pointer to the conv layer
   */
  static nvdla::ILayer* ParseConvolution(const CallNode* root_call, const CallNode* caller,
                                         nvdla::INetwork* network, BlobNameToTensor* tensor,
                                        const std::vector<std::string>& func_args) {
    nvdla::ILayer* layer = nullptr;
    nvdla::Weights kernelWeights;
    nvdla::Weights biasWeights;
    const auto* conv2d_attr = root_call->attrs.as<Conv2DAttrs>();
    ICHECK(conv2d_attr);
    assert(conv2d_attr->data_layout == "NHWC");
    assert(conv2d_attr->kernel_layout == "HWIO");
    auto ishape = GetShape(root_call->args[0]->checked_type());
    auto wshape = GetShape(root_call->args[1]->checked_type());
    if(root_call->args[1].as<VarNode>()) {
      LOG(FATAL)<<"WEIGHT AS VARNODE";
    } else if(root_call->args[1].as<ConstantNode>()) {
      int weightsize = wshape[0] * wshape[1] * wshape[2] * wshape[3];
      Constant argconst = Downcast<Constant>(root_call->args[1]);
      float* weightdata = (float*)malloc(weightsize * sizeof(float));
      memcpy(weightdata, (float*)(argconst->data->data), weightsize * sizeof(float));
      hwcn2nchw(weightdata, wshape, conv2d_attr->kernel_layout, false);
      kernelWeights = nvdla::Weights(nvdla::DataType::FLOAT, weightdata, weightsize);
    }
    int numOutputs = 0;
    if(conv2d_attr->kernel_layout=="HWOI") {
      numOutputs = wshape[2];
    }
    if(conv2d_attr->kernel_layout=="HWIO") {
      numOutputs = wshape[3];
    }

    int numGroups  = conv2d_attr->groups;
  
    // TODO. Is this correct? Should H and W be interchanged?
    int kernelW = wshape[0];
    int kernelH = wshape[1];

    // TODO. Is this correct?
    int strideW = conv2d_attr->strides[0].as<IntImmNode>()->value;
    int strideH = conv2d_attr->strides[1].as<IntImmNode>()->value;

    int padTop = conv2d_attr->padding[0].as<IntImmNode>()->value;
    int padLeft = conv2d_attr->padding[1].as<IntImmNode>()->value;
    int padBottom = conv2d_attr->padding[2].as<IntImmNode>()->value;
    int padRight = conv2d_attr->padding[3].as<IntImmNode>()->value;

    // TODO. Is this correct?
    int dilationW = conv2d_attr->dilation[0].as<IntImmNode>()->value;
    int dilationH = conv2d_attr->dilation[1].as<IntImmNode>()->value;

    // if there is an "nn.pad" op
    if(auto *padcallnode = root_call->args[0].as<CallNode>()) {
      const auto* op_node = padcallnode->op.as<OpNode>();
      const auto op_name = GetRef<Op>(op_node)->name;
      if(op_name == "nn.pad"){
        const auto* pad_attr = padcallnode->attrs.as<PadAttrs>();
        // TODO. "+=" or "="?
        padTop = pad_attr->pad_width[1][0];
        padLeft = pad_attr->pad_width[2][0];
        padBottom = pad_attr->pad_width[1][1];
        padRight = pad_attr->pad_width[2][1];
      }
      else if (op_name == "relay.op.annotation.simulated_quantize") {
        if (const auto *real_pad = padcallnode->args[0].as<CallNode>()) {
          const auto* opnode = real_pad->op.as<OpNode>();
          const auto opname = GetRef<Op>(opnode)->name;
          if(opname == "nn.pad"){
            const auto* pad_attr = real_pad->attrs.as<PadAttrs>();
            // TODO. "+=" or "="?
            padTop = pad_attr->pad_width[1][0];
            padLeft = pad_attr->pad_width[2][0];
            padBottom = pad_attr->pad_width[1][1];
            padRight = pad_attr->pad_width[2][1];
          }
        }
      }
    }
    // if there is an "add" op
    if(const auto* func = caller->op.as<FunctionNode>()) {
      auto biascallnode = func->body.as<CallNode>();
      const auto* op_node = biascallnode->op.as<OpNode>();
      const auto op_name = GetRef<Op>(op_node)->name;
      if(op_name == "add"){
        if(biascallnode->args[1].as<VarNode>()){
          LOG(FATAL)<<"bias Must ConstantNode";
        }
        if(biascallnode->args[1].as<ConstantNode>()){
          auto bshape = GetShape(biascallnode->args[1]->checked_type());
          int biassize = 1;
          for (size_t i = 0; i < bshape.size(); i++)
            biassize = biassize * bshape[i];
          Constant argconst = Downcast<Constant>(biascallnode->args[1]);
          float * biasdata = (float*)malloc(biassize*sizeof(float));
          memcpy(biasdata, (float*)argconst->data->data, biassize*sizeof(float));
          biasWeights = nvdla::Weights(nvdla::DataType::FLOAT, biasdata, biassize);
        }
      } else {
        biasWeights = nvdla::Weights(nvdla::DataType::FLOAT, NULL, 0);
      }
    }
    nvdla::BiasMode biasMode = nvdla::BiasMode::bNONE;
    if ( biasWeights.count == 0 ) {
        biasMode = nvdla::BiasMode::bNONE;
    } else if ( biasWeights.count == 1 ) {
        biasMode = nvdla::BiasMode::bUNIFORM;
    } else if ( biasWeights.count == numOutputs ) {
        biasMode = nvdla::BiasMode::bCHANNEL;
    } else {
        biasMode = nvdla::BiasMode::bm_ELEMENTWISE;
    }

    nvdla::Dims2 tlPadding = nvdla::Dims2(padTop, padLeft);
    nvdla::Dims2 brPadding = nvdla::Dims2(padBottom, padRight);
    nvdla::Dims2 stride    = nvdla::Dims2(strideH, strideW);
    nvdla::Dims2 dilation  = nvdla::Dims2(dilationH, dilationW);
    nvdla::Dims2 kernelSize= nvdla::Dims2(kernelH, kernelW);

    // TODO: cross-correlation vs convolution
    layer = network->addConvolution((*tensor)[func_args[0]], numOutputs, 0,
                                    kernelSize, tlPadding, brPadding, stride, dilation,
                                    kernelWeights, biasWeights, biasMode, numGroups);
    return layer;
  }

  static nvdla::ILayer* ParseDense(const CallNode* root_call, const CallNode* caller,
				   nvdla::INetwork* network, BlobNameToTensor* tensor,
				   const std::vector<std::string>& func_args) {
    nvdla::Weights kernelWeights;
    nvdla::Weights biasWeights;
    /*! \brief weight_flag. true: root_call->args[1] is ConstantNode; false: root_call->args[1] is VarNode */
    bool weight_flag = true;
    auto wshape = GetShape(root_call->args[1]->checked_type());
    nvdla::Dims4 dims = (*tensor)[func_args[0]]->getDimensions();
    wshape.push_back(dims.h);
    wshape.push_back(dims.w);
    wshape.push_back(dims.c);
    LOG(INFO) << "weight ishape (length " << wshape.size() << "): output channel " << wshape[0] << ", input channel " << wshape[1] << ", height " << wshape[2] << ", width " << wshape[3] << ", input channel " << wshape[4]; // Now wshape is OIHWI
    int weightsize = wshape[0] * wshape[1];
    if(root_call->args[1].as<VarNode>()) {
      weight_flag = false;
      LOG(WARNING)<<"WEIGHT IS VARNODE";
      float* weightdata = (float*)malloc(weightsize*sizeof(float));
      for(int _i =0; _i < weightsize; ++_i) {
        weightdata[_i] = 0.;
      }
      kernelWeights = nvdla::Weights(nvdla::DataType::FLOAT, weightdata, weightsize);
    }
    if(root_call->args[1].as<ConstantNode>()) {
      weight_flag = true;
      Constant argconst = Downcast<Constant>(root_call->args[1]);
      float* weightdata = (float*)malloc(weightsize*sizeof(float));
      memcpy(weightdata,(float*)(argconst->data->data),weightsize*sizeof(float));
      // TODO: it seems that for dense op, hwcn2nchw did nothing.
      hwcn2nchw(weightdata, wshape, "HWIO", true);
      kernelWeights = nvdla::Weights(nvdla::DataType::FLOAT, weightdata, weightsize);
    }

    int numOutputs = wshape[0];
    int numGroups  = 1;
    int kernelW = 1;
    int kernelH = 1;
    
    // TODO: what is the purpose here?
    if (dims.c != wshape[1]) {
      kernelW = dims.w;
      kernelH = dims.h;
    }
    
    // TODO: what is the purpose here?
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

    // if there is an "nn.pad" op
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
      } else if (op_name == "relay.op.annotation.simulated_quantize") {
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
    // if there is an "add" op
    if(const auto* func = caller->op.as<FunctionNode>()){
      auto biascallnode = func->body.as<CallNode>();
      const auto* op_node = biascallnode->op.as<OpNode>();
      const auto op_name = GetRef<Op>(op_node)->name;
      if(op_name == "add"){
        if(biascallnode->args[1].as<VarNode>()){
        }
        if(biascallnode->args[1].as<ConstantNode>()){
          auto bshape = GetShape(biascallnode->args[1]->checked_type());
          int biassize = bshape[0];
          Constant argconst = Downcast<Constant>(biascallnode->args[1]);
          float * biasdata = (float*)malloc(biassize*sizeof(float));
          memcpy(biasdata,(float*)argconst->data->data,biassize*sizeof(float));
          biasWeights = nvdla::Weights(nvdla::DataType::FLOAT, biasdata, biassize);
        }
      }
      else{
        biasWeights =nvdla::Weights(nvdla::DataType::FLOAT, NULL, 0);
      }
    }
    nvdla::BiasMode biasMode = nvdla::BiasMode::bNONE;
    // TODO: cross-correlation vs convolution
    if ( biasWeights.count == 0 ) {
      biasMode = nvdla::BiasMode::bNONE;
    } else if ( biasWeights.count == 1 ) {
      biasMode = nvdla::BiasMode::bUNIFORM;
    } else if ( biasWeights.count == numOutputs ) {
      biasMode = nvdla::BiasMode::bCHANNEL;
    } else {
      biasMode = nvdla::BiasMode::bm_ELEMENTWISE;
    }

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
    } else {
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

  static nvdla::ILayer* ParseRelu(nvdla::INetwork* network, BlobNameToTensor* tensor,
                                        const std::vector<std::string>& func_args){
    //LOG(INFO) << func_args[0];
    return network->addActivation((*tensor)[func_args[0]], /*ActivationType::*/nvdla::ActivationType::kRELU);
  }

  static nvdla::ILayer* parsePooling(const CallNode* call,nvdla::INetwork* network, BlobNameToTensor* tensor,
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
  static nvdla::ILayer* parseSoftMax(nvdla::INetwork* network, BlobNameToTensor* tensor,
                                        const std::vector<std::string>& func_args)
  {
    return network->addSoftMax((*tensor)[func_args[0]]);
  }
  static nvdla::ILayer* parseConcatenate(nvdla::INetwork* network, BlobNameToTensor* tensor,
                                         const std::vector<std::string>& func_args)
  {
    std::vector<nvdla::ITensor*> ptrs;
    for (unsigned int i = 0, n = func_args.size(); i < n; i++) {
      ptrs.push_back((*tensor)[func_args[i]]);
    }

    return network->addConcatenation(&ptrs[0], func_args.size());
  }
  static void hwcn2nchw(float *weightdata, std::vector<int>wshape, tvm::String kernel_layout, bool for_dense) {
    int shapesize;
    int N, C, H, W;
    if (for_dense) {
      H = 1;
      W = 1;
      C = wshape[1];
      // when input channel is not the same as weight channel, we use input channel. TODO: why?
      assert(wshape[1] == whape[4]);
      if (wshape[1] != wshape[4]) {
        H = wshape[2];
        W = wshape[3];
        C = wshape[4];
      }
      shapesize = wshape[0] * wshape[1];
      N = wshape[0];
    } else {
      shapesize = wshape[0] * wshape[1] * wshape[2] * wshape[3];
      if (kernel_layout == "HWIO") {
        N = wshape[3];
        C = wshape[2];
        H = wshape[0];
        W = wshape[1];
      } else {
        LOG(FATAL) << "kernel layout not supported " << kernel_layout;
      }
    }
    float *tmp_weightdata = (float*)malloc(shapesize * sizeof(float));
    memcpy(tmp_weightdata, weightdata, shapesize * sizeof(float));

    if (for_dense) {
      for (int i = 0; i < wshape[1]; i++) {
        for (int j = 0; j < N; j++) {
          tmp_weightdata[i * N + j] = weightdata[j * wshape[1] + i];
        }
      }
    }
    // HWIO -> OIHW
    for(int i=0; i<N/*N*/; i++){
      for(int j=0; j<C/*c*/; j++){
        for(int k=0; k<H/*H*/; k++){
          for(int l=0; l<W/*W*/; l++){
            weightdata[i*C*H*W + j*H*W + k*W + l] = tmp_weightdata[k*W*C*N + l*C*N + j*N + i];
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
  nvdla::INetwork *network_;
  // The map from the tensor name to nvdla::ITensor*. Note that blobname_to_tensor_ only contains feature data, not weight data.
  BlobNameToTensor* blobname_to_tensor_;

};

}  // namespace aipu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_AIPU_RELAY_PARSE_CORE_H_
