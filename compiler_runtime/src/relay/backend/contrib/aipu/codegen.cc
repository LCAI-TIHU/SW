/*
 * Inspur.
 * This is a new or modified file.
 */
#include "codegen_aipu.h"
#include "codegen_riscv.h"
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/relay/qnn/attrs.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

#include "../../../transforms/pattern_utils.h"
#include "../../utils.h"
#include "RelayParser.h"
#include "half.h"
#include "main.h"
#include "nvdla/ICompiler.h"
#include "nvdla/ILayer.h"
#include "nvdla/INetwork.h"
#include "nvdla/IProfile.h"
#include "nvdla/IProfiler.h"
#include "nvdla/IRuntime.h"
#include "nvdla/ITargetConfig.h"
#include "nvdla/IWisdom.h"
#include "nvdla/caffe/ICaffeParser.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"

#include "graph_plan_memory.h" //yuanyue 20220622 plan memory
#include "graph_aid_dtype.h" //yuanyue 20220713 aid dtype

#define TARGET_CONFIG_NAME "nv_full"

// quantization mode for NVDLA. NOTE. PER_KERNEL mode is not compatible with group conv
#define DEFAULT_QUANT_MODE nvdla::QuantizationMode::PER_FILTER  // PER_FILTER  PER_KERNEL
#define DEFAULT_BATCH_SIZE 0
#define DEFAULT_DATA_FMT nvdla::DataFormat::NHWC
// #define DEFAULT_DATA_FMT nvdla::DataFormat::NCHW

auto theQuantizationMode = nvdla::QuantizationMode::PER_FILTER;

TVM_REGISTER_GLOBAL("AIPU_config_quantization_PER_FILTER")
.set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv){
  bool if_PER_FILTER = args[0];
  if (if_PER_FILTER) {
    theQuantizationMode = nvdla::QuantizationMode::PER_FILTER;
  } else {
    theQuantizationMode = nvdla::QuantizationMode::PER_KERNEL;
  }
  *rv = true;
});

static CompilerTestAppArgs defaultCompilerTestAppArgs = {
    /* .project = */ "OpenDLA",
    /* .inputPath = */ "./",
    /* .inputName = */ "",
    /* .outputPath = */ "./",
    /* .testname = */ "",
    /* .testArgs = */ "",
    /* .prototxt = */ "",
    /* .caffemodel = */ "",
    /* .cachemodel = */ "",
    /* .profileName = */ "fast-math",
    /* .profileFile = */ "",
    /* .configtarget = */ TARGET_CONFIG_NAME,
    /* .calibtable = */ "", 
    /* .quantizationMode = */ DEFAULT_QUANT_MODE,
    /* .numBatches = */ DEFAULT_BATCH_SIZE,
    /* .inDataFormat = */ DEFAULT_DATA_FMT,
    /* .computePrecision = */ nvdla::DataType::INT8  // nvdla::DataType::INT8 HALF
};

static NvDlaError beginWithNamedProfile(const CompilerTestAppArgs* appArgs, CompilerTestInfo* i) {
  NvDlaError e = NvDlaSuccess;
  nvdla::IProfiler* profiler;
  nvdla::IProfile* profile;

  profiler = i->wisdom->getProfiler();
  if (!profiler) {
    ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "Profiler not initialized");
  }

  profile = profiler->getProfile(appArgs->profileName.c_str());
  if (!profile) {
    ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "Profile %s not initialized",
                         appArgs->profileName.c_str());
  }

fail:
  return e;
}

#if 0
NvDlaError parseTensorScales(const CompilerTestAppArgs* appArgs, CompilerTestInfo *i, nvdla::INetwork* network)
{
    NvDlaError e = NvDlaSuccess;
    NvDlaStatType stat;
    std::string calibTableFile = /*i->calibTablesPath + "/" + */appArgs->calibTable;

    PROPAGATE_ERROR_FAIL(NvDlaStat(calibTableFile.c_str(), &stat));

    // populate the scaling factor/dynamic range of each of the tensors on the network
    {
        FILE* fp = fopen(calibTableFile.c_str(), "r");
        char readBuffer[TEST_PARAM_FILE_MAX_SIZE] = {0};

        rapidjson::Document doc;
        rapidjson::FileReadStream inStr(fp, readBuffer, sizeof(readBuffer));

        doc.ParseStream(inStr);
        if (doc.HasParseError())
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "JSON parsing error: %s", GetParseError_En(doc.GetParseError()));
        }

        {
            std::vector<nvdla::ILayer*> networkLayers = network->getLayers();
            std::vector<nvdla::ITensor*> networkInputs = network->getInputs();

            std::vector<nvdla::ILayer*>::iterator li = networkLayers.begin();
            std::vector<nvdla::ITensor*>::iterator nii = networkInputs.begin();

            // set scaling factor for the network input tensors
            for (; nii != networkInputs.end(); ++nii)
            {
                NvF32 scale = 0.0f;
                NvF32 min = 0.0f;
                NvF32 max = 0.0f;
                std::string tName = (*nii)->getName();
                if (doc[tName.c_str()].HasMember("scale")) {
                    scale = doc[tName.c_str()]["scale"].GetFloat();
                    min = scale * -127.0f;
                    max = scale * 127.0f;
                }
                else if (doc[tName.c_str()].HasMember("min") && doc[tName.c_str()].HasMember("max")) {
                    min = doc[tName.c_str()]["min"].GetFloat();
                    max = doc[tName.c_str()]["max"].GetFloat();
                }
                else {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Atleast 1 of scale or min-max should be specified for %s\n", tName.c_str());
                }

                // set same dynamic range for all channels of the tensor (cIndex = -1)
                PROPAGATE_ERROR_FAIL( (*nii)->setChannelDynamicRange(-1, min, max) );
                const_cast<CompilerTestAppArgs*>(appArgs)->tensorScales.insert(std::pair<std::string, NvF32>(tName, scale));
            }

            for (; li != networkLayers.end(); ++li)
            {
                NvF32 scale = 0.0f;
                NvF32 min = 0.0f;
                NvF32 max = 0.0f;
                std::string lName = (*li)->getName();
                nvdla::ITensor* outTensor = (*li)->getOutput(0);

                if (doc[lName.c_str()].HasMember("scale")) {
                    scale = doc[lName.c_str()]["scale"].GetFloat();
                    min = scale * -127.0f;
                    max = scale * 127.0f;
                }
                else if (doc[lName.c_str()].HasMember("min") && doc[lName.c_str()].HasMember("max")) {
                    min = doc[lName.c_str()]["min"].GetFloat();
                    max = doc[lName.c_str()]["max"].GetFloat();
                }
                else {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Atleast 1 of scale or min-max should be specified for %s\n", lName.c_str());
                }

                // set same dynamic range for all channels of the tensor (cIndex = -1)
                PROPAGATE_ERROR_FAIL( outTensor->setChannelDynamicRange(-1, min, max) );
                const_cast<CompilerTestAppArgs*>(appArgs)->tensorScales.insert(std::pair<std::string, NvF32>(lName, scale));
            }
        }

        fclose(fp);
    }

fail:
    return e;
}
#endif

NvDlaError parseTensorScales(const CompilerTestAppArgs* appArgs, CompilerTestInfo* i,
                             nvdla::INetwork* network, std::vector<float>& calibdata,
                             std::vector<std::vector<float>> i_scale_Vec) {
  NvDlaError e = NvDlaSuccess;
  //NvDlaStatType stat;

  // populate the scaling factor/dynamic range of each of the tensors on the network
  std::vector<nvdla::ILayer*> networkLayers = network->getLayers();
  std::vector<nvdla::ITensor*> networkInputs = network->getInputs();

  std::vector<nvdla::ILayer*>::iterator li = networkLayers.begin();
  std::vector<nvdla::ITensor*>::iterator nii = networkInputs.begin();

  // set scaling factor for the network input tensors
  std::string index_str;
  for (int i = 0; nii != networkInputs.end(); ++nii, ++i) {
    NvF32 scale = 1.0f;
    // just for debug
    if (!calibdata.empty())
      scale = i_scale_Vec[i][0];  // DLA Node has single input
    NvF32 min = scale * -127.0f;
    NvF32 max = scale * 127.0f;
    std::string tName = (*nii)->getName();

    LOG(INFO) << "network->networkInputs() tName: " << tName;
    LOG(INFO) << "network->networkInputs() scale: " << scale;

    // set same dynamic range for all channels of the tensor (cIndex = -1)
    PROPAGATE_ERROR_FAIL((*nii)->setChannelDynamicRange(-1, min, max));
    const_cast<CompilerTestAppArgs*>(appArgs)->tensorScales.insert(
        std::pair<std::string, NvF32>(tName, scale));
  }

  for (int i = 0; li != networkLayers.end(); ++li, ++i) {
    NvF32 scale = 1.0f;
    // just for debug
    if (!calibdata.empty())
      scale = calibdata[i];

    NvF32 min = scale * -127.0f;
    NvF32 max = scale * 127.0f;
    std::string lName = (*li)->getName();
    nvdla::ITensor* outTensor = (*li)->getOutput(0);

    LOG(INFO) << "network->networkLayers() lName: " << lName;
    LOG(INFO) << "network->networkLayers() scale: " << scale;

    // set same dynamic range for all channels of the tensor (cIndex = -1)
    PROPAGATE_ERROR_FAIL(outTensor->setChannelDynamicRange(-1, min, max));
    const_cast<CompilerTestAppArgs*>(appArgs)->tensorScales.insert(
        std::pair<std::string, NvF32>(lName, scale));
  }

fail:
  return e;
}

class CollectOutput : public tvm::relay::ExprMutator {
  using Expr = tvm::RelayExpr;
  using CallNode = tvm::relay::CallNode;
  using Call = tvm::relay::Call;
public:
  std::map<Expr, std::vector<Expr>> child_parent;
  
  Expr VisitExpr_(const CallNode* op) override {
    for (auto arg : op->args) {
      auto it = child_parent.find(arg);
      if (it != child_parent.end())
	it->second.push_back(tvm::runtime::GetRef<Call>(op));
      else {
	std::vector<Expr> parents;
	parents.push_back(tvm::runtime::GetRef<Call>(op));
	child_parent.insert(std::pair<Expr, std::vector<Expr>>(arg, parents));
      }
      arg = VisitExpr(arg);
    }
    return tvm::runtime::GetRef<Call>(op);
  }
};

class CustomizedAnnotation : tvm::relay::ExprMutator {
  using Expr = tvm::RelayExpr;
  using CallNode = tvm::relay::CallNode;
  using Call = tvm::relay::Call;
  using OpNode = tvm::relay::OpNode;
  using Op = tvm::relay::Op;
  using TupleNode = tvm::relay::TupleNode;
  using VarNode = tvm::relay::VarNode;

 public:
  explicit CustomizedAnnotation(std::string compiler, std::map<Expr, std::vector<Expr>> child_parent) {
    compiler_ = compiler;
    child_parent_ = child_parent;
  }
  
  bool riscv_rule(tvm::RelayExpr expr) {
    if (expr->IsInstance<tvm::relay::CallNode>()) {
      const tvm::relay::CallNode* call_node = expr.as<tvm::relay::CallNode>();
      auto* op_node = call_node->op.as<tvm::relay::OpNode>();
      const auto op_name = tvm::runtime::GetRef<tvm::relay::Op>(op_node)->name;
      if (op_name == "add") {
        auto arg = call_node->args[0];
        if (arg->IsInstance<tvm::relay::CallNode>()) {
          const tvm::relay::CallNode* arg_call = arg.as<tvm::relay::CallNode>();
          auto* arg_op_node = arg_call->op.as<tvm::relay::OpNode>();
          const auto arg_op_name = tvm::runtime::GetRef<tvm::relay::Op>(arg_op_node)->name;
          if (arg_op_name != "relay.op.annotation.simulated_quantize" && arg_op_name != "annotation.cast_hint"
	      && arg_op_name != "annotation.stop_fusion") {
            auto arg1 = call_node->args[1];
            if (arg1->IsInstance<tvm::relay::ConstantNode>()) {
              if (arg_op_name == "nn.conv2d" || arg_op_name == "nn.dense") return false;
            }
          } else {
            auto arg1 = call_node->args[1];
            if (arg1->IsInstance<tvm::relay::CallNode>()) {
              const tvm::relay::CallNode* arg1_call = arg1.as<tvm::relay::CallNode>();
              auto* arg1_op_node = arg1_call->op.as<tvm::relay::OpNode>();
              const auto arg1_op_name = tvm::runtime::GetRef<tvm::relay::Op>(arg1_op_node)->name;
              if (arg1_op_name == "relay.op.annotation.simulated_quantize" || arg1_op_name == "annotation.cast_hint"
              || arg1_op_name == "annotation.stop_fusion") {
                for (auto parent : child_parent_[expr]) {
                  const tvm::relay::CallNode* parent_call = parent.as<tvm::relay::CallNode>();
                  auto* parent_op_node = parent_call->op.as<tvm::relay::OpNode>();
                  const auto parent_op_name = tvm::runtime::GetRef<tvm::relay::Op>(parent_op_node)->name;
                  if (parent_op_name == "relay.op.annotation.simulated_quantize") return false;
                }
              }
            }
          }
        }
      //} else if (op_name == "reshape" || op_name == "mean" || op_name == "squeeze") {
      } else if (op_name == "reshape" || op_name == "squeeze") {
        auto arg = call_node->args[0];
        if (arg->IsInstance<tvm::relay::CallNode>()) {
          const tvm::relay::CallNode* arg_call = arg.as<tvm::relay::CallNode>();
          auto* arg_op_node = arg_call->op.as<tvm::relay::OpNode>();
          const auto arg_op_name = tvm::runtime::GetRef<tvm::relay::Op>(arg_op_node)->name;
          if (arg_op_name == "relay.op.annotation.simulated_quantize" || arg_op_name == "annotation.cast_hint"
	      || arg_op_name == "annotation.stop_fusion") {
            for (auto parent : child_parent_[expr]) {
              const tvm::relay::CallNode* parent_call = parent.as<tvm::relay::CallNode>();
              auto* parent_op_node = parent_call->op.as<tvm::relay::OpNode>();
              const auto parent_op_name = tvm::runtime::GetRef<tvm::relay::Op>(parent_op_node)->name;
              if (parent_op_name == "relay.op.annotation.simulated_quantize") return false;
            }
          }
        }
      }
    }
    return true;
  }

  bool dla_rule(tvm::RelayExpr expr) { 
    if (expr->IsInstance<tvm::relay::CallNode>()) {
      const tvm::relay::CallNode* call_node = expr.as<tvm::relay::CallNode>();
      auto* op_node = call_node->op.as<tvm::relay::OpNode>();
      const auto op_name = tvm::runtime::GetRef<tvm::relay::Op>(op_node)->name;
      if (op_name == "nn.dense") {
        auto arg = call_node->args[0];
        if (arg->IsInstance<tvm::relay::CallNode>()) {
          const tvm::relay::CallNode* arg_call = arg.as<tvm::relay::CallNode>();
          auto* arg_op_node = arg_call->op.as<tvm::relay::OpNode>();
          const auto arg_op_name = tvm::runtime::GetRef<tvm::relay::Op>(arg_op_node)->name;
	  if (arg_op_name != "relay.op.annotation.simulated_quantize" && arg_op_name != "annotation.cast_hint"
	      && arg_op_name != "annotation.stop_fusion") return false;
	  else {
	    if (child_parent_[expr].size() != 1) return false;
	    for (auto parent : child_parent_[expr]) {
	      const tvm::relay::CallNode* parent_call = parent.as<tvm::relay::CallNode>();
	      auto* parent_op_node = parent_call->op.as<tvm::relay::OpNode>();
	      const auto parent_op_name = tvm::runtime::GetRef<tvm::relay::Op>(parent_op_node)->name;
	      if (parent_op_name != "add")
		return false;
	      else if (!(parent_call->args[1]->IsInstance<tvm::relay::ConstantNode>()))
		return false;
	    }
	  }
        }
      }
    }
    return true;
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    auto compiler_begin = tvm::relay::Op::Get("annotation.compiler_begin");
    auto compiler_end = tvm::relay::Op::Get("annotation.compiler_end");
    auto attrs_compiler = tvm::runtime::make_object<tvm::relay::CompilerAttrs>();
    auto attrs_default = tvm::runtime::make_object<tvm::relay::CompilerAttrs>();
    attrs_compiler->compiler = compiler_;
    attrs_default->compiler = "riscv";

    tvm::Array<Expr> call_args;
    auto new_op = VisitExpr(call_node->op);

    const auto callop_node = call_node->op.as<OpNode>();
    std::string callop_name = tvm::runtime::GetRef<Op>(callop_node)->name;

    bool callop_on_default_device = false;

    for (auto arg : call_node->args) {
      arg = VisitExpr(arg);

      if (callop_name == "relay.op.annotation.simulated_quantize") {
        if (arg->IsInstance<CallNode>()) {
          const CallNode* arg_call = arg.as<tvm::relay::CallNode>();
          const auto* compilerAttrs = arg_call->attrs.as<tvm::relay::CompilerAttrs>();
          std::string tag = compilerAttrs->compiler;
          if (tag == compiler_)
            arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_compiler), {});
          else {
            arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_default), {});
            callop_on_default_device = true;
          }
        } else if (arg->IsInstance<VarNode>()) {
          arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_default), {});
          callop_on_default_device = true;
        } else {
          // ConstantNode or TupleGetItemNode
          if (callop_on_default_device) {
            arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_default), {});
          } else {
            arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_compiler), {});
          }
        }
      } else if (callop_name == "annotation.stop_fusion" || callop_name == "annotation.cast_hint") {
        if (arg->IsInstance<CallNode>()) {
          const CallNode* arg_call = arg.as<tvm::relay::CallNode>();
          const auto* compilerAttrs = arg_call->attrs.as<tvm::relay::CompilerAttrs>();
          std::string tag = compilerAttrs->compiler;
          if (tag == compiler_)
            arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_compiler), {});
          else {
            arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_default), {});
            callop_on_default_device = true;
          }
        }
      } else {
        auto expr = tvm::runtime::GetRef<Expr>(call_node);
        auto ris = riscv_operator.find(callop_name);
        auto dla = dla_operator.find(callop_name);
        if (ris != riscv_operator.end() && dla != dla_operator.end()) {
          if (riscv_rule(expr) && dla_rule(expr)) {
            if (ris->second > dla->second) {
              arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_default), {});
              callop_on_default_device = true;
            } else {
              arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_compiler), {});
            }
          } else if (riscv_rule(expr)) {
            arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_default), {});
            callop_on_default_device = true;
          } else if (dla_rule(expr)) {
            arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_compiler), {});
          } else {
            LOG(FATAL) << callop_name << " can not satisfy any rule";
          }
        } else if (ris != riscv_operator.end()) {
          arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_default), {});
          callop_on_default_device = true;
        } else if (dla != dla_operator.end()) {
          arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_compiler), {});
        } else {
          LOG(INFO) << AsText(expr, false);
          LOG(FATAL) << callop_name << " can not support by any device";
        }
      }
      call_args.push_back(arg);
    }

    Expr result = tvm::relay::Call(new_op, call_args, call_node->attrs, call_node->type_args,
                                   call_node->span);
    if (callop_on_default_device) {
      result = tvm::relay::Call(compiler_end, {result}, tvm::Attrs(attrs_default), {});
    } else {
      result = tvm::relay::Call(compiler_end, {result}, tvm::Attrs(attrs_compiler), {});
    }
    return result;
  }

  Expr VisitExpr_(const TupleNode* tuple_node) final {
    auto compiler_begin = tvm::relay::Op::Get("annotation.compiler_begin");
    auto compiler_end = tvm::relay::Op::Get("annotation.compiler_end");
    auto attrs_compiler = tvm::runtime::make_object<tvm::relay::CompilerAttrs>();
    auto attrs_default = tvm::runtime::make_object<tvm::relay::CompilerAttrs>();
    attrs_compiler->compiler = compiler_;
    attrs_default->compiler = "riscv";

    tvm::Array<Expr> fields;
    for (auto field : tuple_node->fields) {
      field = VisitExpr(field);
      field = tvm::relay::Call(compiler_begin, {field}, tvm::Attrs(attrs_default), {});
      fields.push_back(field);
    }
    Expr result = tvm::relay::Tuple(fields);
    result = tvm::relay::Call(compiler_end, {result}, tvm::Attrs(attrs_default), {});
    return result;
  }
  
  tvm::relay::Function annotate(tvm::relay::Function func) {
    auto new_expr = VisitExpr(func);
    const tvm::relay::FunctionNode* new_funcnode =
        static_cast<const tvm::relay::FunctionNode*>(new_expr.get());
    return tvm::runtime::GetRef<tvm::relay::Function>(new_funcnode);
  }

  // int below shows priority, 2 means exclusive, priority 1 > 0

  std::map<std::string, int> riscv_operator = {{"nn.softmax", 2},
                                               {"concatenate", 2},
                                               {"image.resize", 2},
                                               {"strided_slice", 2},
                                               {"reshape", 2},
                                               {"multiply", 2},
                                               {"exp", 2},
                                               {"sigmoid", 2},
                                               {"less", 2},
                                               {"where", 2},
                                               {"take", 2},
                                               {"add", 1},
                                               {"subtract", 2},
                                               {"power", 2},
                                               {"sqrt", 2},
                                               {"divide", 2},
                                               {"nn.batch_matmul", 2},
                                               {"transpose", 2},
                                               {"expand_dims", 2},
                                               {"cast", 2},
                                               {"max", 2},
                                               {"sum", 2},
                                               {"erf", 2},
                                               {"split", 2},
                                               {"one_hot", 2},
                                               {"tanh", 2},
                                               {"nn.dense", 0},
                                               {"mean", 1},
                                               {"squeeze", 1},
                                               {"nn.max_pool2d", 2},
                                               {"nn.avg_pool2d", 2},
                                               {"featuretoweight", 2},
                                               {"AIPU_GELU", 2},
                                               {"AIPU_NORM", 2}};

  std::map<std::string, int> dla_operator = {
      {"nn.leaky_relu", 2}, {"nn.dense", 1}, {"nn.conv2d", 2},     {"nn.relu", 2},
      {"add", 0},           {"nn.pad", 2},   {"nn.max_pool2d", 1}, {"mean", 0},
      {"nn.avg_pool2d", 1}, {"squeeze", 0},  {"clip", 2}/*, {"reshape", 0}*/};
  std::map<Expr, std::vector<Expr>> child_parent_;
  std::string compiler_;
};

// TODO: is module_partition used?
tvm::IRModule module_partition(tvm::IRModule module) {
  std::set<std::string> riscv_operator_set = {"tanh",
                                              "nn.dense",
                                              "mean",
                                              "squeeze",
                                              "nn.max_pool2d",
                                              "nn.avg_pool2d"};
  // std::set<std::string> riscv_operator_set = {"nn.softmax",
  //                                             "concatenate",
  //                                             "image.resize",
  //                                             "strided_slice",
  //                                             "reshape",
  //                                             "multiply",
  //                                             "exp",
  //                                             "sigmoid",
  //                                             "add",
  //                                             "less",
  //                                             "where",
  //                                             "take",
  //                                             "subtract",
  //                                             "power",
  //                                             "sqrt",
  //                                             "divide",
  //                                             "nn.batch_matmul",
  //                                             "transpose",
  //                                             "expand_dims",
  //                                             "cast",
  //                                             "max",
  //                                             "sum",
  //                                             "erf",
  //                                             "split",
  //                                             "one_hot",
  //                                             "tanh",
  //                                             "nn.dense",
  //                                             "mean",
  //                                             "squeeze"};
  std::set<std::string> dla_operator_set = {"nn.leaky_relu", "nn.dense", "nn.conv2d",     "nn.relu",
                                            "add",           "nn.pad",   "nn.max_pool2d", "mean",
                                            "nn.avg_pool2d", "squeeze",  "clip"};

  // rule for dla operator
  tvm::runtime::TypedPackedFunc<bool(tvm::RelayExpr)> pattern_dla =
      [](tvm::RelayExpr expr) -> bool {
    if (expr->IsInstance<tvm::relay::CallNode>()) {
      const tvm::relay::CallNode* call_node = expr.as<tvm::relay::CallNode>();
      auto* op_node = call_node->op.as<tvm::relay::OpNode>();
      const auto op_name = tvm::runtime::GetRef<tvm::relay::Op>(op_node)->name;
      if (op_name == "add") {
        auto arg = call_node->args[0];
        if (arg->IsInstance<tvm::relay::CallNode>()) {
          const tvm::relay::CallNode* arg_call = arg.as<tvm::relay::CallNode>();
          auto* arg_op_node = arg_call->op.as<tvm::relay::OpNode>();
          const auto arg_op_name = tvm::runtime::GetRef<tvm::relay::Op>(arg_op_node)->name;
          if (arg_op_name == "sigmoid") return false;
        }
        auto arg1 = call_node->args[1];
        if (arg1->IsInstance<tvm::relay::ConstantNode>()) {
          const tvm::relay::CallNode* arg_call = arg.as<tvm::relay::CallNode>();
          auto* arg_op_node = arg_call->op.as<tvm::relay::OpNode>();
          const auto arg_op_name = tvm::runtime::GetRef<tvm::relay::Op>(arg_op_node)->name;
          if (arg_op_name != "nn.conv2d" && arg_op_name != "nn.dense") return false;
        }
      }
    }
    return true;
  };

  // rule for riscv operator
  tvm::runtime::TypedPackedFunc<bool(tvm::RelayExpr)> pattern_riscv =
      [](tvm::RelayExpr expr) -> bool {
    if (expr->IsInstance<tvm::relay::CallNode>()) {
      const tvm::relay::CallNode* call_node = expr.as<tvm::relay::CallNode>();
      auto* op_node = call_node->op.as<tvm::relay::OpNode>();
      const auto op_name = tvm::runtime::GetRef<tvm::relay::Op>(op_node)->name;
      if (op_name == "add") {
        auto arg = call_node->args[0];
        if (arg->IsInstance<tvm::relay::CallNode>()) {
          const tvm::relay::CallNode* arg_call = arg.as<tvm::relay::CallNode>();
          auto* arg_op_node = arg_call->op.as<tvm::relay::OpNode>();
          const auto arg_op_name = tvm::runtime::GetRef<tvm::relay::Op>(arg_op_node)->name;
          auto arg1 = call_node->args[1];
          if (arg1->IsInstance<tvm::relay::ConstantNode>()) {
            if (arg_op_name == "nn.conv2d" || arg_op_name == "nn.dense") return false;
          }
        }
      } else if (op_name == "nn.dense") {
        auto arg = call_node->args[0];
        if (arg->IsInstance<tvm::relay::CallNode>()) {
          const tvm::relay::CallNode* arg_call = arg.as<tvm::relay::CallNode>();
          auto* arg_op_node = arg_call->op.as<tvm::relay::OpNode>();
          const auto arg_op_name = tvm::runtime::GetRef<tvm::relay::Op>(arg_op_node)->name;
          if (arg_op_name != "one_hot") return false;
        }
      }
    }
    return true;
  };

  auto register_op_attr = tvm::relay::backend::GetPackedFunc("ir.RegisterOpAttr");
  // AIPU. TODO. Is pattern_riscv and pattern_dla used?
  for (auto op_name : riscv_operator_set) {
    (*register_op_attr)(op_name, "target.riscv", pattern_riscv, 10);
  }
  for (auto op_name : dla_operator_set) {
    (*register_op_attr)(op_name, "target.dla", pattern_dla, 10);
  }
  // when an operator can be supported by two devices, first target's priority > second's, for
  // example add operator will be put on dla firstly
  return std::move(tvm::relay::transform::AnnotateTarget({"dla", "riscv"}, false)(module));
}

class CustomizedAnnotation4Dla : tvm::relay::ExprMutator {
 public:
  using Expr = tvm::RelayExpr;
  using CallNode = tvm::relay::CallNode;
  using TupleNode = tvm::relay::TupleNode;
  using OpNode = tvm::OpNode;
  using Op = tvm::Op;

  explicit CustomizedAnnotation4Dla(std::vector<std::vector<size_t>> functions,
                                    std::map<Expr, size_t> expr2index) {
    functions_ = functions;
    expr2index_ = expr2index;
    functionNumber = functions.size();
    // use function number to generate compiler tag
    for (size_t i = 0; i < functionNumber; i++) {
      compilers_.push_back("dla_" + std::to_string(i));
    }
  }

  // give every subfunction a function number
  size_t function_number(Expr expr) {
    size_t i;
    auto it = expr2index_.find(expr);
    auto index = it->second;
    for (i = 0; i < functionNumber; i++)
      if (std::find(functions_[i].begin(), functions_[i].end(), index) != functions_[i].end()) {
        break;
      }
    return i;
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    auto compiler_begin = tvm::relay::Op::Get("annotation.compiler_begin");
    auto compiler_end = tvm::relay::Op::Get("annotation.compiler_end");
    auto attrs_compiler = tvm::runtime::make_object<tvm::relay::CompilerAttrs>();
    auto expr = tvm::runtime::GetRef<Expr>(call_node);
    size_t func_num = function_number(expr);
    attrs_compiler->compiler = compilers_[func_num];
    tvm::Array<Expr> call_args;
    auto new_op = VisitExpr(call_node->op);
    for (auto arg : call_node->args) {
      if (arg->IsInstance<CallNode>()) {
        arg = VisitExpr(arg);
      }
      arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_compiler), {});
      call_args.push_back(arg);
    }
    Expr result = tvm::relay::Call(new_op, call_args, call_node->attrs, call_node->type_args,
                                   call_node->span);
    result = tvm::relay::Call(compiler_end, {result}, tvm::Attrs(attrs_compiler), {});
    return result;
  }

  tvm::relay::Function annotate(tvm::relay::Function func) {
    auto new_expr = VisitExpr(func->body);
    const tvm::relay::FunctionNode* new_funcnode =
        static_cast<const tvm::relay::FunctionNode*>(new_expr.get());
    return tvm::runtime::GetRef<tvm::relay::Function>(new_funcnode);
  }

  std::vector<std::vector<size_t>> functions_;
  std::map<Expr, size_t> expr2index_;
  std::vector<std::string> compilers_;
  size_t functionNumber;
};

// split dla function whose output is tuple into subfunctions with common operators as output
tvm::IRModule eliminate_tuplenode(tvm::IRModule module) {
  using Expr = tvm::RelayExpr;
  using CallNode = tvm::relay::CallNode;
  using VarNode = tvm::relay::VarNode;
  using Var = tvm::relay::Var;
  using TupleNode = tvm::relay::TupleNode;
  using Tuple = tvm::relay::Tuple;
  using TupleGetItemNode = tvm::relay::TupleGetItemNode;
  using TupleGetItem = tvm::relay::TupleGetItem;
  using Function = tvm::relay::Function;
  using FunctionNode = tvm::relay::FunctionNode;
  using String = tvm::runtime::String;

  // we assume that tupleNode can't be in fused function except the output, right now if tupleNode
  // is in fused function, it means the fused function has a concat operator, this can not
  // happen(concat is on riscv device)
  class DependencyMap : public tvm::relay::ExprVisitor {
   public:
    DependencyMap() { index = 0; }
    // map between id and its input id
    std::map<size_t, std::vector<size_t>> dependency_map;
    // map between expr and index id
    std::map<Expr, size_t> expr2index;
    size_t index;

   protected:
    void VisitExpr_(const CallNode* op) override {
      auto expr = tvm::runtime::GetRef<Expr>(op);
      std::vector<size_t> vec;
      for (auto arg : op->args) {
        VisitExpr(arg);
        if (arg->IsInstance<CallNode>() || arg->IsInstance<VarNode>()) {
          auto it = expr2index.find(arg);
          vec.push_back(it->second);
        }
      }
      dependency_map.insert(std::pair<size_t, std::vector<size_t>>(index, vec));
      expr2index.insert(std::pair<Expr, size_t>(expr, index));
      index++;
    }

    void VisitExpr_(const VarNode* op) {
      auto expr = tvm::runtime::GetRef<Expr>(op);
      std::vector<size_t> vec;
      expr2index.insert(std::pair<Expr, size_t>(expr, index));
      dependency_map.insert(std::pair<size_t, std::vector<size_t>>(index, vec));
      index++;
    }
  };

  // imitate DefuseOps pass
  class DefunctionMutator : public tvm::relay::ExprMutator {
   public:
    class FuncBodyMutator : public tvm::relay::ExprMutator {
     public:
      explicit FuncBodyMutator(const tvm::Array<Expr>& args, const tvm::Array<Var>& params)
          : ExprMutator() {
        args_ = args;
        params_ = params;
      }

      Expr VisitExpr_(const VarNode* n) {
        const std::string& name = n->name_hint();
        size_t i;
        for (i = 0; i < params_.size(); i++) {
          const std::string& param_name = params_[i]->name_hint();
          if (param_name == name) break;
        }

        ICHECK(i >= 0 && i < args_.size());
        return args_[i];
      }

      Expr VisitExpr_(const CallNode* op) {
        tvm::Array<Expr> call_args;
        for (auto arg : op->args) {
          arg = VisitExpr(arg);
          call_args.push_back(arg);
        }
        return tvm::relay::Call(op->op, call_args, op->attrs, op->type_args, op->span);
      }

     private:
      tvm::Array<Expr> args_;
      tvm::Array<Var> params_;
    };

    Expr VisitExpr_(const CallNode* n) {
      auto new_n = ExprMutator::VisitExpr_(n);

      if (const auto* call = new_n.as<CallNode>()) {
        if (const auto* func = call->op.as<FunctionNode>()) {
          std::string compiler = func->GetAttr<String>(tvm::relay::attr::kCompiler).value();
          if (compiler == "dla") {
            if (func->body->IsInstance<TupleNode>()) {
              auto expr = FuncBodyMutator(call->args, func->params).Mutate(func->body);
              return expr;
            }
          }
        }
      }
      return new_n;
    }
    Expr VisitExpr_(const TupleGetItemNode* n) {
      auto t = this->Mutate(n->tuple);
      if (const auto* call = t.as<CallNode>()) {
        if (const auto* func = call->op.as<FunctionNode>()) {
          if (n->tuple == t) {
            return tvm::runtime::GetRef<Expr>(n);
          } else {
            return TupleGetItem(t, n->index, n->span);
          }
        }
      }
      return tvm::runtime::Downcast<Tuple>(t)->fields[n->index];
    }
  };

  // traversal class for dealing with dla function with tuple as output
  class Elimitation : public tvm::relay::ExprMutator {
   protected:
    // count every node's out degree (downstream node)
    std::map<size_t, size_t> count_degree(const std::map<size_t, std::vector<size_t>>& dmap) {
      std::map<size_t, size_t> map;
      for (auto dit : dmap) {
        map.insert(std::pair<size_t, size_t>(dit.first, 0));
      }
      for (auto dit : dmap) {
        for (auto vit : dit.second) {
          auto it = map.find(vit);
          it->second += 1;
        }
      }
      return map;
    }

    // partion the graph from the output to input:
    // 1. if the current node's degree is 1, then it will be set in current subfunction
    // 2. if the current node's cross is 1 (its' output will only be consumed by one branch), it
    // will be set in current subfunction
    void partition_internal(const std::map<size_t, std::vector<size_t>>& dep_map,
                            const std::map<size_t, size_t>& deg_map,
                            const std::map<size_t, std::vector<size_t>>& cross_map, size_t expr,
                            const std::vector<size_t>& output, std::vector<size_t>& subfunc) {
      if (std::find(subfunc.begin(), subfunc.end(), expr) == subfunc.end()) {
        subfunc.push_back(expr);
      }
      auto dep_it = dep_map.find(expr);
      // make sure expr is not input node
      if (dep_it->second.size()) {
        for (auto vec_it : dep_it->second) {
          if (std::find(output.begin(), output.end(), vec_it) == output.end()) {
            auto deg_it = deg_map.find(vec_it);
            auto cross_it = cross_map.find(vec_it);
            // partition rule
            if (deg_it->second == 1 || (cross_it->second).size() == 1)
              partition_internal(dep_map, deg_map, cross_map, vec_it, output, subfunc);
          }
        }
      }
    }

    std::vector<size_t> new_output(const std::vector<size_t>& deleted,
                                   const std::map<size_t, std::vector<size_t>>& dep_map) {
      std::vector<size_t> output;
      for (auto vec_it : deleted) {
        auto dep_it = dep_map.find(vec_it);
        if (dep_it->second.size()) {
          for (auto expr : dep_it->second) {
            if (std::find(output.begin(), output.end(), expr) == output.end() &&
                std::find(deleted.begin(), deleted.end(), expr) == deleted.end()) {
              output.push_back(expr);
            }
          }
        }
      }
      return output;
    }

    std::map<size_t, std::vector<size_t>> new_graph(
        const std::vector<size_t>& ne_output,
        const std::map<size_t, std::vector<size_t>>& dep_map) {
      std::map<size_t, std::vector<size_t>> ne_graph;
      std::vector<size_t> reached;
      size_t index = 0;
      for (auto expr : ne_output) {
        reached.push_back(expr);
        while (1) {
          size_t tmp = reached[index];
          auto it = dep_map.find(tmp);
          for (auto elem : it->second) {
            if (std::find(reached.begin(), reached.end(), elem) == reached.end())
              reached.push_back(elem);
          }
          index++;
          if (index == reached.size()) break;
        }
      }
      for (auto node : reached) {
        auto it = dep_map.find(node);
        ne_graph.insert(std::pair<size_t, std::vector<size_t>>(node, it->second));
      }
      return ne_graph;
    }

    // from output to input, calculate every node's output will be consumed by which branch
    std::map<size_t, std::vector<size_t>> calculate_cross(
        const std::vector<size_t>& output, const std::map<size_t, std::vector<size_t>>& dep_map) {
      size_t branch_num = 0;
      std::map<size_t, std::vector<size_t>> cross_map;
      for (auto it : dep_map) {
        std::vector<size_t> vec;
        cross_map.insert(std::pair<size_t, std::vector<size_t>>(it.first, vec));
      }
      for (auto expr : output) {
        std::vector<size_t> reached;
        reached.push_back(expr);
        size_t index = 0;
        while (1) {
          size_t tmp = reached[index];
          auto it = dep_map.find(tmp);
          for (auto elem : it->second) {
            if (std::find(reached.begin(), reached.end(), elem) == reached.end())
              reached.push_back(elem);
          }
          index++;
          if (index == reached.size()) break;
        }
        for (auto node : reached) {
          auto it = cross_map.find(node);
          it->second.push_back(branch_num);
        }
        branch_num++;
      }
      return cross_map;
    }

    Expr VisitExpr_(const CallNode* op) override {
      tvm::Array<Expr> call_args;
      for (auto arg : op->args) {
        arg = VisitExpr(arg);
        call_args.push_back(arg);
      }
      Function func;
      if (op->op.as<FunctionNode>()) {
        func = tvm::runtime::GetRef<Function>(op->op.as<FunctionNode>());
        // LOG(INFO) << AsText(func, false);
      } else {
        LOG(INFO) << tvm::AsText(op->op, false);
        LOG(FATAL) << "TVM runtime does not support calls to " << op->op->GetTypeKey();
      }

      std::string compiler = func->GetAttr<String>(tvm::relay::attr::kCompiler).value();
      if (compiler == "dla") {
        if (func->body->IsInstance<TupleNode>()) {
          // store partitions
          std::vector<std::vector<size_t>> functions;
          auto symbol_name = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
          DependencyMap dependency_map;
          dependency_map.VisitExpr(func->body);
          std::map<size_t, std::vector<size_t>> dep_map = dependency_map.dependency_map;
          std::map<Expr, size_t> expr2index = dependency_map.expr2index;
          std::vector<size_t> output;
          auto tuple = tvm::runtime::Downcast<Tuple>(func->body);
          // get output id
          for (auto field : tuple->fields) {
            if (field->IsInstance<CallNode>()) {
              auto it = expr2index.find(field);
              output.push_back(it->second);
            }
          }

          do {
            auto deg_map = count_degree(dep_map);
            auto cross_map = calculate_cross(output, dep_map);
            // put a node in subfunction, it will be deleted from original function
            std::vector<size_t> deleted;
            for (auto expr : output) {
              std::vector<size_t> subfunc;
              partition_internal(dep_map, deg_map, cross_map, expr, output, subfunc);
              functions.push_back(subfunc);
              for (auto sub : subfunc) deleted.push_back(sub);
            }
            // get new output from deleted nodes
            output = new_output(deleted, dep_map);
            // get new graph from new output
            dep_map = new_graph(output, dep_map);
          } while (output.size());  // finally, every node will be put on a subfunction, then the
                                    // output will be empty

          // we use partitions and expr2index map to split the original function
          CustomizedAnnotation4Dla custom_annotation(functions, expr2index);
          auto new_func = custom_annotation.annotate(func);
          auto mod = tvm::IRModule::FromExpr(new_func);
          mod = tvm::relay::transform::MergeCompilerRegions()(mod);
          mod = tvm::relay::transform::PartitionGraph()(mod);
          mod = tvm::relay::transform::FuseOps()(mod);
          mod = tvm::relay::transform::Inline()(mod);
          mod = tvm::relay::transform::InferType()(mod);
          new_func = tvm::runtime::Downcast<Function>(mod->Lookup("main"));
          new_func = WithAttr(std::move(new_func), tvm::attr::kGlobalSymbol, symbol_name.value());
          new_func = WithAttr(std::move(new_func), tvm::relay::attr::kCompiler,
                              tvm::runtime::String("dla"));

          // new func's output is a tuple, and has no single operators except fused subfunction
          // whose output is regular operator
          return tvm::relay::Call(new_func, call_args, op->attrs, op->type_args, op->span);
        }
      }
      return tvm::relay::Call(op->op, call_args, op->attrs, op->type_args, op->span);
    }
  };

  auto func = tvm::runtime::Downcast<Function>(module->Lookup("main"));
  auto expr = Elimitation().VisitExpr(func->body);
  // like defuse op pass, we take out the body of dla function whose output is tuple and eliminate
  // redundant tuple
  expr = DefunctionMutator().Mutate(expr);
  auto result = tvm::IRModule::FromExpr(expr);
  result = tvm::relay::transform::InferType()(result);
  return result;
}

namespace tvm {
namespace relay {
namespace contrib {
namespace aipu {

using IntegerArray = Array<Integer>;

TVM_REGISTER_GLOBAL("relay.ext.aipu.PatterTableResult")
    .set_body_typed([](Array<String> pattern_names, Array<DFPattern> patterns,
                       Array<String> checkfuncname) {
      return PatterTableResult(pattern_names, patterns, checkfuncname);
    });
    
std::pair<uint64_t, uint8_t*> CompileFunc4Dla(Function func, std::vector<float> calibdata,
                                              std::vector<std::vector<float>> i_scale_Vec) {
  LOG(INFO) << "################################ Begin CompileFunc4Dla #######################################";
  LOG(INFO) << "Original Relay Function: " << std::endl << AsText(func, false);
  if (func->body->IsInstance<VarNode>()) {
    return std::pair<uint64_t, uint8_t*>(0, nullptr);
  }

  // merge expressions to deal with the case where multiple relay oeprators map to a single DLA operator
  auto symbol_name = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  func = WithAttr(std::move(func), tvm::attr::kGlobalSymbol, runtime::String("main"));
  func = WithAttr(std::move(func), attr::kCompiler, NullValue<ObjectRef>());
  auto mod = tvm::IRModule::FromExpr(func);
  std::string ext_patterntable = "relay.ext.aipu.patterntable";
  auto pf = tvm::runtime::Registry::Get(ext_patterntable);
  if (pf != nullptr) {
    Array<PatterTableResult> pattern_table = (*pf)();
    std::vector<PackedFunc> checksfunc;
    for (auto pattern : pattern_table) {
      for (auto checkname : pattern->checkfuncname) {
        auto getpf = tvm::runtime::Registry::Get(static_cast<std::string>(checkname));
        checksfunc.push_back(*getpf);
      }
    }
    mod = transform::MergeComposite(pattern_table[0]->pattern_names, pattern_table[0]->patterns,
                                    checksfunc)(mod);
  }
  auto funcmod = Downcast<Function>(mod->Lookup("main"));
  funcmod = WithAttr(std::move(funcmod), tvm::attr::kGlobalSymbol, symbol_name.value());
  LOG(INFO) << "Relay Function after Composition: " << std::endl << AsText(funcmod, false);

  // prepare configs
  CompilerTestAppArgs testAppArgs = defaultCompilerTestAppArgs;
  CompilerTestInfo testInfo;
  testInfo.wisdom = nullptr;
  testInfo.pData = nullptr;
  testInfo.wisdomPath = testAppArgs.outputPath + "wisdom.dir/";
  std::string removeCmd = "rm -rf " + testInfo.wisdomPath;
  if (std::system(removeCmd.c_str()) != 0) LOG(FATAL) << "system command failed: " << removeCmd;
  NvDlaMkdir(const_cast<char*>(testInfo.wisdomPath.c_str())); // is const_cast<char*> necessary?

  testInfo.wisdom = nvdla::createWisdom();
  if (!testInfo.wisdom) LOG(FATAL) << "createWisdom() failed";
  if (!testInfo.wisdom->open(testInfo.wisdomPath))
    LOG(FATAL) << "wisdom->open() failed to open: " << testInfo.wisdomPath;

  // IR translation: relay to INetwork
  nvdla::INetwork* network = nullptr;
  network = nvdla::createNetwork();
  if (!network) LOG(FATAL) << "createNetwork() failed";
  RelayParser parser;
  parser.parse(funcmod, network);

  // Parse quantization scales
  if (testAppArgs.computePrecision == nvdla::DataType::INT8) {
    LOG(INFO) << "parsing calibration table...";
    parseTensorScales(&testAppArgs, &testInfo, network, calibdata, i_scale_Vec);
  }

  // compile INetwork to loadable
  LOG(INFO) << "attaching parsed network to the wisdom...";
  if (!testInfo.wisdom->setNetworkTransient(network)) LOG(FATAL) << "wisdom->setNetworkTransient() failed";
  if (testInfo.wisdom->getNetwork()->getNumOutputs() == 0) LOG(FATAL) << "output number is 0";
  nvdla::ICompiler* compiler = testInfo.wisdom->getCompiler();
  if (!compiler) LOG(FATAL) << "wisdom->getCompiler() failed";
  if (testAppArgs.configtarget == "") LOG(FATAL) << "No target config found to load";

  std::string profileName = "";
  if (testAppArgs.profileName != "") {
    beginWithNamedProfile(&testAppArgs, &testInfo);
    profileName = testAppArgs.profileName;
  } else {
    LOG(FATAL) << "No profile name supplied";
  }
  nvdla::DataFormat inDataFormat = testAppArgs.inDataFormat;

  nvdla::IProfiler* profiler;
  nvdla::IProfile* profile;
  profiler = testInfo.wisdom->getProfiler();
  if (!profiler) LOG(FATAL) << "wisdom->getProfiler() failed";
  profile = profiler->getProfile(profileName.c_str());
  if (!profile) LOG(FATAL) << "profiler->getProfile() failed";
  profile->setComputePrecision(testAppArgs.computePrecision);
  profile->setNetworkInputDataFormat(inDataFormat);

  // TODO. What is the difference between setNetworkInputDataFormat and setNetworkInputSurfaceFormat?
  switch (inDataFormat) {
    case nvdla::DataFormat::NHWC:
      if (testAppArgs.computePrecision == nvdla::DataType::HALF) {
        profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::A16B16G16R16_F);
      } else if (testAppArgs.computePrecision == nvdla::DataType::INT8) {
        profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::FEATURE); // A8B8G8R8 R8 FEATURE
      } else {
        LOG(FATAL) << "NHWC and compute precision " << testAppArgs.computePrecision.v() << " is not yet supported";
      }
      break;
    case nvdla::DataFormat::NCxHWx:
    case nvdla::DataFormat::NCHW:
    case nvdla::DataFormat::UNKNOWN:  // at least start the test with feature data format
    default:
      if (std::strcmp(testAppArgs.configtarget.c_str(), "opendla-small") == 0)
        profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::FEATURE_X8);
      else
        profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::FEATURE);
  }
  if (testAppArgs.computePrecision == nvdla::DataType::INT8) {
    profile->setTensorScalingMode(nvdla::TensorScalingMode::PER_TENSOR);
    //switch (appArgs->quantizationMode) {}
    switch (theQuantizationMode) {
      case nvdla::QuantizationMode::PER_FILTER:
        profile->setQuantizationMode(nvdla::QuantizationMode::PER_FILTER);
        break;
      case nvdla::QuantizationMode::PER_KERNEL:
      case nvdla::QuantizationMode::NONE:  // default to per-kernel; find a way to run int8 tests w/
                                           // NONE qtzMode cleanly
      default:
        profile->setQuantizationMode(nvdla::QuantizationMode::PER_KERNEL);
    }
  } else {
    profile->setTensorScalingMode(nvdla::TensorScalingMode::NONE);
    profile->setQuantizationMode(nvdla::QuantizationMode::NONE);
  }

  profile->setNetworkOutputDataFormat(nvdla::DataFormat::NCxHWx);

  if (std::strcmp(testAppArgs.configtarget.c_str(), "opendla-small") == 0)
    profile->setNetworkOutputSurfaceFormat(nvdla::PixelFormat::FEATURE_X8);
  else
    profile->setNetworkOutputSurfaceFormat(nvdla::PixelFormat::FEATURE);

  if (testAppArgs.numBatches > 0)
    profile->setMultiBatchSize(testAppArgs.numBatches);

  compiler->compile(profileName.c_str(), testAppArgs.configtarget.c_str(), &(testInfo.compiledLoadable));
  uint64_t size = 0;
  compiler->getLoadableImageSize(profileName.c_str(), &size);
  if (size == 0) LOG(FATAL) << "Invalid size for a loadable";
  uint8_t* buf = nullptr;
  buf = (NvU8*)NvDlaAlloc(size);
  if (buf == nullptr) LOG(FATAL) << "Failed to allocate buffer for loadable";
  compiler->getLoadableImage(profileName.c_str(), buf);

  // Destroy network before closing wisdom context 
  nvdla::destroyNetwork(testInfo.wisdom->getNetwork());
  LOG(INFO) << "closing wisdom context...";
  testInfo.wisdom->close();
  if (testInfo.wisdom != nullptr) {
    nvdla::destroyWisdom(testInfo.wisdom);
    testInfo.wisdom = nullptr;
  }
  LOG(INFO) << "################################  End CompileFunc4Dla  #######################################";
  return std::pair<uint64_t, uint8_t*>(size, buf);
}

/**
 * @brief dense split
 * VisitExpr by ExprMutator, functionally updates the AST.
 * axis = 1
 * "dense-add"
 * splited in axis 1 both for input and weight
 * used operators: split, nn.dense, add, expand_dims, concatenate, sum
 * annotation: weight_thr is the threshold of dla weight bank.
 * dense with input = (x,y), weight = (z,y)
 * y>weight_thr, y%weight_thr!=0, y%(int(y/weight_thr)+1)==0
 * split the intput and weight into n_part = int(y/weight_thr)+1
 * the splited dense with input = (x, y/n_part), weight = (z,y/n_part)
 * the weight memcpy:
 * w = [T11,T21,...Tx1,T21,...Tx2,......T1y...Txy]
 * memcpy copy the splitted weight part from the origion weight tensor.
 * example: vgg16 fc1:
 * ty_SRC=Tensor[(1, 25088), float32]
 * ty_AUX=Tensor[(4096, 25088), float32]
 * ty_DST=Tensor[(1, 4096), float32]
 * it will be splitted into 4 small dense:
 * ty_SRC_sp=Tensor[(1, 6272), float32]
 * ty_AUX_sp=Tensor[(4096, 6272), float32]
 * ty_DST_sp=Tensor[(1, 4096), float32]
 */
class DenseExpr : public ExprMutator {
  using Branch = std::vector<const CallNode*>;
  using Group = std::vector<Branch>;

 public:
  int weight_thr = 7168;//2048;//7168;
  bool is_quantize = false;
  bool is_layout_ = false; ///yuaynue for layout

  DenseExpr() {}
  // split the overthreshold dense
  Call SplitByC(tvm::Array<Expr> new_args, const CallNode* op) {
    auto add_op = tvm::relay::Op::Get("add");
    auto dense_op = tvm::relay::Op::Get("nn.dense");
    const tvm::relay::CallNode* add_call;
    const tvm::relay::CallNode* dense_call;
    const tvm::relay::CallNode* arg_call = new_args[0].as<tvm::relay::CallNode>();
    Constant Add_Constant;
    if(arg_call->op == add_op){
      add_call = new_args[0].as<tvm::relay::CallNode>();
      dense_call = add_call->args[0].as<tvm::relay::CallNode>();
      if (add_call->args[1]->IsInstance<ConstantNode>()) {
        Add_Constant = Downcast<Constant>(add_call->args[1]);
      } else {
        LOG(FATAL) << "arg error" << AsText(add_call->args[1], false);
      }
    }
    else if(arg_call->op == dense_op){
      dense_call = new_args[0].as<tvm::relay::CallNode>();
      if (new_args[1]->IsInstance<ConstantNode>()) {
        Add_Constant = Downcast<Constant>(new_args[1]);
      } else {
        LOG(FATAL) << "arg error" << AsText(new_args[1], false);
      }
    }
    if (!dense_call) LOG(FATAL) << "arg error" << AsText(new_args[0], false);

    const tvm::relay::CallNode* dense_call2 = op->args[0].as<tvm::relay::CallNode>();
    auto ishape = GetShape(dense_call2->args[0]->checked_type());
	
    unsigned int quantize_args_size = 2;
    if(arg_call->op == add_op){
      auto quantize_call = dense_call->args[0].as<tvm::relay::CallNode>();
      quantize_args_size = quantize_call->args.size();
    }
    Constant scale_in_[quantize_args_size-1];
    Constant scale_out_[quantize_args_size-1];
    //quantize par
    if(arg_call->op == add_op){
      tvm::relay::Expr quantize_in = dense_call->args[0];
      tvm::relay::Expr quantize_out = tvm::runtime::GetRef<Expr>(op);
      const tvm::relay::CallNode* arg_call_quantize_in = quantize_in.as<tvm::relay::CallNode>();
      for(unsigned int i=1; i<quantize_args_size; i++){
        Constant quantize_constant = Downcast<Constant>(arg_call_quantize_in->args[i]);
        auto quantize_data = tvm::runtime::NDArray::Empty({}, DLDataType{kDLFloat, 32, 1}, {kDLCPU, 0});
        void* param_quantize;
        size_t size_quantize = 1* sizeof(float);
        param_quantize = (void*)(new char[size_quantize]);
        memcpy(param_quantize, static_cast<char*>(quantize_constant->data->data), size_quantize);
        quantize_data.tvm::runtime::NDArray::CopyFromBytes(param_quantize, size_quantize);
        scale_in_[i-1] =  relay::Constant(quantize_data);
      }
      const tvm::relay::CallNode* arg_call_quantize_out = quantize_out.as<tvm::relay::CallNode>();
      for(unsigned int i=1; i<quantize_args_size; i++){
        Constant quantize_constant = Downcast<Constant>(arg_call_quantize_out->args[i]);
        auto quantize_data = tvm::runtime::NDArray::Empty({}, DLDataType{kDLFloat, 32, 1}, {kDLCPU, 0});
        void* param_quantize;
        size_t size_quantize = 1* sizeof(float);
        param_quantize = (void*)(new char[size_quantize]);
        memcpy(param_quantize, static_cast<char*>(quantize_constant->data->data), size_quantize);
        quantize_data.tvm::runtime::NDArray::CopyFromBytes(param_quantize, size_quantize);
        scale_out_[i-1] =  relay::Constant(quantize_data);
      }
    }
    // split n part
    auto Dense_Constant = dense_call->args[1];
    int dense_channels = int(tvm::relay::backend::GetShape(Dense_Constant->checked_type())[0]);
    int data_shape_row = int(tvm::relay::backend::GetShape(Dense_Constant->checked_type())[1]);
    int sp_n_part = int(data_shape_row / weight_thr) +
                    1;  // yuanyue why is "+1" ,why not use "std:: ceil" instead
    if (data_shape_row % weight_thr == 0) {
      sp_n_part = int(data_shape_row / weight_thr);
    } else if (data_shape_row % weight_thr != 0 && sp_n_part != 0 &&
               data_shape_row % sp_n_part == 0) {
      //sp_n_part = sp_n_part;
    } else if (data_shape_row % weight_thr != 0 && sp_n_part != 0 &&
               data_shape_row % sp_n_part != 0 && data_shape_row % (sp_n_part + 1) == 0) {
      sp_n_part = sp_n_part + 1;
    } else {
      LOG(FATAL) << "split not divisible, please check";
    }

    // dense weight
    Constant weight_const;
    std::vector<int> wshape;
    size_t weight_size_get = 1;
    if (Dense_Constant->IsInstance<ConstantNode>()) {
      weight_const = Downcast<Constant>(Dense_Constant);
      wshape = tvm::relay::backend::GetShape(Dense_Constant->checked_type());
    } else {
      LOG(FATAL) << "arg error" << AsText(Dense_Constant, false);
    }
    for (size_t i = 0; i < wshape.size(); i++) {
      weight_size_get *= static_cast<size_t>(wshape[i]);
    }
    Constant weight_split[sp_n_part];
    size_t weight_size = weight_size_get;
    size_t weight_size_sp = weight_size / sp_n_part;
    size_t weight_size_sp_bit = (sizeof(float) * weight_size) / sp_n_part;

    void* param_weights[sp_n_part];
    int size[sp_n_part][dense_channels];
    int from_offset[sp_n_part][dense_channels];
    for (int i = 0; i < sp_n_part; i++) {
      param_weights[i] = (void*)(new char[weight_size_sp_bit]);
    }
    for (int i = 0; i < sp_n_part; i++) {
      for (int j = 0; j < dense_channels; j++) {
        size[i][j] = (data_shape_row / sp_n_part) * sizeof(float);
        from_offset[i][j] = j * sp_n_part * size[i][j] + i * size[i][j];
        memcpy(static_cast<char*>(param_weights[i]) + j*size[i][j], static_cast<char*>(weight_const->data->data) + from_offset[i][j], size[i][j]);
      }
    }
    for (int i = 0; i < sp_n_part; i++) {
      auto weight_data = tvm::runtime::NDArray::Empty({dense_channels, data_shape_row / sp_n_part},
                                                      DLDataType{kDLFloat, 32, 1}, {kDLCPU, 0});
      weight_data.tvm::runtime::NDArray::CopyFromBytes(param_weights[i],
                                                       weight_size_sp * sizeof(float));
      weight_split[i] = relay::Constant(weight_data);
    }

    int64_t spTh = data_shape_row / sp_n_part;

    //int index = 0;
    tvm::RelayExpr slice[sp_n_part];
    tvm::RelayExpr dense[sp_n_part];
    tvm::RelayExpr dense_exp[sp_n_part];
    Array<Expr> tuple;
    static const Op& quantize_op = relay::Op::Get("relay.op.annotation.simulated_quantize");
    auto quantize_attrs = make_object<tvm::relay::qnn::SimulatedQuantizeAttrs>();
    // quantize_attrs->kind = 1;
    // quantize_attrs->sign = true;
    // quantize_attrs->rounding = "round";
    quantize_attrs->axis = -1;

    for (int i = 0; i < sp_n_part; i++) {
      Array<Integer> begin;
      Array<Integer> end;
      Array<Integer> strides;
      for (size_t k = 0; k < wshape.size() - 1; ++k) {
        begin.push_back(0);
        end.push_back(ishape[0]);
        strides.push_back(1);
      }
	  
      begin.push_back(0 + i * spTh);
      end.push_back(((i + 1)*spTh));
	  strides.push_back(1);

	  slice[i] = MakeStridedSlice(dense_call->args[0], begin, end, strides, "end");
	  
      if(is_quantize){
        auto splitted_quantize = relay::Call(quantize_op, {slice[i], scale_in_[0], scale_in_[1], scale_in_[2]}, Attrs(quantize_attrs), {});
        dense[i] = MakeDense(splitted_quantize, weight_split[i], dense_channels,DataType::Float(32));
      }else{
        dense[i] = MakeDense(slice[i], weight_split[i], dense_channels,DataType::Float(32));
      }
	  
      //"add" after dense needed
      auto add_op = relay::Op::Get("add");
      auto c_data = tvm::runtime::NDArray::Empty({dense_channels}, {kDLFloat, 32, 1}, {kDLCPU, 0});
      auto c1 = relay::Constant(c_data);
      if (i < sp_n_part - 1){
        dense[i] = relay::Call(add_op, {dense[i], c1});
        if(is_quantize){
          dense[i] = relay::Call(quantize_op, {dense[i], scale_out_[0], scale_out_[1], scale_out_[2]}, Attrs(quantize_attrs), {});
        }
      }
      if (i == sp_n_part - 1){
        dense[i] = relay::Call(add_op, {dense[i], Add_Constant});
        if(is_quantize){
          dense[i] = relay::Call(quantize_op, {dense[i], scale_out_[0], scale_out_[1], scale_out_[2]}, Attrs(quantize_attrs), {});
        }
      }
      // expand the shape (axis = 2)
      dense_exp[i] = MakeExpandDims(dense[i], 0, 1);
      tuple.push_back(dense_exp[i]);
    }

    // concatenate and sum
    auto new_op = MakeConcatenate(Tuple(tuple), 0);
    Array<Integer> axis;
    axis.push_back(0);
    auto sp_dense_call_merge = MakeReduce(new_op, axis, 0, 0, "sum");
    if(is_quantize) sp_dense_call_merge = relay::Call(quantize_op, {sp_dense_call_merge, scale_out_[0], scale_out_[1], scale_out_[2]}, Attrs(quantize_attrs), {});
    auto sp_dense_call = Downcast<Call>(sp_dense_call_merge);
    // LOG(INFO)<<"splitted after split merge "<<AsText(sp_dense_call_merge, false);

    return sp_dense_call;
  }

  Call Layout_weights(const CallNode* op, const CallNode* reshape_call) {
    //const tvm::relay::CallNode* reshape_call = op->args[0].as<tvm::relay::CallNode>();
    if (!reshape_call)
      LOG(FATAL) << "arg error" << AsText(op->args[0], false);
    auto wshape = tvm::relay::backend::GetShape(op->args[1]->checked_type());
    int weightsize = wshape[0]*wshape[1];
    //LOG(INFO)<<"wshape: "<<wshape[0]<<", "<<wshape[1];
    Constant argconst = Downcast<Constant>(op->args[1]);
    float * weightdata = (float*)malloc(weightsize*sizeof(float));
    memcpy(weightdata,(float*)argconst->data->data,weightsize*sizeof(float));

    std:: vector <int> iishape;
    iishape=tvm::relay::backend::GetShape(reshape_call->args[0]->checked_type());
    //LOG(INFO) << "wshape.size(): " << wshape.size() << "iishape.size(): " << iishape.size();
    //LOG(INFO) << AsText(GetRef<Expr>(op),false);
    if (iishape.size() !=4){
      auto call_op =GetRef<Call>(op);
      return call_op;
      //return Call(op->op, op->args, op->attrs, op->type_args);
    }

    if (iishape[0]*iishape[1]*iishape[2]*iishape[3] != wshape[1])
      iishape[0]=wshape[1]/(iishape[1]*iishape[2]*iishape[3]);
    iishape.insert(iishape.begin(),wshape[0]);
    convert_dense_weights_NHWC2NCHW(weightdata,iishape);

    memcpy((float*)argconst->data->data,weightdata,weightsize*sizeof(float));
    return Call(op->op, op->args, op->attrs, op->type_args);
  }

  static void convert_dense_weights_NHWC2NCHW(float* weightdata, std::vector<int> wshape) {
    int shapesize = wshape[0] * wshape[1] * wshape[2] * wshape[3] * wshape[4];
    int Ws = wshape[0];
    int N = wshape[1];
    int C = wshape[4];
    int H = wshape[2];
    int W = wshape[3];
    float* tmp_weightdata = (float*)malloc(shapesize * sizeof(float));
    memcpy(tmp_weightdata, weightdata, shapesize * sizeof(float));
    // TODO: Jasper DWdense WsNHWC -> WsNCHW
    for (int w = 0; w < Ws /*N*/; w++) {
      for (int i = 0; i < N /*N*/; i++) {
        for (int j = 0; j < C /*c*/; j++) {
          for (int k = 0; k < H /*H*/; k++) {
            for (int l = 0; l < W /*W*/; l++) {
              // weightdata[i*C*H*W+j*H*W+k*W+l]=tmp_weightdata[i*C*H*W+k*W*C+l*C+j];
              weightdata[w * N * C * H * W + i * C * H * W + j * H * W + k * W + l] =
                  tmp_weightdata[w * N * C * H * W + i * C * H * W + k * W * C + l * C + j];
            }
          }
        }
      }
    }
    free(tmp_weightdata);
  }

  Expr GetNotQuantizedExpr(Expr expr){
    if (expr->IsInstance<CallNode>()){
      auto op = Downcast<Call>(expr);
      if (const auto* op_node = op->op.as<OpNode>()) { 
      //const auto* op_node = call->op.as<OpNode>();
      std::string op_name = GetRef<Op>(op_node)->name;
      //LOG(INFO) << "op_name: " << op_name;
      if (op_name  == "relay.op.annotation.simulated_quantize" || op_name.substr(0, 10) == "annotation")
        return GetNotQuantizedExpr(op->args[0]);
      }
    }
    return expr;
  }

 protected:
  Expr VisitExpr_(const CallNode* op) override {
    Array<RelayExpr> new_args;
    auto add_op = tvm::relay::Op::Get("add");
    auto dense_op = tvm::relay::Op::Get("nn.dense");
    auto* op_node = op->op.as<tvm::relay::OpNode>();
    const auto op_name = tvm::runtime::GetRef<tvm::relay::Op>(op_node)->name;
    if(op_name == "relay.op.annotation.simulated_quantize" &&
       op->args[0]->IsInstance<tvm::relay::CallNode>()){
      const tvm::relay::CallNode* add_call = op->args[0].as<tvm::relay::CallNode>();
      if (add_call->op == add_op && 
          add_call->args[1]->IsInstance<tvm::relay::ConstantNode>() &&
          add_call->args[0]->IsInstance<tvm::relay::CallNode>()) {
        const tvm::relay::CallNode* dense_call = add_call->args[0].as<tvm::relay::CallNode>();
        if (dense_call->op == dense_op){
          is_quantize = true;
        }
      }
    }
    // do the split and updates the AST
    for (auto arg : op->args) {
      arg = VisitExpr(arg);
      new_args.push_back(arg);
    }

    if(is_quantize){
      if (op_name == "relay.op.annotation.simulated_quantize") {
        if (op->args[0]->IsInstance<tvm::relay::CallNode>() &&
            op->args[1]->IsInstance<tvm::relay::ConstantNode>() &&
            op->args[2]->IsInstance<tvm::relay::ConstantNode>() &&
            op->args[3]->IsInstance<tvm::relay::ConstantNode>()) {
          const tvm::relay::CallNode* add_call = op->args[0].as<tvm::relay::CallNode>();
          if (add_call->op == add_op && add_call->args[1]->IsInstance<tvm::relay::ConstantNode>()) {
            if (add_call->args[0]->IsInstance<tvm::relay::CallNode>()) {
              const tvm::relay::CallNode* dense_call = add_call->args[0].as<tvm::relay::CallNode>();
              if (dense_call->op == dense_op &&
                  dense_call->args[0]->IsInstance<tvm::relay::CallNode>() &&
                  dense_call->args[1]->IsInstance<tvm::relay::ConstantNode>() &&
                  tvm::relay::backend::GetShape(dense_call->args[1]->checked_type()).at(1) > weight_thr) {
                return SplitByC(new_args, op);
              }
            }
          }
        }
      }
    }
    else{
      if (op_name == "add") {
        if (op->args[1]->IsInstance<tvm::relay::ConstantNode>()) {
          if (op->args[0]->IsInstance<tvm::relay::CallNode>()) {
            const tvm::relay::CallNode* dense_call = op->args[0].as<tvm::relay::CallNode>();
            if (dense_call->op == dense_op &&
                dense_call->args[0]->IsInstance<tvm::relay::CallNode>() &&
                dense_call->args[1]->IsInstance<tvm::relay::ConstantNode>() &&
                tvm::relay::backend::GetShape(dense_call->args[1]->checked_type()).at(1) > weight_thr) {
              return SplitByC(new_args, op);
            }
          }
        }
      }
    }
    ///lay out yuanyue 20220623
    if (op_name == "nn.conv2d"){
      is_layout_ = false;
    }    
    if (op_name == "nn.dense" && is_layout_) {
      if (op->args[1]->IsInstance<tvm::relay::ConstantNode>()) {
        Expr expr=GetNotQuantizedExpr(op->args[0]);
        if (expr->IsInstance<CallNode>()){
          const tvm::relay::CallNode* arg_call = expr.as<tvm::relay::CallNode>();
          if (const auto* arg_op_node = arg_call->op.as<OpNode>()) { 
          //const auto* op_node = call->op.as<OpNode>();
          const auto arg_op_name = GetRef<Op>(arg_op_node)->name;
          //LOG(INFO) << "arg_op_name: " << arg_op_name;
          if (arg_op_name == "reshape"){  
              return Layout_weights(op, arg_call);
            }
          }
        }
      }
    }
    return Call(op->op, new_args, op->attrs, op->type_args);
  }
};

class Conv2dExpr : public ExprMutator {
  public:
    int weight_thr = 7168;
    bool is_quantize = false;
    
    Conv2dExpr(){}

    Call SplitByC(tvm::Array<Expr> new_args, const CallNode* op){
      auto add_op = tvm::relay::Op::Get("add");
      auto conv2d_op = tvm::relay::Op::Get("nn.conv2d");
      const tvm::relay::CallNode* add_call;
      const tvm::relay::CallNode* conv2d_call;
      const tvm::relay::CallNode* conv2d_call2;
	  
      const tvm::relay::CallNode* root_call = new_args[0].as<tvm::relay::CallNode>();
      //const tvm::relay::CallNode* arg_call = op->args[0].as<tvm::relay::CallNode>();
	    
      //add constant
      Constant Add_Constant;
      if(root_call->op == add_op){
        add_call = new_args[0].as<tvm::relay::CallNode>();
        conv2d_call = add_call->args[0].as<tvm::relay::CallNode>();
        if (add_call->args[1]->IsInstance<ConstantNode>()) {
          Add_Constant = Downcast<Constant>(add_call->args[1]);
        } else {
          LOG(FATAL) << "arg error" << AsText(add_call->args[1], false);
        }
      }
      else if(root_call->op == conv2d_op){
        conv2d_call = op->args[0].as<tvm::relay::CallNode>();
        if (new_args[1]->IsInstance<ConstantNode>()) {
          Add_Constant = Downcast<Constant>(new_args[1]);
        } else {
          LOG(FATAL) << "arg error" << AsText(new_args[1], false);
        }
      }
      if (!conv2d_call)
        LOG(FATAL) << "arg error" << AsText(new_args[0], false);

      conv2d_call2 = op->args[0].as<tvm::relay::CallNode>();
      auto ishape = GetShape(conv2d_call2->args[0]->checked_type());    
      LOG(INFO) << "ishape:" << ishape[0] << "," << ishape[1] << "," << ishape[2] << "," << ishape[3];
	  
      auto wshape = GetShape(conv2d_call->args[1]->checked_type());    
      LOG(INFO) << "wshape:" << wshape[0] << "," << wshape[1] << "," << wshape[2] << "," << wshape[3];

      const auto* conv2d_attr = conv2d_call->attrs.as<Conv2DAttrs>();
      ICHECK(conv2d_attr);
      //LOG(INFO) << AsText(conv2d_call->args[0], false);
	  
      //unsigned int quantize_args_size = 4;
      unsigned int quantize_args_size2 = 4;

      Constant scale_in_[quantize_args_size2-1];
      Constant scale_out_[quantize_args_size2-1];
      tvm::relay::Expr quantize_in;
      if(root_call->op == add_op){
        auto quantize_call = conv2d_call->args[0].as<tvm::relay::CallNode>();
        auto quantize_call2 = quantize_call->args[0].as<tvm::relay::CallNode>();
        auto *padcallnode = conv2d_call->args[0].as<CallNode>();
        const auto* op_node = padcallnode->op.as<OpNode>();
        const auto op_name = GetRef<Op>(op_node)->name;
        //LOG(INFO) << AsText(op_name, false);
        if (op_name=="annotation.stop_fusion"){
          quantize_in = quantize_call2->args[0];
        }
        else if (op_name=="annotation.cast_hint"){
          quantize_in = quantize_call->args[0];
        }
        else{
          quantize_in = conv2d_call->args[0];
        }
        tvm::relay::Expr quantize_out = tvm::runtime::GetRef<Expr>(op);

        const tvm::relay::CallNode* arg_call_quantize_in = quantize_in.as<tvm::relay::CallNode>();
        for(unsigned int i=1; i<quantize_args_size2; i++){
          Constant quantize_constant = Downcast<Constant>(arg_call_quantize_in->args[i]);
          auto quantize_data = tvm::runtime::NDArray::Empty({}, DLDataType{kDLFloat, 32, 1}, {kDLCPU, 0});
          void* param_quantize;
          size_t size_quantize = 1* sizeof(float);
          param_quantize = (void*)(new char[size_quantize]);
          memcpy(param_quantize, static_cast<char*>(quantize_constant->data->data), size_quantize);
          quantize_data.tvm::runtime::NDArray::CopyFromBytes(param_quantize, size_quantize);
          scale_in_[i-1] =  relay::Constant(quantize_data);
        }

        const tvm::relay::CallNode* arg_call_quantize_out = quantize_out.as<tvm::relay::CallNode>();
        for(unsigned int i=1; i<quantize_args_size2; i++){
          Constant quantize_constant = Downcast<Constant>(arg_call_quantize_out->args[i]);
          auto quantize_data = tvm::runtime::NDArray::Empty({}, DLDataType{kDLFloat, 32, 1}, {kDLCPU, 0});
          void* param_quantize;
          size_t size_quantize = 1* sizeof(float);
          param_quantize = (void*)(new char[size_quantize]);
          memcpy(param_quantize, static_cast<char*>(quantize_constant->data->data), size_quantize);
          quantize_data.tvm::runtime::NDArray::CopyFromBytes(param_quantize, size_quantize);
          scale_out_[i-1] =  relay::Constant(quantize_data);
        }
      }
	  
      int N = 0;
      int H = 0;
      int W = 0;
      int C = 0;
      int numOutputs = 0;
      std::string kernel_layout = conv2d_attr->kernel_layout;
      if(kernel_layout=="HWOI"){
        N = wshape[2];
        H = wshape[0];
        W = wshape[1];
        C = wshape[3];
        numOutputs = wshape[2];
      }
      if(kernel_layout=="HWIO"){
        N = wshape[3];
        H = wshape[0];
        W = wshape[1];
        C = wshape[2];
        numOutputs = wshape[3];
      }
      
      LOG(INFO) << "numOutputs "<< numOutputs;
      LOG(INFO) << "kernel_layout:" << conv2d_attr->kernel_layout;
	  
      int numGroups  = conv2d_attr->groups;
      LOG(INFO) << "numGroups:" << numGroups;
  
      std::string data_layout = conv2d_attr->data_layout;
      LOG(INFO) << data_layout;
      std::string out_layout = data_layout;
		
      int weightsize = wshape[0]*wshape[1]*wshape[2];
      LOG(INFO) << "weightsize：" << weightsize;
	  
      int sp_n_part = int(weightsize / weight_thr) + 1;
      if(sp_n_part <= 2){
        sp_n_part = 2;
      }else if (sp_n_part > 2 && sp_n_part <= 4) {
        sp_n_part = 4;
      }else if (sp_n_part > 4 && sp_n_part <= 8) {
        sp_n_part = 8;
      }else if (sp_n_part > 8 && sp_n_part <= 16) {
        sp_n_part = 16;
      }else if (sp_n_part > 16 && sp_n_part <= 32) {
        sp_n_part = 32;
      }else{
        LOG(FATAL) << "split not divisible, please check";
      }
      LOG(INFO) << "sp_n_part：" << sp_n_part;
      //conv2d weight
      Constant weight_const;
      size_t weight_size_get = 1;
      if(conv2d_call->args[1].as<ConstantNode>())
      {
        LOG(INFO) << "Conv2d WEIGHT AS ConstantNode";
        weight_const = Downcast<Constant>(conv2d_call->args[1]);
        LOG(INFO) << "Conv2d WEIGHT AS ConstantNode data:" << *((float*)weight_const->data->data);
      }
	
      for (size_t i = 0; i < wshape.size(); i++) {
        weight_size_get *= static_cast<size_t>(wshape[i]);
      }
  
      Constant weight_split[sp_n_part];
      size_t weight_size = weight_size_get;
      size_t weight_size_sp = weight_size / sp_n_part;
      size_t weight_size_sp_bit = (sizeof(float) * weight_size) / sp_n_part;
      //LOG(INFO) << "weight_size " << weight_size;
      //LOG(INFO) << "weight_size_sp " << weight_size_sp;
      //LOG(INFO) << "weight_size_sp_bit " << weight_size_sp_bit;
      
      void* param_weights[sp_n_part];
      int size[sp_n_part][H][W];
      int from_offset[sp_n_part][H][W];
      for(int i = 0; i < sp_n_part; i++){
        param_weights[i]=(void *)(new char[weight_size_sp_bit]);
      }
	  
      int in_channel = C / sp_n_part;
      LOG(INFO) << "in_channel " << in_channel;
      for(int i = 0; i < sp_n_part; i++){
        for(int j = 0; j < H; j++){
          for(int k = 0; k < W; k++){
            size[i][j][k] = (C / sp_n_part) * N * sizeof(float);
            from_offset[i][j][k] = i * size[i][j][k]  +  k * C * N * sizeof(float) + j * W * C * N * sizeof(float);	
            memcpy(static_cast<char*>(param_weights[i]) + k * size[i][j][k] + j * W * (C / sp_n_part) * N * sizeof(float), 
                   static_cast<char*>(weight_const->data->data) + from_offset[i][j][k], size[i][j][k]);
          }
        }
      }
	  
      for(int i = 0; i < sp_n_part; i++){
        auto weight_data = tvm::runtime::NDArray::Empty({H, W, C / sp_n_part, N}, DLDataType{kDLFloat, 32, 1},{kDLCPU, 0});
        weight_data.tvm::runtime::NDArray::CopyFromBytes(param_weights[i], weight_size_sp*sizeof(float));
        weight_split[i] = relay::Constant(weight_data);
      }
	  
      int64_t spTh = C / sp_n_part;
      tvm::RelayExpr slice[sp_n_part];
      tvm::RelayExpr conv2d[sp_n_part];
      tvm::RelayExpr conv2d_exp[sp_n_part];
      Array<Expr> tuple;
	  
      static const Op& quantize_op = relay::Op::Get("relay.op.annotation.simulated_quantize");
      auto quantize_attrs = make_object<tvm::relay::qnn::SimulatedQuantizeAttrs>();
      quantize_attrs->axis = -1;
	
      for (int i=0; i<sp_n_part; i++) {
        //auto split_data = TupleGetItem(splitted, index++);
        //LOG(INFO) << AsText(split_data, false);
        //splitted conv2d
        std::string opConv = "nn.conv2d";
        auto attrs = make_object<Conv2DAttrs>();
        attrs->strides = conv2d_attr->strides;
        attrs->padding = conv2d_attr->padding;
    	attrs->dilation = conv2d_attr->dilation;
        attrs->groups = numGroups;
        attrs->channels = numOutputs;
        attrs->kernel_size = conv2d_attr->kernel_size;
        attrs->data_layout = data_layout;
        attrs->kernel_layout = kernel_layout;
        attrs->out_layout = out_layout;
        attrs->out_dtype = DataType::Float(32);
        const Op& opC = Op::Get(opConv);

        Array<Integer> begin;
        Array<Integer> end;
        Array<Integer> strides;

        begin.push_back(0);
        begin.push_back(0);
        begin.push_back(0);
        begin.push_back(0 + i * spTh);

        end.push_back(ishape[0]);
        end.push_back(ishape[1]);
        end.push_back(ishape[2]);
        end.push_back(((i + 1)*spTh));
	
        strides.push_back(1);
        strides.push_back(1);
        strides.push_back(1);
        strides.push_back(1);

        slice[i] = MakeStridedSlice(conv2d_call->args[0], begin, end, strides, "end");
	  
        if(is_quantize){
          auto splitted_quantize = relay::Call(quantize_op, {slice[i], scale_in_[0], scale_in_[1], scale_in_[2]}, Attrs(quantize_attrs), {});
          conv2d[i] = tvm::relay::Call(opC, {splitted_quantize, weight_split[i]}, Attrs(attrs), {});
        }else{
          conv2d[i] = tvm::relay::Call(opC, {slice[i], weight_split[i]}, Attrs(attrs), {});
        }
		
        //"add" after conv2d needed
        auto add_op = relay::Op::Get("add");
        auto c_data = tvm::runtime::NDArray::Empty({N}, {kDLFloat, 32, 1}, {kDLCPU, 0});
        auto c1 = relay::Constant(c_data);
        if(i < sp_n_part - 1){
          conv2d[i] = relay::Call(add_op, {conv2d[i], c1});
          if(is_quantize){
            conv2d[i] = tvm::relay::Call(quantize_op, {conv2d[i], scale_out_[0], scale_out_[1], scale_out_[2]}, Attrs(quantize_attrs), {});
          }
        }
        if(i==sp_n_part-1){
          conv2d[i] = relay::Call(add_op, {conv2d[i], Add_Constant});
          if(is_quantize){
            conv2d[i] = tvm::relay::Call(quantize_op, {conv2d[i], scale_out_[0], scale_out_[1], scale_out_[2]}, Attrs(quantize_attrs), {});
          }
        }
        //LOG(INFO) << AsText(conv2d[i], false); 
        
        // expand the shape
        //conv2d_exp[i] = MakeExpandDims(conv2d[i], 0, 1);
        //tuple.push_back(conv2d_exp[i]);
        tuple.push_back(conv2d[i]);
        //LOG(INFO) << AsText(tuple, false);
      }
	  
      //concatenate and sum
      auto new_op = MakeConcatenate(Tuple(tuple), 0);
      //LOG(INFO) <<"new_op "<< AsText(new_op, false);

      Array<Integer> axis;
      axis.push_back(0);
      auto sp_conv_call_merge = MakeReduce(new_op, axis, 1, 0, "sum");
	  
      if(is_quantize){
        sp_conv_call_merge = relay::Call(quantize_op, {sp_conv_call_merge, scale_out_[0], scale_out_[1], scale_out_[2]}, Attrs(quantize_attrs), {});
      }
      auto sp_conv_call = Downcast<Call>(sp_conv_call_merge);
      LOG(INFO) << "splitted after split merge " << AsText(sp_conv_call_merge, false);

      return sp_conv_call; 
    }

  protected:
    Expr VisitExpr_(const CallNode* op) override {
      Array<RelayExpr> new_args;
      auto add_op = tvm::relay::Op::Get("add");
      auto conv2d_op = tvm::relay::Op::Get("nn.conv2d");
      auto *op_node = op->op.as<tvm::relay::OpNode>();
      const auto op_name = tvm::runtime::GetRef<tvm::relay::Op>(op_node)->name;
	
      if(op_name == "relay.op.annotation.simulated_quantize" && op->args[0]->IsInstance<tvm::relay::CallNode>()){
        const tvm::relay::CallNode* arg_call = op->args[0].as<tvm::relay::CallNode>();
        if (arg_call->op == add_op && 
            arg_call->args[0]->IsInstance<tvm::relay::CallNode>() &&
            arg_call->args[1]->IsInstance<tvm::relay::ConstantNode>()) {
          const tvm::relay::CallNode* conv2d_call = arg_call->args[0].as<tvm::relay::CallNode>();
          if (conv2d_call->op == conv2d_op){
            is_quantize = true;
          }
        }
      }
	  
      for (auto arg : op->args) {
        arg = VisitExpr(arg);
        new_args.push_back(arg);
      }
	  
      if(is_quantize){
        if (op_name == "relay.op.annotation.simulated_quantize") {
          if (op->args[0]->IsInstance<tvm::relay::CallNode>() &&
              op->args[1]->IsInstance<tvm::relay::ConstantNode>() &&
              op->args[2]->IsInstance<tvm::relay::ConstantNode>() &&
              op->args[3]->IsInstance<tvm::relay::ConstantNode>()) {
            const tvm::relay::CallNode* add_call = op->args[0].as<tvm::relay::CallNode>();
            if (add_call->op == add_op && add_call->args[1]->IsInstance<tvm::relay::ConstantNode>()) {
              if (add_call->args[0]->IsInstance<tvm::relay::CallNode>()) {
                const tvm::relay::CallNode* conv2d_call = add_call->args[0].as<tvm::relay::CallNode>();
                auto ishape = GetShape(conv2d_call->args[0]->checked_type());
                auto wshape = GetShape(conv2d_call->args[1]->checked_type());
                LOG(INFO)<<"input ishape:"<<ishape[0]<<","<<ishape[1]<<","<<ishape[2]<<","<<ishape[3];
                LOG(INFO) << "wshape:" << wshape[0] << "," << wshape[1] << "," << wshape[2] << "," << wshape[3];
                int weightsize1 = wshape[0] * wshape[1] * wshape[2];
                if (conv2d_call->op == conv2d_op &&
                    conv2d_call->args[0]->IsInstance<tvm::relay::CallNode>() &&
                    conv2d_call->args[1]->IsInstance<tvm::relay::ConstantNode>() &&
                    weightsize1 > weight_thr) {
                  return SplitByC(new_args, op);
                }
              }
            }
          }
        }
      }
      else{
        if (op_name == "add") {
          if (op->args[1]->IsInstance<tvm::relay::ConstantNode>()){
            if (op->args[0]->IsInstance<tvm::relay::CallNode>()){
              const tvm::relay::CallNode* arg_call = op->args[0].as<tvm::relay::CallNode>();
              auto *arg_op_node = arg_call->op.as<tvm::relay::OpNode>();
              const auto arg_op_name = tvm::runtime::GetRef<tvm::relay::Op>(arg_op_node)->name;
              LOG(INFO) << AsText(arg_call->args[1], false);
              auto ishape = GetShape(arg_call->args[0]->checked_type());
              auto wshape = GetShape(arg_call->args[1]->checked_type());
              LOG(INFO)<<"input ishape:"<<ishape[0]<<","<<ishape[1]<<","<<ishape[2]<<","<<ishape[3];
              LOG(INFO) << "wshape:" << wshape[0] << "," << wshape[1] << "," << wshape[2] << "," << wshape[3];
              int weightsize2 = wshape[0] * wshape[1] * wshape[2];
              if (arg_call->op == conv2d_op &&
                  arg_call->args[0]->IsInstance<tvm::relay::CallNode>() &&
                  arg_call->args[1]->IsInstance<tvm::relay::ConstantNode>() &&
                  weightsize2 > weight_thr){
               return SplitByC(new_args, op);
             }
           }
         }
       }
    }
    return Call(op->op, new_args, op->attrs, op->type_args);
  }
};

class GenCalibdata : public ExprVisitor {
 public:
  using Expr = tvm::RelayExpr;
  using CallNode = tvm::relay::CallNode;
  using TupleNode = tvm::relay::TupleNode;
  using OpNode = tvm::OpNode;
  using Op = tvm::Op;

  GenCalibdata(std::vector<std::vector<float>> scale_vec, std::string compiler) { inscle_vec = scale_vec; compiler_ = compiler;}

  void VisitExpr_(const TupleNode* op) final {
    for (auto field : op->fields) {
      VisitExpr(field);
    }
  }

  void VisitExpr_(const CallNode* call) final {
    for (auto arg : call->args) {
      if (arg->IsInstance<CallNode>()) {
        VisitExpr(arg);
      }
    }
    const auto* op_node = call->op.as<OpNode>();
    const auto op_name = GetRef<Op>(op_node)->name;

    //LOG(INFO) << "test scale op_name: " << op_name;
    if (op_name == "relay.op.annotation.simulated_quantize") {
      float scale;
      if (call->args[0]->IsInstance<CallNode>()) {
        const CallNode* arg_call = call->args[0].as<CallNode>();
        auto* arg_op_node = arg_call->op.as<OpNode>();
        const auto arg_op_name = tvm::runtime::GetRef<Op>(arg_op_node)->name;
        // if (arg_op_name == "mean" || arg_op_name == "nn.max_pool2d" || arg_op_name == "nn.avg_pool2d") {  // Quantitative penetration
        //   if (arg_call->args[0]->IsInstance<CallNode>()) {
        //     const CallNode* pre_call = arg_call->args[0].as<CallNode>();
        //     auto* pre_op_node = pre_call->op.as<OpNode>();
        //     const auto pre_op_name = tvm::runtime::GetRef<Op>(pre_op_node)->name;
        //     if (pre_op_name == "relay.op.annotation.simulated_quantize") {
        //       auto argconst = Downcast<Constant>(pre_call->args[1]);
        //       scale = *((float*)argconst->data->data);
        //     } else if (pre_op_name == "annotation.cast_hint") {
        //       const CallNode* pre_arg_call = pre_call->args[0].as<CallNode>();
        //       auto argconst = Downcast<Constant>(pre_arg_call->args[1]);
        //       scale = *((float*)argconst->data->data);
        //     } else {
        //       return;
        //     }
        //   } else if (arg_call->args[0]->IsInstance<VarNode>()) {
        //     scale = inscle_vec[0][0];  // Jasper TODO
        //   }
        // } 
        if (arg_op_name == "nn.pad" || (compiler_ == "dla" && arg_op_name == "reshape")) {
          return;
        } else {
          auto argconst = Downcast<Constant>(call->args[1]);
          scale = *((float*)argconst->data->data);
        }
      } else {
        auto argconst = Downcast<Constant>(call->args[1]);
        scale = *((float*)argconst->data->data);
      }

      calibdata_.push_back(scale);
      //LOG(INFO) << "pass relay.op.annotation.simulated_quantize!";
      //LOG(INFO) << "scale: " << scale;

      op_out_scale_map_.insert(std::pair<Expr, float>(call->args[0], scale));
    }
  }

  std::vector<float> GetCalibdata() { return calibdata_; }
  std::map<Expr, float> GetOpOutScaleMap() { return op_out_scale_map_; }

  std::vector<float> GetSubfuncOutScaleMap(Expr body) {
    std::vector<float> subfunc_out_scale;
    std::vector<Expr> subfunc_out_Expr;
    if (body->IsInstance<CallNode>()) {
      const CallNode* call = body.as<CallNode>();
      auto* op_node = call->op.as<tvm::OpNode>();
      const auto op_name = tvm::runtime::GetRef<Op>(op_node)->name;
      // if (op_name == "mean" || op_name == "nn.max_pool2d" || op_name == "nn.avg_pool2d"){
      //   if (op_name == "relay.op.annotation.simulated_quantize") { 
      //   auto argconst = Downcast<Constant>(call->args[1]);
      //   float scale = *((float*)argconst->data->data);
      //   subfunc_out_scale.push_back(scale);
      //   } else if (op_name == "annotation.cast_hint") {
      //     const CallNode* arg_call = call->args[0].as<CallNode>();
      //     auto argconst = Downcast<Constant>(arg_call->args[1]);
      //     float scale = *((float*)argconst->data->data);
      //     subfunc_out_scale.push_back(scale);
      //     }
      //   } 
      if (op_name == "relay.op.annotation.simulated_quantize") { 
        auto argconst = Downcast<Constant>(call->args[1]);
        float scale = *((float*)argconst->data->data);
        subfunc_out_scale.push_back(scale);
      } else if (op_name == "annotation.cast_hint") {
        const CallNode* arg_call = call->args[0].as<CallNode>();
        auto argconst = Downcast<Constant>(arg_call->args[1]);
        float scale = *((float*)argconst->data->data);
        subfunc_out_scale.push_back(scale);
      }
      // For extended conv op, we need to add a featuretoweight op. deal with separately
      else if (call->op == Op::Get("featuretoweight")) {
        // we assume its arg is relay.op.annotation.simulated_quantize. TODO: what if this is not the case?
	    const CallNode* arg_call = call->args[0].as<CallNode>();
	    auto argconst = Downcast<Constant>(arg_call->args[1]);
	    float scale = *((float*)argconst->data->data);
        subfunc_out_scale.push_back(scale);
      }
    } else if (body->IsInstance<TupleNode>()) {
      const TupleNode* call = body.as<TupleNode>();
      for (auto field : call->fields) {
        if (field->IsInstance<CallNode>()) {
          const CallNode* node = (static_cast<const CallNode *>(field.get()));
          const auto op_node = node->op.as<OpNode>();
          std::string op_name = tvm::runtime::GetRef<Op>(op_node)->name;
          if (op_name == "relay.op.annotation.simulated_quantize") { 
            auto argconst = Downcast<Constant>(node->args[1]);
            float scale = *((float*)argconst->data->data);
            subfunc_out_scale.push_back(scale);
          } else if (op_name == "annotation.cast_hint") {
            const CallNode* arg_call = node->args[0].as<CallNode>();
            auto argconst = Downcast<Constant>(arg_call->args[1]);
            float scale = *((float*)argconst->data->data);
            subfunc_out_scale.push_back(scale);
          }
          // For extended conv op, we need to add a featuretoweight op. deal with separately
          //else if (field->op == Op::Get("featuretoweight")) {}
          else if (op_name == "featuretoweight") {
            // we assume its arg is relay.op.annotation.simulated_quantize. TODO: what if this is not the case?
            const CallNode* arg_call = node->args[0].as<CallNode>();
            auto argconst = Downcast<Constant>(arg_call->args[1]);
            float scale = *((float*)argconst->data->data);
            subfunc_out_scale.push_back(scale);
          }
          else {
            subfunc_out_scale.push_back(-1.0f);
          }
          subfunc_out_Expr.push_back(field);
        }
      }
    }

    // fixed (%8,%7): %8 is simulated_quantize, %7 is input of %8
    if (body->IsInstance<TupleNode>()) {
      const TupleNode* call = body.as<TupleNode>();
      size_t i_ = 0;
      for (auto field : call->fields) {
        if (field->IsInstance<CallNode>()) {
          const CallNode* node = (static_cast<const CallNode *>(field.get()));
          for (auto arg: node->args) {
            if (std::find(subfunc_out_Expr.begin(), subfunc_out_Expr.end(), arg) != subfunc_out_Expr.end()) {
              size_t index = std::find(subfunc_out_Expr.begin(), subfunc_out_Expr.end(), arg) - subfunc_out_Expr.begin();
              subfunc_out_scale[index] = subfunc_out_scale[i_];
            }
          }
        }
        i_++;
      }
    }

    return subfunc_out_scale;
  }


 protected:
  std::vector<float> calibdata_;
  std::map<Expr, float> op_out_scale_map_;
  std::vector<std::vector<float>> inscle_vec;
  std::string compiler_;
};

// used for traversal fused function
class TraversalModule : public ExprVisitor {
 public:
  // yuanyue 20220622 plan memory
  TraversalModule(Map<Expr, Array<IntegerArray>> storage_device_map,  Map<Expr, Array<IntegerArray>> aid_dtype_map, std::map<int, size_t> temporary_data_storage,
          std::map<int, size_t> temporary_data_offset, size_t total_memory_used) {
    storage_device_map_ = storage_device_map;
    aid_dtype_map_ = aid_dtype_map;
    temporary_data_storage_ = temporary_data_storage;
    temporary_data_offset_ = temporary_data_offset;
    total_memory_used_ = total_memory_used;
    //data_memory_used_ = 0;
    loadable_order_ = 1;
    riscv_order_ = -1;
  }

  runtime::Module AIPUModuleCreate() {
    auto n = make_object<tvm::runtime::contrib::AIPUModuleNode>(loadable_, riscv_code_, riscv_addr_list_, riscv_wt_list_, execution_order_,
                                                                io_offset_, input_, output_, total_memory_used_);

    return runtime::Module(n);
  }

  size_t divRoundUp(size_t size, size_t word_size) { return (size + word_size - 1) / word_size; }

  void debug() {
    std::stringstream log_stream_tmp;
    LOG(INFO)<< "******************************** temporary_data_offset_ *******************************";
    for (auto it : temporary_data_offset_)
      LOG(INFO) << "storage_id " << it.first << " offset " << it.second;

    LOG(INFO)<< "******************************** Fused_funtion_offsets ********************************";
    log_stream_tmp << std::endl;
    for (uint32_t i = 0; i < io_offset_.size(); i++) {
        log_stream_tmp << i << " -- Fused_funtion_offsets {" << std::endl << "\tinput_offsets:";
        for (uint32_t j = 0; j < io_offset_[i].input_offsets.size(); j++) {
            log_stream_tmp << " " << io_offset_[i].input_offsets[j];
        }
        log_stream_tmp << "\n\toutput_offsets:";
        for (uint32_t j = 0; j < io_offset_[i].output_offsets.size(); j++) {
            log_stream_tmp << " " << io_offset_[i].output_offsets[j];
        }
        log_stream_tmp << "\n\tinput_size:";
        for (uint32_t j = 0; j < io_offset_[i].input_size.size(); j++) {
            log_stream_tmp << " " << io_offset_[i].input_size[j];
        }
        log_stream_tmp << "\n\toutput_size:";
        for (uint32_t j = 0; j < io_offset_[i].output_size.size(); j++) {
            log_stream_tmp << " " << io_offset_[i].output_size[j];
        }
        log_stream_tmp << "\n\t}\n";
    }
    LOG(INFO)<< log_stream_tmp.str();
    log_stream_tmp.clear();

    LOG(INFO)<< "********************************* Net_io **********************************************";
    for (auto i = input_.begin(); i != input_.end(); i++) {
        LOG(INFO) << i->first << ", address: " << i->second.first << ", size is " 
            << i->second.second << "\n"; 
    }

    for (auto i = output_.begin(); i != output_.end(); i++) {
        LOG(INFO) << i->first << ", address: "  << i->second.first << ", size is " 
             << i->second.second << "\n"; 
    }

    LOG(INFO)<< "******************************** execute_order ****************************************";
    log_stream_tmp << std::endl;
    log_stream_tmp << "\tGot execute_order, size is " << execution_order_.size() << std::endl;
    for (uint32_t i = 0; i < execution_order_.size(); i++) {
        log_stream_tmp << "    " << execution_order_[i];
    }
    log_stream_tmp << std::endl;

    log_stream_tmp << "\tGot riscv_addr_list, size is "
        << riscv_addr_list_.size() << std::endl;
    for (uint32_t i = 0; i < riscv_addr_list_.size(); i++) {
        log_stream_tmp << "The " << i << "th task address list is [wt_index, offset]:\n\t{";
        for (uint32_t j = 0; j < riscv_addr_list_[i].size(); j++) {
            log_stream_tmp << "[" << riscv_addr_list_[i][j].first << ", " << riscv_addr_list_[i][j].second << "], ";
        }
        log_stream_tmp << "}" << std::endl;
    }

    log_stream_tmp << "\tGot riscv_wt_list, size is " << riscv_wt_list_.size() << std::endl
        << "task wt list is [size, address]:\n\t{";
    for (uint32_t i = 0; i < riscv_wt_list_.size(); i++) {
        log_stream_tmp << "[" << riscv_wt_list_[i].first << ", "<< riscv_wt_list_[i].second << "], ";
    }
    log_stream_tmp << "}" << std::endl;
    LOG(INFO) << log_stream_tmp.str();
    log_stream_tmp.clear();

    LOG(INFO)<< "******************************** riscv_code_ ******************************************";
    for (size_t s=0; s<riscv_code_.size();s++ ){
      LOG(INFO) << std::endl <<  "riscv code " << s;
      std::vector<Cpu_param *> test_riscv_code = riscv_code_[s];
      for (size_t i =0 ; i < test_riscv_code.size() ; i++ ){
        u_int32_t op_type = test_riscv_code[i]->cpu_operation.common_only_op.common.op_type;
        if (op_type == tvm::runtime::contrib::SOFTMAX ){
          LOG(INFO) << "************SOFTMAX************";
          CodegenAIPU().output_info(test_riscv_code[i]->cpu_operation.common_only_op.common,
                  test_riscv_code[i]->cpu_operation_buffer.common_only_op_buffers.src_data,
                  test_riscv_code[i]->cpu_operation_buffer.common_only_op_buffers.dst_data);

        }else if (op_type == tvm::runtime::contrib::CONCAT){
          LOG(INFO) << "************CONCAT************";
          CodegenAIPU().output_info(test_riscv_code[i]->cpu_operation.concat_op.common,
                  test_riscv_code[i]->cpu_operation_buffer.concat_buffers.src_data,
                  test_riscv_code[i]->cpu_operation_buffer.concat_buffers.dst_data);
          LOG(INFO) << "concat_param axis: " << test_riscv_code[i]->cpu_operation.concat_op.axis;
          LOG(INFO) << "common.input_num: " << test_riscv_code[i]->cpu_operation.concat_op.common.input_num;
          for (size_t j = 0; j <  test_riscv_code[i]->cpu_operation.concat_op.common.input_num; j++)
            CodegenAIPU().output_info_other(test_riscv_code[i]->cpu_operation.concat_op.src_data[j]);

        }else if (op_type == tvm::runtime::contrib::SPLIT){
          LOG(INFO) << "************SPLIT************";
          CodegenAIPU().output_info(test_riscv_code[i]->cpu_operation.split_op.common,
                  test_riscv_code[i]->cpu_operation_buffer.split_buffers.src_data,
                  test_riscv_code[i]->cpu_operation_buffer.split_buffers.dst_data);
          LOG(INFO) << "split_param axis: " << test_riscv_code[i]->cpu_operation.split_op.axis;
          LOG(INFO) << "split_param indices[INPUT_MAX]: " ;
          for (size_t j=0;j<INPUT_MAX;j++){
            LOG(INFO) << test_riscv_code[i]->cpu_operation.split_op.indices[j];
          }
          LOG(INFO) << "common.output_num: " << test_riscv_code[i]->cpu_operation.split_op.common.output_num;
          for (size_t j = 0; j <  test_riscv_code[i]->cpu_operation.split_op.common.output_num; j++)
            CodegenAIPU().output_info_other(test_riscv_code[i]->cpu_operation.split_op.dst_data[j]);

        }else if (op_type == tvm::runtime::contrib::SUM){
          LOG(INFO) << "************SUM************";
          CodegenAIPU().output_info(test_riscv_code[i]->cpu_operation.reduce_op.common,
                  test_riscv_code[i]->cpu_operation_buffer.reduce_buffers.src_data,
                  test_riscv_code[i]->cpu_operation_buffer.reduce_buffers.dst_data);
          int32_t * tmp = test_riscv_code[i]->cpu_operation.reduce_op.axis;
          LOG(INFO) << "SUM axis: " << tmp[0] <<", "<< tmp[1] <<", "<< tmp[2]<<", " << tmp[3] ;
          LOG(INFO) << "SUM keepdims: " << test_riscv_code[i]->cpu_operation.reduce_op.keepdims ;
          LOG(INFO) << "SUM exclude: " << test_riscv_code[i]->cpu_operation.reduce_op.exclude ;

        }else if (op_type == tvm::runtime::contrib::EXPAND_DIMS){
          LOG(INFO) << "************EXPAND_DIMS************";
          CodegenAIPU().output_info(test_riscv_code[i]->cpu_operation.expand_dims_op.common,
                  test_riscv_code[i]->cpu_operation_buffer.expand_dims_buffers.src_data,
                  test_riscv_code[i]->cpu_operation_buffer.expand_dims_buffers.dst_data);
          LOG(INFO) << "EXPAND_DIMS axis: " << test_riscv_code[i]->cpu_operation.expand_dims_op.axis ;
          LOG(INFO) << "EXPAND_DIMS num_newaxis: " << test_riscv_code[i]->cpu_operation.expand_dims_op.num_newaxis ;

        }else if (op_type == tvm::runtime::contrib::RESHAPE){
          LOG(INFO) << "************RESHAPE************";
          CodegenAIPU().output_info(test_riscv_code[i]->cpu_operation.common_only_op.common,
                  test_riscv_code[i]->cpu_operation_buffer.common_only_op_buffers.src_data,
                  test_riscv_code[i]->cpu_operation_buffer.common_only_op_buffers.dst_data);
        }
        else if (op_type == tvm::runtime::contrib::ADD){
          LOG(INFO) << "************ADD************";
          CodegenAIPU().output_info(test_riscv_code[i]->cpu_operation.with_weight_op.common,
                  test_riscv_code[i]->cpu_operation_buffer.with_weight_op_buffers.src_data,
                  test_riscv_code[i]->cpu_operation_buffer.with_weight_op_buffers.dst_data);
          LOG(INFO) << "with_weight_op_param weight_ioscale: " << test_riscv_code[i]->cpu_operation.with_weight_op.weight_ioscale ;
          LOG(INFO) << "with_weight_op_param sub_op_type: " << test_riscv_code[i]->cpu_operation.with_weight_op.sub_op_type ;
          CodegenAIPU().output_info_other(test_riscv_code[i]->cpu_operation.with_weight_op.weight);
        }
        else if (op_type == tvm::runtime::contrib::DENSE){
          LOG(INFO) << "************DENSE************";
          CodegenAIPU().output_info(test_riscv_code[i]->cpu_operation.with_weight_op.common,
                  test_riscv_code[i]->cpu_operation_buffer.with_weight_op_buffers.src_data,
                  test_riscv_code[i]->cpu_operation_buffer.with_weight_op_buffers.dst_data);
          LOG(INFO) << "with_weight_op_param weight_ioscale: " << test_riscv_code[i]->cpu_operation.with_weight_op.weight_ioscale ;
          LOG(INFO) << "with_weight_op_param sub_op_type: " << test_riscv_code[i]->cpu_operation.with_weight_op.sub_op_type ;
          CodegenAIPU().output_info_other(test_riscv_code[i]->cpu_operation.with_weight_op.weight);
        }
        else{
          LOG(INFO) << "!!!!!!!!!!!!!!!!!!!! op_type: " << op_type;
        }
      }
    }
  }
  /*
    void debug() {
      for (size_t i = 0; i < loadable_.size(); i++) {
        auto pa = loadable_[i];
        LOG(INFO) << "loadable_ first: " << pa.first;
      }
      for (size_t i = 0; i < riscv_code_.size(); i++) {
        auto ri = riscv_code_[i];
        LOG(INFO) << "riscv_code_: " << ri.size();
      }
      for (size_t i = 0; i < execution_order_.size(); i++) {
        LOG(INFO) << execution_order_[i];
      }
      for (size_t i = 0; i < io_offset_.size(); i++) {
        LOG(INFO) << "id " << io_offset_[i].id << " output offset " << io_offset_[i].output_offset;
        for (size_t j = 0; j < io_offset_[i].input_offsets.size(); j++)
          LOG(INFO) << "input offset " << io_offset_[i].input_offsets[j];
        for (size_t k = 0; k < io_offset_[i].input_fused.size(); k++)
          LOG(INFO) << "input fused function " << io_offset_[i].input_fused[k];
      }
    }
  */
  void preprocess(Network_io &input, Network_io &output) {
    // weight up limit offset 1.5G. TODO
    weight_offset_ = 0x5FF00000 ;

    input_ = input;
    output_ = output;

    //index = 0;
    std::string calibTableFile = defaultCompilerTestAppArgs.calibTable;
    if (calibTableFile != "") {
      FILE* fp = fopen(calibTableFile.c_str(), "r");
      char readBuffer[TEST_PARAM_FILE_MAX_SIZE] = {0};
      rapidjson::FileReadStream inStr(fp, readBuffer, sizeof(readBuffer));
      calibTable.ParseStream(inStr);
    } else {
      std::string str_ = "";
      int count_ = 300;
      for (int i = 0; i < count_; i++) {
        str_ = str_ + "\"" + std::to_string(i) + "\"" + ":" + "{\"scale\":0.01}";
        if (i != count_ - 1) str_ = str_ + ",";
      }
      str_ = "{" + str_ + "}";
      calibTable.Parse(str_.c_str());
    }
  }

 protected:
  void VisitExpr_(const VarNode* op) override {
    // LOG(INFO) << AsText(GetRef<Expr>(op), false);
  }

  void VisitExpr_(const ConstantNode* op) override { 
    // LOG(INFO) << GetRef<Expr>(op); 
  }

  //yuanyue 20220622 plan memory
  // do not allocate memory for tuple type 
  void VisitExpr_(const TupleNode* op) override {

    for (auto field : op->fields) 
      VisitExpr(field);
  }

  //yuanyue 20220622 plan memory
  // do not allocate memory for tuple type 
  void VisitExpr_(const TupleGetItemNode* op) override {
    VisitExpr(op->tuple);
  }

  void VisitExpr_(const CallNode* op) override {
    
    Fused_function_offsets fused_function_offsets;

    std::vector<int> istorage_id;
    std::vector<int> ostorage_id;
    std::vector<size_t> input_size;
    std::vector<size_t> output_size;

    //yuanyue 20220622 plan memory
    //int ostorage_id = storage_device_map_[GetRef<Expr>(op)][0][0];
    
    for (size_t i = 0; i < storage_device_map_[GetRef<Expr>(op)][0].size(); i++) {
      ostorage_id.push_back(storage_device_map_[GetRef<Expr>(op)][0][i]);
      output_size.push_back(storage_device_map_[GetRef<Expr>(op)][2][i]);
    }
    // int index_tmp = 0;
    for (auto arg : op->args) {
      VisitExpr(arg);
      for (size_t i = 0; i < storage_device_map_[arg][0].size(); i++) {
        istorage_id.push_back(storage_device_map_[arg][0][i]);
        input_size.push_back(storage_device_map_[arg][2][i]);
      }
    }

    i_scale_Vec.clear();
    for (auto arg : op->args) {
      if (scale_function_map_.find(arg) != scale_function_map_.end()) {
        LOG(INFO) << "DLA/RISC-V output scale(call): " << scale_function_map_[arg][0];
        i_scale_Vec.push_back(scale_function_map_[arg]);
      } else if (arg->IsInstance<tvm::relay::TupleNode>()) {
        LOG(INFO) << "DLA/RISC-V output scale(TupleNode)";
        const tvm::relay::TupleNode* arg_call = arg.as<tvm::relay::TupleNode>();
        i_scale_tuple.clear();
        for (auto field : arg_call->fields) {
          if (scale_function_map_.find(field) != scale_function_map_.end()) {
            for(auto arg: scale_function_map_[field]) {
              i_scale_tuple.push_back(arg);
              LOG(INFO) << "DLA/RISC-V output scale(tuple): " << arg;
            }
          } else {
            i_scale_tuple.push_back(-1.0f);
            LOG(INFO) << "DLA/RISC-V output scale(tuple) None";
          }
        }
        i_scale_Vec.push_back(i_scale_tuple);
      } else if (arg->IsInstance<tvm::relay::TupleGetItemNode>()) {
        LOG(INFO) << "DLA/RISC-V output scale(TupleGetItemNode)";
        const tvm::relay::TupleGetItemNode* arg_call = arg.as<tvm::relay::TupleGetItemNode>();
        i_scale_tuple.clear();
        if (scale_function_map_.find(arg_call->tuple) != scale_function_map_.end()) {
          i_scale_tuple.push_back(scale_function_map_[arg_call->tuple][arg_call->index]);
          LOG(INFO) << "DLA/RISC-V output scale(tupleGetItem): " << scale_function_map_[arg_call->tuple][arg_call->index];
        } else {
          i_scale_tuple.push_back(-1.0f);
          LOG(INFO) << "DLA/RISC-V output scale(tupleGetItem) None";
        }
        i_scale_Vec.push_back(i_scale_tuple);
      } else if (arg->IsInstance<tvm::relay::CallNode>() || arg->IsInstance<tvm::relay::VarNode>()) {
        LOG(INFO) << "DLA/RISC-V output scale(call || var) None";
        std::vector<float> scale_tmp{-1.0f};
        i_scale_Vec.push_back(scale_tmp);
      } else {
        LOG(INFO) << AsText(arg, false);
        LOG(INFO) << "DLA/RISC-V output other";
      }
    }

    // LOG(INFO) << AsText(GetRef<Expr>(op), false);
    // Expr expr = GetRef<Expr>(op);
    Function func;
    if (op->op.as<OpNode>()) {
      LOG(FATAL) << "Operators should be transformed away; try applying"
                 << "the fuse_ops transformation to the expression.";
    } else if (op->op.as<GlobalVarNode>()) {
      LOG(FATAL) << "Not implemented";
    } else if (op->op.as<FunctionNode>()) {
      func = GetRef<Function>(op->op.as<FunctionNode>());
      // LOG(INFO) << AsText(func, false);
    } else {
      LOG(FATAL) << "TVM runtime does not support calls to " << op->op->GetTypeKey();
    }
    //LOG(INFO) << "output: " << storage_device_map_[GetRef<Expr>(op)][0][0] <<",  "
    //<< storage_device_map_[GetRef<Expr>(op)][1][0] <<",  "
    //<< storage_device_map_[GetRef<Expr>(op)][2][0];

    //LOG(INFO) << "input: " << storage_device_map_[op->args[0]][0][0] <<",  "
    //<< storage_device_map_[op->args[0]][1][0] <<",  "
    //<< storage_device_map_[op->args[0]][2][0];

    //yuanyue 20220622 plan memory
    std::vector<size_t> input_offsets;
    std::vector<size_t> output_offset;

    for (size_t i = 0; i < istorage_id.size(); i++) {
      input_offsets.push_back(temporary_data_offset_[istorage_id[i]]);
    }
    for (size_t i=0; i<ostorage_id.size(); i++) {
      output_offset.push_back(temporary_data_offset_[ostorage_id[i]]);
    }
    fused_function_offsets.output_offsets = output_offset;
    fused_function_offsets.input_offsets = input_offsets;
    fused_function_offsets.input_size = input_size;
    fused_function_offsets.output_size = output_size;

    std::string compiler = func->GetAttr<String>(attr::kCompiler).value();
    // LOG(INFO) << compiler;
    if (compiler == "riscv") {
      LOG(INFO) << "(DLA) riscv func: " << AsText(func, false);
      auto genCalibdata = GenCalibdata(i_scale_Vec, "riscv");
      genCalibdata.VisitExpr(func->body);
      auto calibdata = genCalibdata.GetCalibdata();
      auto op_out_scale_map = genCalibdata.GetOpOutScaleMap();
      std::vector<float> subfunc_in_scale = genCalibdata.GetSubfuncOutScaleMap(func->body);

      if (!subfunc_in_scale.empty()) {
        scale_function_map_.insert(std::pair<Expr, std::vector<float>>(GetRef<Expr>(op), subfunc_in_scale));
      }

      // yuanyue 20220622 plan memory, need to add CompileFunc4Riscv
      Map<Expr, Array<IntegerArray>> storage_input_output_map; //expr id offset
      std:: vector <Expr> params;
      for (Var param : func->params) {     
        params.push_back(GetRef<Expr>(param.operator->()));
      }
      if (1){
        std::vector<Integer> offsets;
        for (size_t i = 0; i < output_offset.size(); i++){
          offsets.push_back(output_offset[i]);
        }
        ///yuanyue 20220621 quantized
        Expr expr=StorageAllocator().GetNotQuantizedExpr(func->body);
        //LOG(INFO) << "offsets: " << offsets[0];
        storage_input_output_map.Set(expr,Array<IntegerArray>({storage_device_map_[GetRef<Expr>(op)][0],offsets}));
      }
      int num_expr=0;
      for (auto arg : op->args) {
        //std::vector<Integer> storage_ids;      
        std::vector<Integer> offsets;
        for (size_t i =0; i < storage_device_map_[arg][0].size(); i++){
          int storage_id = storage_device_map_[arg][0][i];
          offsets.push_back(temporary_data_offset_[storage_id]);
        }
        //LOG(INFO) << "storage_device_map_[arg][0]: " << storage_device_map_[arg][0].size();
        //LOG(INFO) << "offsets: " << offsets[0];
        storage_input_output_map.Set(params[num_expr], Array<IntegerArray>({storage_device_map_[arg][0],offsets}));
        num_expr++;
      }

      assert(func->params.size() == i_scale_Vec.size());
      subfunc_in_scale_map.clear();
      for(size_t i=0;  i < func->params.size(); i++) {
        subfunc_in_scale_map.insert(std::pair<Expr, std::vector<float>>(func->params[i], i_scale_Vec[i]));
      }
      auto riscv_code = CompileFunc4Riscv(func, riscv_addr_list_, riscv_wt_list_, aid_dtype_map_,
                                          storage_input_output_map, total_memory_used_, weight_offset_, op_out_scale_map, subfunc_in_scale_map);

      // riscv_code_ = CompileFunc4Riscv(func, scale_map, iscale_);
      // std::vector<Cpu_param*> sub_order;
      
      if (riscv_code.empty()){
        riscv_addr_list_.pop_back();
        return;
      } 
      riscv_code_.push_back(riscv_code);
      execution_order_.push_back(riscv_order_);
      fused_function_offsets.id = riscv_order_;
      fused_function_map_.insert(std::pair<Expr, int>(GetRef<Expr>(op), riscv_order_));
      io_offset_.push_back(fused_function_offsets);
      //LOG(INFO) << "riscv_order_: " << riscv_order_;
      riscv_order_ -= 1;

    } else if (compiler.substr(0, 3) == "dla") {
      // LOG(INFO) << "DLA func: " << AsText(func, false);
      if (func->body->IsInstance<TupleNode>()) {
        LOG(FATAL) << "partition failed, can not have multiple output dla function";
      } else {
        LOG(INFO) << "DLA subfunc: " << AsText(func, false);

        auto genCalibdata = GenCalibdata(i_scale_Vec, "dla");
        genCalibdata.VisitExpr(func->body);
        auto calibdata = genCalibdata.GetCalibdata();
        std::vector<float> subfunc_in_scale = genCalibdata.GetSubfuncOutScaleMap(func->body);
        if (!subfunc_in_scale.empty()) {
          scale_function_map_.insert(std::pair<Expr, std::vector<float>>(GetRef<Expr>(op), subfunc_in_scale));
        }

        loadable_.push_back(CompileFunc4Dla(func, calibdata, i_scale_Vec));
        execution_order_.push_back(loadable_order_);
        fused_function_offsets.id = loadable_order_;
        fused_function_map_.insert(std::pair<Expr, int>(GetRef<Expr>(op), loadable_order_));
        io_offset_.push_back(fused_function_offsets);
        //LOG(INFO) << "loadable_order_: " << loadable_order_;
        loadable_order_ += 1;
      }
    } else {
      LOG(FATAL) << "unannotated call nodes are not allowed";
    }
  }

  void VisitExpr_(const LetNode* op) override {
    VisitExpr(op->value);
    VisitExpr(op->body);
    //LOG(INFO) << GetRef<Expr>(op);
  }
  /*
  void VisitExpr_(const TupleGetItemNode* op) override {
    tvm::runtime::contrib::Fused_function_offsets fused_function_offsets;
    VisitExpr(op->tuple);
    if (fused_function_map_.find(op->tuple) != fused_function_map_.end())
      fused_function_offsets.input_fused.push_back(fused_function_map_[op->tuple]);
    int index = op->index;
    fused_function_offsets.input_offsets.push_back(index);

    execution_order_.push_back(riscv_order_);
    fused_function_offsets.id = riscv_order_;
    fused_function_map_.insert(std::pair<Expr, int>(GetRef<Expr>(op), riscv_order_));
    riscv_order_ -= 1;
    std::vector<cpu_param*> dummy;
    riscv_code_.push_back(dummy);
    io_offset_.push_back(fused_function_offsets);

    // LOG(INFO) << storage_device_map_[GetRef<Expr>(op)][0][0];
  }
  */

  void VisitExpr_(const OpNode* op) override {
    throw std::runtime_error("can not compile op in non-eta expanded form");
  }

  void VisitExpr_(const GlobalVarNode* op) override { throw std::runtime_error(""); }

  void VisitExpr_(const IfNode* op) override { throw std::invalid_argument("if not supported"); }

  void VisitExpr_(const FunctionNode* op) override {
    ICHECK(op->GetAttr<String>(attr::kCompiler).defined())
        << "Only functions supported by custom codegen";
  }

  void VisitExpr_(const RefCreateNode* op) override {
    throw std::invalid_argument("reference not supported");
  }

  void VisitExpr_(const RefReadNode* op) override {
    throw std::invalid_argument("reference not supported");
  }

  void VisitExpr_(const RefWriteNode* op) override {
    throw std::invalid_argument("reference not supported");
  }

  void VisitExpr_(const ConstructorNode* op) override {
    throw std::invalid_argument("ADT constructor case not yet implemented");
  }

  void VisitExpr_(const MatchNode* op) override {
    throw std::invalid_argument("match case not yet implemented");
  }

  // every element is a dla function's loadable, the first is loadable size, second is a pointer of
  // loadable
  std::vector<std::pair<uint64_t, uint8_t*>> loadable_;
  // every element is a riscv function, every riscv function has many riscv operators
  std::vector<std::vector<Cpu_param*>> riscv_code_;
  uint32_t index;
  rapidjson::Document calibTable;

  // execution order
  // positive for loadable
  // negative for riscv
  std::vector<int> execution_order_;
  // dla function index, start from 1
  int loadable_order_;
  // riscv function index, start from -1
  int riscv_order_;

  // 20220622 yuanyue new 
  // key: plan memory of operators , 
  // value: three array are singly sorage_id ,s max_bytes (the bytes for a storage id) and the memory size for a expr
  Map<Expr, Array<IntegerArray>> storage_device_map_;
  // key: expr , 
  // value: two array are singly ref_dtypes, dtype_names
  Map<Expr, Array<IntegerArray>> aid_dtype_map_;

  std::map<Expr, int> fused_function_map_;
  // used for accumulated operator input and output, these memory block can be  reused
  //size_t data_memory_used_;
  Network_io input_;

  Network_io output_;

  size_t weight_offset_;
  // yuanyue 20220622 plan memory
  size_t total_memory_used_;
  // map between memory block and its' size
  std::map<int, size_t> temporary_data_storage_;
  // map between memory block and its' offset
  std::map<int, size_t> temporary_data_offset_;

  Riscv_addr_list riscv_addr_list_;

  Riscv_wt_list riscv_wt_list_;
  // yuanyue 20220602 plan memory
  // every fused function , tuple, tupleGetItem (the last two are deleted) has a Fused_function_offsets, it records
  // the dependency relations, include its' id, input functions' id
  std::vector<tvm::runtime::contrib::Fused_function_offsets> io_offset_;
  // map DLA DLA input scale
  std::map<Expr, std::vector<float>> scale_function_map_;
  // input scale ==> for TupleNode
  std::vector<float> i_scale_tuple;
  // input scale (subfunc)
  std::vector<std::vector<float>> i_scale_Vec;
  std::map<Expr, std::vector<float>> subfunc_in_scale_map;

};

runtime::Module CompileAipuFunc(Function func) {

  // some passes require global_symbol to be main
  func = WithAttr(std::move(func), tvm::attr::kGlobalSymbol, runtime::String("main"));

  DenseExpr DenseFunc;
  auto fDenseSp = DenseFunc.VisitExpr(func->body);
  auto mod_split = IRModule::FromExpr(fDenseSp);
  mod_split = transform::InferType()(mod_split);
  auto func_split = Downcast<Function>(mod_split->Lookup("main"));

  auto mod_batchmatmul = IRModule::FromExpr(func_split);
  mod_batchmatmul = transform::InferType()(mod_batchmatmul);
  mod_batchmatmul = transform::AIPU_fuse_GELU()(mod_batchmatmul);
  mod_batchmatmul = transform::AIPU_fuse_NORM()(mod_batchmatmul);
  mod_batchmatmul = transform::BatchMatMulTransform()(mod_batchmatmul);
  mod_batchmatmul = transform::FeatureToWeightTransform()(mod_batchmatmul);
  auto func_batchmatmul = Downcast<Function>(mod_batchmatmul->Lookup("main"));
  
  CollectOutput collectoutput;
  collectoutput.VisitExpr(func_batchmatmul->body);
  CustomizedAnnotation custom("dla", collectoutput.child_parent);
  auto new_func = custom.annotate(func_batchmatmul);
  
  auto mod = IRModule::FromExpr(new_func);
  // the first partition
  // mod = module_partition(mod);
  mod = relay::transform::MergeCompilerRegions()(mod);
  mod = relay::transform::PartitionGraph()(mod);

  mod = transform::FuseOps()(mod);
  mod = transform::Inline()(mod);
  mod = transform::InferType()(mod);
  // the second partition
  mod = eliminate_tuplenode(mod);
  LOG(INFO) << AsText(mod, false);
  // this pass is used to save the RISC-V relay subfunctions
  //mod = transform::subfunctionsWriterTransform()(mod);
  //
  //yuanyue 20220713 aid_dtype
  Map<Expr, Array<IntegerArray>> aid_dtype_map;
  aid_dtype_map = AidDtypeExpr().GetAidDtype(Downcast<Function>(mod->Lookup("main")));

  //yuanyue 20220622 plan memory
  Map<Expr, Array<IntegerArray>> storage_device_map;
  std::string c_name="func"; 
  std::map<int, size_t> temporary_data_storage;
  std::map<int, size_t> temporary_data_offset;
  size_t total_memory_used = 0;

  StorageAllocator storage_allocator;
  storage_device_map = storage_allocator.Plan(Downcast<Function>(mod->Lookup("main")), aid_dtype_map, total_memory_used, c_name);
  total_memory_used = storage_allocator.GetTotalMemory();
  temporary_data_storage = storage_allocator.GetDataStorage();
  temporary_data_offset = storage_allocator.GetDataOffset();

  Network_io input;
  Network_io output;

  for (auto param : Downcast<Function>(mod->Lookup("main"))->params) {
    std::string name = param->name_hint();
    size_t offset = temporary_data_offset[storage_device_map[param][0][0]];
    size_t size = storage_device_map[param][2][0];
    std::pair<size_t, size_t> pa(offset, size);
    input.insert(std::pair<std::string, std::pair<size_t, size_t>>(name, pa));
  }

  for (size_t i = 0; i < storage_device_map[Downcast<Function>(mod->Lookup("main"))->body][0].size(); i++) {
    std::string name = std::to_string(i);
    size_t offset = temporary_data_offset[storage_device_map[Downcast<Function>(mod->Lookup("main"))->body][0][i]];
    size_t size = storage_device_map[Downcast<Function>(mod->Lookup("main"))->body][2][i];
    std::pair<size_t, size_t> pa(offset, size);
    output.insert(std::pair<std::string, std::pair<size_t, size_t>>(name, pa));
  }

  //auto traversal_module = TraversalModule(storage_device_map);
  auto traversal_module = TraversalModule(storage_device_map, aid_dtype_map, temporary_data_storage, temporary_data_offset, total_memory_used);
  traversal_module.preprocess(input, output);
  traversal_module.VisitExpr(Downcast<Function>(mod->Lookup("main"))->body);

  //traversal_module.debug();
  return traversal_module.AIPUModuleCreate();
}

TVM_REGISTER_GLOBAL("relay.ext.aipu").set_body_typed(CompileAipuFunc);
}  // namespace aipu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
