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
 * \file relay/backend/build_module.cc
 * \brief Code generation for TVM's graph executor.
 */
#include <tvm/driver/driver_api.h>
#include <tvm/ir/expr.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/qnn/transform.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/device_api.h>
#include <tvm/relay/attrs/annotation.h>

#include <memory>

#include "../../target/func_registry_generator.h"
#include "../../target/source/codegen_source_base.h"
#include "compile_engine.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace backend {

using TargetsMap = Map<tvm::Integer, tvm::Target>;
using namespace tvm::relay::transform;

/*!
 * \brief Output of building module
 *
 */
struct BuildOutput {
  std::string graph_json;
  runtime::Module mod;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
};

/*!
 * \brief GraphCodegen module wrapper
 *
 */
struct GraphCodegen {
 public:
  GraphCodegen() {
    auto pf = GetPackedFunc("relay.build_module._GraphExecutorCodegen");
    mod = (*pf)();
  }
  ~GraphCodegen() {}

  void Init(runtime::Module* m, TargetsMap targets) { CallFunc("init", m, targets); }

  void Codegen(const Function& func) { CallFunc("codegen", func); }

  std::string GetJSON() { return CallFunc<std::string>("get_graph_json", nullptr); }

  Array<tvm::runtime::Module> GetExternalModules() {
    return CallFunc<Array<tvm::runtime::Module>>("get_external_modules", nullptr);
  }

  Map<String, IRModule> GetIRModule() {
    return CallFunc<Map<String, IRModule>>("get_irmodule", nullptr);
  }

  std::unordered_map<std::string, tvm::runtime::NDArray> GetParams() {
    std::unordered_map<std::string, tvm::runtime::NDArray> ret;
    auto names = CallFunc<Array<runtime::String>>("list_params_name", nullptr);
    for (const auto& expr : names) {
      // Implicit cast from runtime::String to std::string
      std::string key = expr;
      ret[key] = CallFunc<runtime::NDArray>("get_param_by_name", key);
    }
    return ret;
  }

  std::unordered_map<std::string, int64_t> GetParamIds() {
    std::unordered_map<std::string, int64_t> ret;
    auto names = CallFunc<Array<runtime::String>>("list_params_name", nullptr);
    for (const auto& expr : names) {
      // Implicit cast from runtime::String to std::string
      std::string key = expr;
      ret[key] = CallFunc<int64_t>("get_param_id", key);
    }
    return ret;
  }

 protected:
  tvm::runtime::Module mod;
  template <typename R, typename... Args>
  R CallFunc(const std::string& name, Args... args) {
    auto pf = mod.GetFunction(name, false);
    return pf(std::forward<Args>(args)...);
  }
  template <typename... Args>
  void CallFunc(const std::string& name, Args... args) {
    auto pf = mod.GetFunction(name, false);
    pf(std::forward<Args>(args)...);
    return;
  }
};


/*!
 * \brief customized annotation for Function
 *
 */
class CustomizedAnnotation : tvm::relay::ExprMutator {
 public:
  explicit CustomizedAnnotation(std::string compiler)
    : compiler_(compiler) {}

  Expr VisitExpr_(const CallNode* call_node) final {
    auto compiler_begin = tvm::relay::Op::Get("annotation.compiler_begin");
    auto compiler_end = tvm::relay::Op::Get("annotation.compiler_end");
    auto attrs_compiler = tvm::runtime::make_object<tvm::relay::CompilerAttrs>();
    auto attrs_default = tvm::runtime::make_object<tvm::relay::CompilerAttrs>();
    attrs_compiler->compiler = compiler_;
    attrs_default->compiler = "default";

    tvm::Array<Expr> call_args;
    auto new_op = VisitExpr(call_node->op);

    const auto callop_node = call_node->op.as<OpNode>();
    std::string callop_name = tvm::runtime::GetRef<Op>(callop_node)->name;
    bool callop_on_default_device = operator_set.find(callop_name) != operator_set.end();

    for (auto arg : call_node->args) {
      if (arg->IsInstance<CallNode>()) {
        arg = VisitExpr(arg);
      }
      else if (arg->IsInstance<TupleNode>()) {
        arg = VisitExpr(arg);
      }

      if (callop_on_default_device) {
        arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_default), {});
      }
      else {
        arg = tvm::relay::Call(compiler_begin, {arg}, tvm::Attrs(attrs_compiler), {});
      }

      call_args.push_back(arg);
    }

    Expr result = tvm::relay::Call(new_op, call_args, call_node->attrs, call_node->type_args, call_node->span);
    if (callop_on_default_device) {
      result = tvm::relay::Call(compiler_end, {result}, tvm::Attrs(attrs_default), {});
    }
    else {
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
    attrs_default->compiler = "default";

    tvm::Array<Expr> tmp_fields;
    tvm::Array<Expr> fields;
    Expr result;
    // whether there is a field on aipu, if it is true, we put the tuple node on the aipu, else put it on the default device(cpu).
    bool field_on_aipu = false;
    for (auto field : tuple_node->fields) {
      // assuming that the tuple_node field are not all constant
      if (field->IsInstance<CallNode>()) {
        const CallNode* node = (static_cast<const CallNode *>(field.get()));
        const auto op_node = node->op.as<OpNode>();
        std::string op_name = tvm::runtime::GetRef<Op>(op_node)->name;
        if (operator_set.find(op_name) == operator_set.end()) {
          field_on_aipu = true;
        }
        field = VisitExpr(field);
      }
      tmp_fields.push_back(field);
    }

    if (field_on_aipu) {
      for (auto field : tmp_fields) {
        field = tvm::relay::Call(compiler_begin, {field},
                                 tvm::Attrs(attrs_compiler), {});
        fields.push_back(field);
      }
      result = tvm::relay::Tuple(fields, {});
      result = tvm::relay::Call(compiler_end, {result},
                                tvm::Attrs(attrs_compiler), {});
    }
    else {
      for (auto field : tmp_fields) {
        field = tvm::relay::Call(compiler_begin, {field},
                                 tvm::Attrs(attrs_default), {});
        fields.push_back(field);
      }
      result = tvm::relay::Tuple(fields, {});
      result = tvm::relay::Call(compiler_end, {result},
                                tvm::Attrs(attrs_default), {});
    }
    return result;
  }

  tvm::relay::Function annotate(tvm::relay::Function func){
    auto new_expr = VisitExpr(func);
    const tvm::relay::FunctionNode * new_funcnode= static_cast<const tvm::relay::FunctionNode *>(new_expr.get());
    return tvm::runtime::GetRef<tvm::relay::Function>(new_funcnode);
  }

  // std::set<std::string> operator_set = {"nn.softmax"};
  std::set<std::string> operator_set;
  std::string compiler_;
};

/*!
 * \brief Relay build module
 *
 */
class RelayBuildModule : public runtime::ModuleNode {
 public:
  /*!
   * \brief Get member function to front-end
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_graph_json") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetGraphJSON(); });
    } else if (name == "get_module") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetModule(); });
    } else if (name == "build") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        // LOG(INFO)<<"PACKEDFUNC PRINT";
        ICHECK_EQ(args.num_args, 3);
        this->Build(args[0], args[1], args[2]);
      });
    } else if (name == "list_params") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->ListParamNames(); });
    } else if (name == "get_params") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetParams(); });
    } else if (name == "set_params") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Map<String, Constant> params = args[0];
        for (const auto& kv : params) {
          this->SetParam(kv.first, kv.second->data);
        }
      });
    } else if (name == "get_irmodule") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->graph_codegen_->GetIRModule();
      });
    } else if (name == "get_external_modules") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->graph_codegen_->GetExternalModules();
      });
    } else if (name == "optimize") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 2);
        *rv = this->Optimize(args[0], args[1], this->params_);
      });
    } else {
      LOG(FATAL) << "Unknown packed function: " << name;
      return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
    }
  }

  /*!
   * \brief Get the GraphJSON for runtime
   *
   * \return const std::string graph_json
   */
  const std::string& GetGraphJSON() { return ret_.graph_json; }

  /*!
   * \brief Get the Module object
   *
   * \return runtime::Module
   */
  runtime::Module GetModule() { return ret_.mod; }

  /*!
   * \brief List all paramter names
   *
   * \return Array<runtime::String> names of params
   */
  Array<runtime::String> ListParamNames() {
    Array<runtime::String> ret;
    for (const auto& kv : params_) {
      ret.push_back(kv.first);
    }
    return ret;
  }

  /*!
   * \brief Get params dictionary
   *
   * \return Map<String, Constant> params dictionary
   */
  Map<String, Constant> GetParams() {
    Map<String, Constant> ret;
    for (const auto& kv : ret_.params) {
      ret.Set(kv.first, Constant(kv.second));
    }
    return ret;
  }

  /*!
   * \brief Set the parameters
   *
   * \param name name of parameter
   * \param data_in input DLTensor
   */
  void SetParam(const std::string& name, runtime::NDArray data_in) { params_[name] = data_in; }

  /*!
   * \brief type key
   *
   * \return const char*
   */
  const char* type_key() const final { return "RelayBuildModule"; }

  /*!
   * \brief Build relay IRModule for graph executor
   *
   * \param mod Relay IRModule
   * \param target Target device
   * \param target_host Host target device
   */
  void Build(IRModule mod, const TargetsMap& targets, const tvm::Target& target_host) {
    // Create protected variable targets_ from ground up
    targets_ = targets;
    target_host_ = target_host;
    CheckAndUpdateHostConsistency(&targets_, &target_host_);
    BuildRelay(mod, params_);
    // Clear compile engine so that tuning schedules can be changed between runs. See issue #6096.
    CompileEngine::Global()->Clear();
  }

 protected:
  /*!
   * \brief Optimize a Relay IRModule.
   *
   * \param relay_module The input IRModule where optmization will be applied on.
   * \param targets The device type to `Target` mapping.
   * \param params The param name to value mapping.
   *
   * \return relay::IRModule The updated Relay IR module after optimization.
   */
  IRModule Optimize(IRModule relay_module, const TargetsMap& targets,
                    const std::unordered_map<std::string, runtime::NDArray>& params) {
    ICHECK(relay_module.defined()) << "The IRModule must be defined for the Relay compiler.";

    if (params.size()) {
      ICHECK(relay_module->ContainGlobalVar("main")) << "Missing the main entry function";
      GlobalVar main_glb_var = relay_module->GetGlobalVar("main");
      Function main_func = Downcast<Function>(relay_module->Lookup(main_glb_var));
      auto new_main = BindParamsByName(main_func, params);
      IRModuleNode* relay_module_ptr = relay_module.CopyOnWrite();
      relay_module_ptr->Update(main_glb_var, new_main);
    }

    Array<Pass> pass_seqs;
    Array<runtime::String> entry_functions{"main"};
    pass_seqs.push_back(transform::RemoveUnusedFunctions(entry_functions));
    pass_seqs.push_back(transform::ToBasicBlockNormalForm());

    // Run all dialect legalization passes.
    pass_seqs.push_back(relay::qnn::transform::Legalize());

    // Legalize pass is restricted to homogeneous execution for now.
    if (targets.size() == 1) {
      pass_seqs.push_back(transform::Legalize());
    }

    pass_seqs.push_back(transform::SimplifyInference());

    // Convert Dynamic ops to static versions
    pass_seqs.push_back(transform::DynamicToStatic());

    PackedFunc fskip = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
      Expr expr = args[0];
      *rv = false;
      if (expr.as<CallNode>()) {
        auto call_node = expr.as<CallNode>();
        auto op_node = call_node->op.as<OpNode>();
        if (op_node->name == "cast") {
          auto attrs = call_node->attrs.as<CastAttrs>();
          if (attrs->dtype == DataType::Int(32)) {
            *rv = true;
          }
        }
      }
    });
    pass_seqs.push_back(transform::EliminateCommonSubexpr(fskip));
    pass_seqs.push_back(transform::SimplifyExpr());
    pass_seqs.push_back(transform::CombineParallelConv2D(3));
    pass_seqs.push_back(transform::CombineParallelDense(3));
    pass_seqs.push_back(transform::CombineParallelBatchMatmul(3));
    pass_seqs.push_back(transform::FoldConstant());
    pass_seqs.push_back(transform::FoldScaleAxis());
    pass_seqs.push_back(transform::CanonicalizeCast());
    pass_seqs.push_back(transform::CanonicalizeOps());

    // Alter layout transformation is only applied to homogeneous execution yet.
    if (targets.size() == 1) {
      pass_seqs.push_back(transform::InferType());
      pass_seqs.push_back(transform::AlterOpLayout());
    }

    // Fast math optimizations.
    pass_seqs.push_back(transform::FastMath());
    pass_seqs.push_back(transform::FoldConstant());

    // Create a sequential pass and perform optimizations.
    transform::Pass seq = transform::Sequential(pass_seqs);
    if (targets.size() == 1) {
      const auto& it = targets.begin();
      With<Target> tctx((*it).second);
      relay_module = seq(relay_module);
    } else {
      relay_module = seq(relay_module);
    }

    // Handle heterogeneous compilation.
    transform::PassContext pass_ctx = PassContext::Current();
    if (targets_.size() > 1) {
      Optional<Integer> opt_fallback_dev =
          pass_ctx->GetConfig("relay.fallback_device_type", Integer(static_cast<int>(kDLCPU)));
      auto fallback_dev = opt_fallback_dev.value();
      ICHECK_GT(fallback_dev->value, 0U);
      relay_module = RunDeviceAnnotationPass(relay_module, fallback_dev->value);
    }

    // Fuse the operations if it is needed.
    relay_module = transform::FuseOps()(relay_module);

    // Do layout rewrite for auto-scheduler.
    if (backend::IsAutoSchedulerEnabled() && targets.size() == 1) {
      const auto& target = (*targets.begin()).second;
      Pass major_pass = transform::AutoSchedulerLayoutRewrite();
      bool enable_layout_rewrite_targets =
          target->kind->device_type == kDLCPU || target->GetAttr<String>("device", "") == "mali";
      if (enable_layout_rewrite_targets && pass_ctx.PassEnabled(major_pass->Info())) {
        With<Target> tctx(target);
        relay_module = major_pass(relay_module);
        // Defuse ops to fold constants, then fuse them again
        relay_module = transform::DefuseOps()(relay_module);
        relay_module = transform::FoldConstant()(relay_module);
        relay_module = transform::FuseOps()(relay_module);
      }
    }

    relay_module = transform::InferType()(relay_module);

    // Inline the functions that have been lifted by the module scope.
    //
    // TODO(@zhiics) Note that we need to be careful about the subgraphs with
    // global function calls. We should make sure that these callees are also
    // inline functions. However, this should be very unlikely for accelerators
    // and vendor-provided libraries. So we don't handle for now.
    relay_module = transform::Inline()(relay_module);
    relay_module = transform::InferType()(relay_module);

    ICHECK(relay_module.defined());

    return relay_module;
  }

  /*!
   * \brief Create a default type.
   * \param device_type The device type index.
   * \return the default target for the device.
   */
  Target CreateDefaultTarget(int device_type) {
    std::string name = runtime::DeviceName(device_type);
    if (name == "cpu") return Target("llvm");
    if (name == "gpu") return Target("cuda");
    return Target(name);
  }

  /*!
   * \brief Update the target and fallback device required for heterogeneous
   * compilation. CPU is used as the fallback device if it wasn't provided.
   * Meanwhile, a CPU device type and "llvm" pair will be added to the target
   * dictionary in this case.
   *
   * \param fallback_device The fallback device for heterogeneous execution.
   */
  void UpdateHeterogeneousInputs(int fallback_device) {
    std::unordered_map<int64_t, tvm::Target> tmp_map;
    for (const auto& kv : targets_) {
      tmp_map[kv.first->value] = kv.second;
    }
    if (tmp_map.count(fallback_device) == 0) {
      targets_.Set(fallback_device, CreateDefaultTarget(fallback_device));
    }
  }

  /*!
   * \brief Execute the device annotation passes to update the input program and
   *        target information.
   *
   * \param relay_module The input Relay module.
   * \param fallback_device The fallback device for heterogeneous execution.
   *
   * \return updated_module The updated module after device annotation.
   */
  IRModule RunDeviceAnnotationPass(const IRModule& relay_module, int fallback_device) {
    UpdateHeterogeneousInputs(fallback_device);
    auto rewrite = transform::RewriteAnnotatedOps(fallback_device);
    auto updated_module = rewrite(relay_module);
    ICHECK(updated_module.defined());

    tvm::Map<Expr, Integer> device_map;
    for (const auto& it : updated_module->functions) {
      device_map = relay::CollectDeviceInfo(it.second);
      if (!device_map.empty()) break;
    }

    if (device_map.empty()) {
      tvm::Map<Expr, Integer> annotation_map;
      for (const auto& it : relay_module->functions) {
        annotation_map = relay::CollectDeviceAnnotationOps(it.second);
        if (!annotation_map.empty()) break;
      }
      // None op is annotated but they are fallen back to the default device.
      if (annotation_map.empty()) {
        targets_.Set(0, CreateDefaultTarget(fallback_device));
      } else {
        // All ops are annotated to the same device type.
        int64_t dev_type = -1;
        for (auto kv : annotation_map) {
          dev_type = kv.second->value;
          break;
        }
        for (auto kv : annotation_map) {
          ICHECK_EQ(kv.second->value, dev_type) << "Expressions in the function are "
                                                << "annotated with various device types,"
                                                << "but not device copy operators "
                                                << "found. Please check the "
                                                << "RewriteAnnotation pass.";
        }
        targets_.Set(0, CreateDefaultTarget(dev_type));
      }
    }
    return updated_module;
  }

  /*!
   * \brief Compile a Relay IR module to runtime module.
   *
   * \param relay_module The Relay IR module.
   * \param params The parameters.
   */
  void BuildRelay(IRModule relay_module,
                  const std::unordered_map<std::string, tvm::runtime::NDArray>& params) {
    Target target_host = GetTargetHost();
    // If no target_host has been set, we choose a default one, which is
    // llvm if "codegen.LLVMModuleCreate" is accessible.
    const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.LLVMModuleCreate");
    if (!target_host.defined()) target_host = (pf != nullptr) ? Target("llvm") : Target("stackvm");

    // Update all the targets in the targets_ TargetsMap
    CheckAndUpdateHostConsistency(&targets_, &target_host);

    // Relay IRModule -> IRModule optimizations.
    relay_module = Optimize(relay_module, targets_, params);

    // auto tmp_func = Downcast<Function>(relay_module->Lookup("main"));
    // LOG(INFO) << AsText(tmp_func, false);
    // used for aipu target
    if (targets_.size() == 1)
      {
        const auto& target = (*targets_.begin()).second;
        if (target->kind->name == "aipu")
          {
            auto defused_module = transform::DefuseOps()(relay_module);
            auto defused_func = Downcast<Function>(defused_module->Lookup("main"));
            // auto tmp_node = static_cast<const Call *>(&(defused_func->body));
            // LOG(INFO) << (*tmp_node)->args.size();
            // CustomizedAnnotation custom("aipu");
            // auto new_func = custom.annotate(defused_func);
            // LOG(INFO) << AsText(new_func, false);
            // auto mod = IRModule::FromExpr(new_func);
            // auto mod = IRModule::FromExpr(func);
            // mod = relay::transform::InferType()(mod);
            // auto merge_pass = relay::transform::MergeCompilerRegions();
            // mod = relay::transform::MergeCompilerRegions()(mod);
            // mod = relay::transform::PartitionGraph()(mod);
            // auto fx = mod->Lookup("main");
            // LOG(INFO) << AsText(mod, false);
            // wrap the whole function
            // set the necessary attributes
            defused_func = WithAttr(std::move(defused_func), "Compiler", tvm::runtime::String("aipu"));
            defused_func = WithAttr(std::move(defused_func), "global_symbol", tvm::runtime::String("main"));
            defused_func = WithAttr(std::move(defused_func), "Primitive", IntImm(tvm::runtime::DataType::Int(32), 1));

            // copy the origin function's parameters
            tvm::Array<Expr> tmp_params = {};
            int count = 0;
            for (auto arg : defused_func->params)
              {
                auto type_annotation = arg->type_annotation;
                std::string tmp_string = "p" + std::to_string(count);
                auto tmp_arg = Var(tmp_string, type_annotation);
                tmp_params.push_back(tmp_arg);
                count += 1;
              }

            // finish wrapping
            auto call = Call(defused_func, tmp_params);
            auto mod = IRModule::FromExpr(call, {}, {});
            relay_module->Update(mod);
            // InferType pass is needed by new module
            relay_module = transform::InferType()(relay_module);
            // relay_module = Optimize(relay_module, targets_, params);

            // targets_.Set(0, Target("llvm"));
          }
      }

    auto func = Downcast<Function>(relay_module->Lookup("main"));
    // LOG(INFO)<<"The optimized func is "<<AsText(func,false);
    graph_codegen_ = std::unique_ptr<GraphCodegen>(new GraphCodegen());
    // LOG(INFO)<<"Before Init";
    graph_codegen_->Init(nullptr, targets_);
    // LOG(INFO)<<"After Init";
    graph_codegen_->Codegen(func);
    // LOG(INFO)<<"After Codegen";

    ret_.graph_json = graph_codegen_->GetJSON();
    ret_.params = graph_codegen_->GetParams();

    auto lowered_funcs = graph_codegen_->GetIRModule();

    // Generate a placeholder function that attaches linked params as its arguments.
    if (target_host->GetAttr<Bool>("link-params").value_or(Bool(false))) {
      CHECK(pf != nullptr) << "Unable to link-params with no target_host and no llvm codegen.";
      auto param_ids = graph_codegen_->GetParamIds();
      auto link_params = Map<String, tir::LinkedParam>();
      for (auto param : ret_.params) {
        link_params.Set(param.first, tir::LinkedParam(param_ids[param.first], param.second));
      }

      Map<String, ObjectRef> dict;
      dict.Set(tvm::tir::attr::kLinkedParams, link_params);
      dict.Set(tvm::attr::kGlobalSymbol, String(::tvm::runtime::symbol::tvm_lookup_linked_param));
      DictAttrs attrs{dict};
      auto prim = tir::PrimFunc(Array<tir::Var>(), tir::SeqStmt(Array<tir::Stmt>()), VoidType(),
                                Map<tir::Var, tir::Buffer>(), attrs);
      if (lowered_funcs.find(target_host->str()) == lowered_funcs.end()) {
        lowered_funcs.Set(target_host->str(), IRModule(Map<GlobalVar, BaseFunc>({})));
      }
      lowered_funcs[target_host->str()]->Add(
          GlobalVar(::tvm::runtime::symbol::tvm_lookup_linked_param), prim);
    }

    // When there is no lowered_funcs due to reasons such as optimization.
    if (lowered_funcs.size() == 0) {
      if (target_host.defined() && target_host->kind->name == "llvm") {
        // If we can decide the target is LLVM, we then create an empty LLVM module.
        ret_.mod = (*pf)(target_host->str(), "empty_module");
      } else {
        // If we cannot decide the target is LLVM, we create an empty CSourceModule.
        // The code content is initialized with ";" to prevent complaining
        // from CSourceModuleNode::SaveToFile.
        ret_.mod = tvm::codegen::CSourceModuleCreate(";", "", Array<String>{});
      }
    } else {
      ret_.mod = tvm::build(lowered_funcs, target_host_);
    }

    auto ext_mods = graph_codegen_->GetExternalModules();
    ret_.mod = tvm::codegen::CreateMetadataModule(ret_.params, ret_.mod, ext_mods, GetTargetHost());
  }

 private:
  Target GetTargetHost() {
    Target target_host = target_host_;
    if (!target_host_.defined()) {
      for (const auto& it : targets_) {
        if (it.second->kind->device_type == kDLCPU) {
          target_host = it.second;
          break;
        }
      }
    }
    return target_host;
  }

 protected:
  std::unique_ptr<GraphCodegen> graph_codegen_;
  /*! \brief target device */
  TargetsMap targets_;
  /*! \brief target host device */
  tvm::Target target_host_;
  /*! \brief parameters */
  std::unordered_map<std::string, runtime::NDArray> params_;
  /*! \brief building output */
  BuildOutput ret_;
};

runtime::Module RelayBuildCreate() {
  auto exec = make_object<RelayBuildModule>();
  return runtime::Module(exec);
}

TVM_REGISTER_GLOBAL("relay.build_module._BuildModule").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = RelayBuildCreate();
});

TVM_REGISTER_GLOBAL("relay.build_module.BindParamsByName")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      Map<String, Constant> params = args[1];
      std::unordered_map<std::string, runtime::NDArray> params_;
      for (const auto& kv : params) {
        params_[kv.first] = kv.second->data;
      }
      *rv = relay::backend::BindParamsByName(args[0], params_);
    });

}  // namespace backend
}  // namespace relay
}  // namespace tvm
