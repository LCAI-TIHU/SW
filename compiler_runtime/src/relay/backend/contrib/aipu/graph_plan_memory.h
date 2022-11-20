#ifndef TVM_RELAY_BACKEND_CONTRIB_GRAPH_PLAN_MEMORY_H_
#define TVM_RELAY_BACKEND_CONTRIB_GRAPH_PLAN_MEMORY_H_


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
 * \file relay/backend/graph_plan_memory.h
 * \brief Memory index assignment pass for executing
 *   the program in the graph executor.
 */


#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/tir/op.h>

#include "../../../../support/arena.h"
#include "../../utils.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace aipu {

using IntegerArray = Array<Integer>;

struct StorageToken {
  /*! \brief Reference counter */
  int ref_counter{0};
  /*! \brief number of bytes */
  size_t max_bytes{0};
  /*! \brief The corresponding tensor type node. */
  const TensorTypeNode* ttype{nullptr};
  /*! \brief virtual device index that corresponds to the device_type in
   * DLDevice. */
  int device_type{0};
  /*! \brief The storage id */
  int64_t storage_id{-1};
  ///* \brief The expr type, 0 for mid expr. 1 for input expr, 2 for output expr  */
  //int expr_type{0};
};

class TuplePlan : public ExprVisitor {
 public:
  explicit TuplePlan(const Function& func, const Map<Expr, Array<IntegerArray>>& smap, int tuple_id): smap_(smap), func_(func), tuple_id_(tuple_id){   
  }
  //TuplePlan() {}

  // run the visitor on a function.
  Map<Expr, Array<IntegerArray>>  Run() {
    storage_tuple_map_.clear();
    c_name_ = "func_";
    VisitExpr(func_);
    //LOG(INFO)<< "smap_: " << smap_.size() << " storage_tuple_map_: " << storage_tuple_map_.size();
    //LOG(INFO)<< "smap_: " << smap_ << " storage_tuple_map_: " << storage_tuple_map_;
    return storage_tuple_map_;
  }
  int GetTuoleID(){
    return tuple_id_;
  }
 protected:
  void SetTupleTypeNode(const CallNode* op, Function func_node){
    Expr expr = GetRef<Expr>(op);
    //auto op_type_node = op->checked_type().as<TupleTypeNode>();
    auto func_tuple_node = func_node->body.as<TupleNode>();
    if (func_tuple_node) {// tuple node
      std::vector <bool> flag_quantized_ops;
      bool flag_quantized_op = false;
      for (auto field : func_tuple_node->fields){
        if (field == GetNotQuantizedExpr(field)){
          flag_quantized_ops.push_back(false);
        }
        else{
          flag_quantized_ops.push_back(true);
          flag_quantized_op = true;
        }
      }
      if (true){ //quantized op in tuple
        // if the tuple branch is not quantized op ,its value is -1; 
        // if the tuple branch is quantized op , but the expr its pointed to that is not in tuple. its value is also -1; 
        // if the tuple branch is quantized op , and the expr its pointed to that is in tuple. its value is a new storage id;
        std::vector <int> quantized_index(flag_quantized_ops.size(),-1); 
        bool flag_quantized_op_in_tuple_branch = false;
        for (size_t i = 0 ; i< flag_quantized_ops.size(); i++){
          if (flag_quantized_ops[i]){
            Expr base_expr = GetNotQuantizedExpr(func_tuple_node->fields[i]);
            for (size_t j = 0; j< func_tuple_node->fields.size(); j++ ){
              if (base_expr == func_tuple_node->fields[j]){
                flag_quantized_op_in_tuple_branch = true;
                quantized_index[i] = tuple_id_;
                quantized_index[j] = tuple_id_;
                tuple_id_++;
              }
            }
          }
        }
        //
        if (flag_quantized_op_in_tuple_branch){//the expr quantized op pointed to that is in tuple
          //LOG(INFO)<<"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^";
          //LOG(INFO)<<"visit into the expr quantized op pointed to that is in tuple" << AsText(func_node,false);
          std::vector<Integer> storage_ids;
          std::vector<Integer> max_bytes;  // the max bytes for a storage id
          std::vector<Integer> memory_sizes;  // the memory size for a expr
          for (size_t i = 0 ; i< quantized_index.size(); i++) {
            if (quantized_index[i] == -1){
              storage_ids.push_back(smap_[expr][0][i]);
              max_bytes.push_back(smap_[expr][1][i]);
              memory_sizes.push_back(smap_[expr][2][i]);
            }
            else{
              int id = quantized_index[i];
              storage_ids.push_back(id);
              max_bytes.push_back(smap_[expr][2][i]);
              memory_sizes.push_back(smap_[expr][2][i]);
            }
          }
          storage_tuple_map_.Set(expr, Array<IntegerArray>({storage_ids, max_bytes, memory_sizes}));
        }
        else{//the expr quantized op pointed to that is not in tuple
          storage_tuple_map_.Set(expr,smap_[expr]);
        }
      }
      else{ // no quantized op in tuple
        storage_tuple_map_.Set(expr,smap_[expr]);
      }
    } else { // call node 
      storage_tuple_map_.Set(expr,smap_[expr]);
      //LOG(FATAL) << "TVM TuplePlan does not support calls to " << func_node->body->GetTypeKey();
    }
  }
  // The call map
  void VisitExpr_(const CallNode* op) final {
    Expr expr = GetRef<Expr>(op);
    for (Expr arg : op->args) {
      arg = GetNotQuantizedExpr(arg);
      VisitExpr(arg);
    }
    //LOG(INFO)<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!CallNode!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: "<<storage_tuple_map_.size();
    // deal with function
    if (c_name_ == "func_"){
      Function func_node;
      if(op->op.as<FunctionNode>()){
        func_node = GetRef<Function>(op->op.as<FunctionNode>());
      } 
      else {
        LOG(FATAL) << "TVM TuplePlan does not support calls to " << op->op->GetTypeKey();
      }
      std::string compiler = func_node->GetAttr<String>(attr::kCompiler).value();
      //LOG(INFO)<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!CallNode!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: "<<compiler;
      if (compiler == "riscv")  // riscv func
      {
        auto type_node = op->checked_type().as<TupleTypeNode>();
        if (type_node) {
          SetTupleTypeNode(op, func_node);
        } else {
          storage_tuple_map_.Set(expr,smap_[expr]);
        }
      }
      else if(compiler.substr(0, 3) == "dla"){
        storage_tuple_map_.Set(expr,smap_[expr]);
      }
      else {
        LOG(FATAL) << "TVM TuplePlan does not support compiler name: " << compiler;
      }
    }
  }

  void VisitExpr_(const ConstantNode* op) final {
    Expr expr = GetRef<Expr>(op);
    storage_tuple_map_.Set(expr,smap_[expr]);
  }
  void VisitExpr_(const VarNode* op) final {
    Expr expr = GetRef<Expr>(op);
    storage_tuple_map_.Set(expr,smap_[expr]);
  }

  void VisitExpr_(const TupleNode* op) final {
    //LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!! Go into tuple !!!!!!!!!!!!!!!!!!!!!!!" ;
    Expr expr = GetRef<Expr>(op);
    std::vector<Integer> storage_ids;
    std::vector<Integer> max_bytes;  // the max bytes for a storage id
    std::vector<Integer> memory_sizes;  // the memory size for a expr
    for (Expr field : op->fields) {
      Expr field_branch = GetNotQuantizedExpr(field);
      VisitExpr(field_branch);
      for (size_t i = 0;i< smap_[field_branch][0].size();i++){
        storage_ids.push_back(smap_[field_branch][0][i]);
        max_bytes.push_back(smap_[field_branch][1][i]);
        memory_sizes.push_back(smap_[field_branch][2][i]);
      }
    }
    storage_tuple_map_.Set(expr, Array<IntegerArray>({storage_ids, max_bytes, memory_sizes}));
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    Expr expr = GetRef<Expr>(op);
    //LOG(INFO)<<"!!!!!!!!!!!!!!!!!TupleGetItemNode!!!!!!!!!!!!!!!!!!!!"<<op->tuple->GetTypeKey();
    //LOG(INFO)<<"smap_[expr][0].size(): "<< smap_[expr][0].size();
    //LOG(INFO)<<"smap_[op->tuple][0].size(): "<< smap_[op->tuple][0].size();
    //LOG(INFO)<<"op->index: "<< op->index << " op->tuple: " << AsText(op->tuple, false);
    VisitExpr(op->tuple);

    std::vector<Integer> storage_ids;
    std::vector<Integer> max_bytes;  // the max bytes for a storage id
    std::vector<Integer> memory_sizes;  // the memory size for a expr
    storage_ids.push_back(storage_tuple_map_[op->tuple][0][op->index]);
    max_bytes.push_back(storage_tuple_map_[op->tuple][1][op->index]);
    memory_sizes.push_back(storage_tuple_map_[op->tuple][2][op->index]);
    storage_tuple_map_.Set(expr, Array<IntegerArray>({storage_ids, max_bytes, memory_sizes}));
  }

  void VisitExpr_(const LetNode* op) final {
    Expr expr = GetRef<Expr>(op);
    VisitExpr(op->body);
    storage_tuple_map_.Set(expr,smap_[op->body]);
    Expr expr_var = GetRef<Expr>(op->var.operator->());
    VisitExpr(op->value);
    storage_tuple_map_.Set(expr_var,smap_[op->value]);
  }

  //void VisitExpr_(const FunctionNode* op) final {
    // do not recurse into sub function.
  //  LOG(INFO)<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!FunctionNode!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1";
  //}

  void VisitExpr_(const GlobalVarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const OpNode* op) final {
    // Do nothing.
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

 private:
  Map<Expr, Array<IntegerArray>> smap_;// 
  Map<Expr, Array<IntegerArray>> storage_tuple_map_;// 
  Function func_;
  std::string c_name_; 
  int tuple_id_;
};

class StorageAllocaBaseVisitor : public ExprVisitor {
 public:

  std::string device_name;
  int io_num_=0;
  /* \brief The expr include input expr, output expr , related "reshape" expr and so on */ 
  // for riscv
  std::map <Expr, int32_t> io_expr_; 
  //tvm::relay::Expr expr_;  //func->body
  // run the visitor on a function.
  void Run(const Function& func) {
    
    for (Var param : func->params) {     
      CreateToken(param.operator->(), false);
      if (device_name =="riscv"){
        Expr expr = GetRef<Expr>(param.operator->());
        io_expr_.insert(std::pair<Expr, int32_t>(expr, io_num_));
        auto type_node = expr->checked_type().as<TupleTypeNode>();
        if (type_node) {
          io_num_ += type_node->fields.size();
        }
        else{
          io_num_++;
        }
      }
    }

    for (StorageToken* tok : GetToken(func->body)) {
      tok->ref_counter += 1;
    }
  }

  void VisitExpr_(const ConstantNode* op) final {
     this->CreateToken(op, false); }

  void VisitExpr_(const VarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const FunctionNode* op) final {
    // do not recurse into sub function.
  }

  void VisitExpr_(const GlobalVarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const OpNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const TupleNode* op) final {
    //LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!! Go into tuple !!!!!!!!!!!!!!!!!!!!!!!" ;
    std::vector<StorageToken*> fields;
    for (Expr field : op->fields) {
      auto tokens = GetToken(field);
      fields.insert(fields.end(), tokens.begin(), tokens.end());
    }
    token_map_[op] = fields;
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    const auto& tok = GetToken(op->tuple);
    ICHECK_LT(static_cast<size_t>(op->index), tok.size());
    token_map_[op] = {tok[op->index]};
  }

  void VisitExpr_(const IfNode* op) final { LOG(FATAL) << "if is not supported."; }

  void VisitExpr_(const LetNode* op) final {
    auto token = GetToken(op->value);
    token_map_[op->var.operator->()] = token;
    token_map_[op] = GetToken(op->body);
  }
  void debug(){
    LOG(INFO)<< "device_name: " << device_name;
    LOG(INFO)<< "io_num_: "<< io_num_;
    LOG(INFO)<< "size(io_expr_): "<< io_expr_.size();
    
    for (const auto& kv : token_map_) {
      Expr expr_op = GetRef<Expr>(kv.first);
      if (expr_op->IsInstance<CallNode>()){
        const CallNode* call_node = expr_op.as<tvm::relay::CallNode>();
        if (call_node->op.as<OpNode>()){
          auto callop_node = call_node->op.as<OpNode>();
          std::string callop_name = tvm::runtime::GetRef<Op>(callop_node)->name;
          LOG(INFO)<< "op_name: " << callop_name;
        }
        else if(call_node->op.as<FunctionNode>()){
          Function func_node = GetRef<Function>(call_node->op.as<FunctionNode>());
          std::string global_symbol = func_node->GetAttr<String>(tvm::attr::kGlobalSymbol).value();
          LOG(INFO)<< "kComposite: " << global_symbol;
        }
      }
      else{
        LOG(INFO)<< "other node: "; //<< AsText(call,false);
      }
      LOG(INFO)<<"(kv.first)->checked_type(): "<<(kv.first)->checked_type()->GetTypeKey();
      if (const auto* ttype = (kv.first)->checked_type().as<TensorTypeNode>()){
        auto shape = tvm::relay::backend::GetIntShape(ttype->shape);
        if (shape.size()==0){
          LOG(INFO) << "(kv.first)->checked_type() shape: is zero" ;
        }
        else if (shape.size()==1){
          LOG(INFO) << "(kv.first)->checked_type() shape: " << shape[0];
        }
        else if (shape.size()==2){
          LOG(INFO) << "(kv.first)->checked_type() shape: " << shape[0] << ", "<< shape[1];
        }
        else if (shape.size()==3){
          LOG(INFO) << "(kv.first)->checked_type() shape: " << shape[0] << ", "<< shape[1] << ", "<< shape[2];
        }
        else if (shape.size()==4){
          LOG(INFO) << "(kv.first)->checked_type() shape: " << shape[0] << ", "<< shape[1] << ", "<< shape[2] << ", "<< shape[3];
        }
        else
        {for (size_t i = 0; i < shape.size(); i++) LOG(INFO) << "(kv.first)->checked_type() shape:" << shape[i];}
      }
      else if (const auto* tutype = (kv.first)->checked_type().as<TupleTypeNode>()){
        int s = 0;
        for (auto field : tutype->fields) {
          const auto* tmptype = field.as<TensorTypeNode>();
          auto shape = tvm::relay::backend::GetIntShape(tmptype->shape);
          if (shape.size()==0){
            LOG(INFO) << "(kv.first)->checked_type() shape: is zero" ;
          }
          else if (shape.size()==1){
            LOG(INFO) << "(kv.first)->checked_type() shape: " << shape[0];
          }
          else if (shape.size()==2){
            LOG(INFO) << "(kv.first)->checked_type() shape: " << shape[0] << ", "<< shape[1];
          }
          else if (shape.size()==3){
            LOG(INFO) << "(kv.first)->checked_type() shape: " << shape[0] << ", "<< shape[1] << ", "<< shape[2];
          }
          else if (shape.size()==4){
            LOG(INFO) << "(kv.first)->checked_type() shape: " << shape[0] << ", "<< shape[1] << ", "<< shape[2] << ", "<< shape[3];
          }
          else
          {for (int i = 0; i < shape.size(); i++) LOG(INFO) << "(kv.first)->checked_type() shape:" << shape[i];}
          s++;
        }
      }
      for (StorageToken* tok : kv.second) {
          LOG(INFO)<<"storage_id: " <<tok->storage_id <<" device_type: " <<tok->device_type 
           <<" ref_counter: " <<tok->ref_counter<< " max_bytes: " <<tok->max_bytes;
      }
    }
    LOG(INFO)<< "********************************io_expr_******************************"<< io_expr_.size();
    for (auto kv: io_expr_){
      Expr expr_op = kv.first;
      LOG(INFO)<< "io_expr_ index: "<<  kv.second;
      if (expr_op->IsInstance<CallNode>()){
        const CallNode* call_node = expr_op.as<tvm::relay::CallNode>();
        if (call_node->op.as<OpNode>()){
          auto callop_node = call_node->op.as<OpNode>();
          std::string callop_name = tvm::runtime::GetRef<Op>(callop_node)->name;
          LOG(INFO)<< "op_name: " << callop_name;
        }
        else if(call_node->op.as<FunctionNode>()){
          Function func_node = GetRef<Function>(call_node->op.as<FunctionNode>());
          std::string global_symbol = func_node->GetAttr<String>(tvm::attr::kGlobalSymbol).value();
          LOG(INFO)<< "kComposite: " << global_symbol;
        }
      }
      else{
        LOG(INFO)<< "other node: "; //<< AsText(call,false);
      }
    }
  }


 protected:
  /*! \brief internal token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*> > token_map_;

  /*!
   * \brief Get the necessary token.
   * \param expr The expression.
   * \return The corresponding token.
   */
  const std::vector<StorageToken*>& GetToken(const Expr& expr) {
    if (expr->IsInstance<CallNode>()){
      //auto op = Downcast<Call>(expr);
      const CallNode* call_node = expr.as<tvm::relay::CallNode>();
      if (const auto* op_node = call_node->op.as<OpNode>()) { 
      //const auto* op_node = call->op.as<OpNode>();
      const auto op_name = GetRef<Op>(op_node)->name;
      //LOG(INFO) << "op_name: " << op_name;
      if (op_name == "relay.op.annotation.simulated_quantize" || op_name == "annotation.stop_fusion" || op_name == "annotation.cast_hint")
        return GetToken(call_node->args[0]);
      }
    }
    this->VisitExpr(expr);
    auto it = token_map_.find(expr.operator->());
    ICHECK(it != token_map_.end());
    return it->second;
  }
  /*!
   * \brief Populate the token map to set op's tokens
   * \param op The node to be processed.
   * \param can_realloc Whether we can re-allocate the memory.
   */
  virtual void CreateToken(const ExprNode* op, bool can_realloc) = 0;
};

class StorageAllocaInit : protected StorageAllocaBaseVisitor {
 public:
  explicit StorageAllocaInit(support::Arena* arena) : arena_(arena) {}

  /*! \return The internal token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*> > GetInitTokenMap(
      const Function& func) {
    node_device_map_ = CollectDeviceInfo(func);
    this->Run(func);
    //debug();
    return std::move(token_map_);
  }

 protected:
  using StorageAllocaBaseVisitor::VisitExpr_;

  void CreateToken(const ExprNode* op, bool can_realloc) final {
    ICHECK(!token_map_.count(op));
    ///////////yuanyue
    //LOG(INFO) << op->checked_type()->GetTypeKey();
    //auto ishape = tvm::relay::backend::GetShape(op->checked_type());
    //LOG(INFO) << "the input ishape is as follows:";
    //for (int i = 0; i < ishape.size(); i++) LOG(INFO) << "input ishape:" << ishape[i];
    auto tmp = GetRef<Expr>(op);
    // auto tmp1=node_device_map_.count(GetRef<Expr>(op));
    // auto tmp2=node_device_map_[GetRef<Expr>(op)]->value ;
    //////////yuanyue
   
    std::vector<StorageToken*> tokens;
    int device_type =
        node_device_map_.count(GetRef<Expr>(op)) ? node_device_map_[GetRef<Expr>(op)]->value : 0;
    if (const auto* tuple_type = op->checked_type().as<TupleTypeNode>()) {
      for (Type t : tuple_type->fields) {
        const auto* ttype = t.as<TensorTypeNode>();
        ICHECK(ttype);
        StorageToken* token = arena_->make<StorageToken>();
        token->ttype = ttype;
        token->device_type = device_type;
        tokens.push_back(token);
      }
    } else {
      const auto* ttype = op->checked_type().as<TensorTypeNode>();
      ICHECK(ttype);
      StorageToken* token = arena_->make<StorageToken>();
      token->ttype = ttype;
      token->device_type = device_type;
      tokens.push_back(token);
    }
    token_map_[op] = tokens;
  }

  void VisitExpr_(const CallNode* op) final {
    // create token for the call node.
    /*
    if (const auto* op_node = op->op.as<OpNode>()) { 
      const auto op_name = GetRef<Op>(op_node)->name;
      LOG(INFO) << "op_name: " << op_name;

      if (op_name == "relay.op.annotation.simulated_quantize" || op_name == "annotation.stop_fusion" || op_name == "annotation.cast_hint")
        return VisitExpr(op->args[0]);
    }
    else if (const auto* func = op->op.as<FunctionNode>()) {
      LOG(INFO) << "func ...: " << func;
    }
    else{
      LOG(INFO) << op->checked_type()->GetTypeKey();
      LOG(INFO) << op->op->checked_type()->GetTypeKey();
    }
    */

    CreateToken(op, true);
    // for each input, visit argument token.
    for (Expr arg : op->args) {

      for (StorageToken* tok : GetToken(arg)) {
        tok->ref_counter += 1;
      }
    }
  }

 private:
  // allocator
  support::Arena* arena_;
  Map<Expr, Integer> node_device_map_;
};

class StorageAllocator : public StorageAllocaBaseVisitor {
 public:
  /*!
   * \return totoal number of bytes allocated
   */
  size_t TotalAllocBytes() const {
    size_t total = 0;
    for (const auto* p : data_) {
      total += p->max_bytes;
    }
    return total;
  }

  /*!
   * \return temporary_data_offset_
   */
  std::map<int, size_t>  GetDataOffset() const {
    return temporary_data_offset_;
  }

    /*!
   * \return temporary_data_offset_
   */
  std::map<int, size_t>  GetDataStorage() const {
    return temporary_data_storage_;
  }

    /*!
   * \return memory_used , for the next offset
   */
  size_t GetTotalMemory() const {
    return memory_used_;
  }

    /*!
   * \return Not Quantized Expr , for riscv output offset
   */
  Expr GetNotQuantizedExpr(Expr expr , bool flag=false){
    if (expr->IsInstance<CallNode>()){
      const CallNode* call_node = expr.as<tvm::relay::CallNode>();        
      //auto op = Downcast<Call>(expr);
      if (const auto* op_node = call_node->op.as<OpNode>()) { 
        //const auto* op_node = call->op.as<OpNode>();
        const auto op_name = GetRef<Op>(op_node)->name;
        //LOG(INFO) << "op_name: " << op_name;
        if (op_name == "relay.op.annotation.simulated_quantize" || op_name == "annotation.stop_fusion" || op_name == "annotation.cast_hint")
          return GetNotQuantizedExpr(call_node->args[0], flag);
        if (flag){
          if (op_name == "reshape" || op_name == "expand_dims" || op_name == "squeeze"){            
            Expr expr_arg = GetNotQuantizedExpr(call_node->args[0]);
            size_t type = aid_dtype_map_[expr][0][0];
            size_t type_arg = aid_dtype_map_[expr_arg][0][0];
            size_t lable = aid_dtype_map_[expr][1][0];
            size_t lable_arg = aid_dtype_map_[expr_arg][1][0];
            if ((type == type_arg) && (lable == lable_arg)){
              //LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!right!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
              return GetNotQuantizedExpr(call_node->args[0], flag);
            }
            else
              return expr;
          }  
        }
      }
    }
    return expr;
  }

  // Run storage allocation for a function.
  Map<Expr, Array<IntegerArray> > Plan(const Function& func, Map<Expr, Array<IntegerArray>> aid_dtype_map, size_t total_memory_used, std::string c_name) {
    device_name = c_name;
    io_expr_.clear();
    temporary_data_offset_.clear();
    temporary_data_storage_.clear();
    aid_dtype_map_.clear();
    prototype_ = StorageAllocaInit(&arena_).GetInitTokenMap(func);
    memory_used_ = total_memory_used;
    aid_dtype_map_ = aid_dtype_map;
    //LOG(INFO)<<"AAAAAAAAAAAAAAAAA------plan-----AAAAAAAAAAAAAAA";
    this->Run(func);
    
    //int num = data_.size();
    //LOG(INFO)<< "num: " << num;

    // The value of smap contains two integer arrays where the first array
    // contains the planned storage ids and the second holds the max_bytes.
    Map<Expr, Array<IntegerArray> > smap;
    int num_annotated_nodes = 0;
    int num_nodes = 0;

    std::vector<int> io_storage_ids;
    Expr func_body=GetNotQuantizedExpr(func->body); //the output op
    
    if (device_name =="riscv"){
      io_expr_.insert(std::pair<Expr, int32_t>(func_body, io_num_));
      /*
      auto type_node = func_body->checked_type().as<TupleTypeNode>();
      if (type_node) {
        io_num_ += type_node->fields.size();
      }
      else{
        io_num_++;
      }
      */
    }
    
    GetIOReuseExprfromOutput(func_body);
    // bool flag_output_tuplenode = false; // if riscv output is tuplenode, it will true

    //debug();

    for (const auto& kv : token_map_) {
      std::vector<Integer> storage_ids;
      std::vector<Integer> max_bytes;  // the max bytes for a storage id
      std::vector<Integer> memory_sizes;  // the memory size for a expr
      //std::vector<Integer> expr_type;
 
      std::vector<const tvm::relay::TensorTypeNode *> tensortypes;
      size_t vector_index = 0;
      if (const auto* ttype = (kv.first)->checked_type().as<TensorTypeNode>()){
        tensortypes.push_back(ttype);
      }
      else if (const auto* tutype = (kv.first)->checked_type().as<TupleTypeNode>()){
        for (auto field : tutype->fields) {
          const auto* tmptype = field.as<TensorTypeNode>();
          tensortypes.push_back(tmptype);
        }
      }
      Expr expr_op = GetRef<Expr>(kv.first);
      Array<IntegerArray> aid_infos;
      if (aid_dtype_map_.count(expr_op)){
        for (int i =0;i<aid_dtype_map_[expr_op][0].size();i++){
          aid_infos.insert(aid_infos.end(),{aid_dtype_map_[expr_op][0][i], aid_dtype_map_[expr_op][1][i]});
        }
      }
      else{
        LOG(FATAL) << "aid_info does not support : " << AsText(expr_op, false);
      }
   
      /*
      if (device_name =="riscv"){
        LOG(INFO) << " !!!!!!!!!!!!!!!!!!!!!!!!dtype_map_!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" ;
        for (int t = 0; t<aid_dtype_map_[expr_op][0].size() ; t++){
          LOG(INFO) <<"dtype: " << aid_dtype_map_[expr_op][0][t] << " name: " << aid_dtype_map_[expr_op][1][t]  << " type: " << kv.first->GetTypeKey();
        }
        Expr call = GetRef<Expr>(kv.first);
        if (call->IsInstance<CallNode>()){
          const CallNode* call_node = call.as<tvm::relay::CallNode>();
          if (call_node->op.as<OpNode>()){
            auto callop_node = call_node->op.as<OpNode>();
            std::string callop_name = tvm::runtime::GetRef<Op>(callop_node)->name;
            LOG(INFO)<< "op_name: " << callop_name;
          }
          else if(call_node->op.as<FunctionNode>()){
            Function func_node = GetRef<Function>(call_node->op.as<FunctionNode>());
            std::string global_symbol = func_node->GetAttr<String>(tvm::attr::kGlobalSymbol).value();
            LOG(INFO)<< "kComposite: " << global_symbol;
          }
        }
        else{
          LOG(INFO)<< "other node: "; 
        }
        LOG(INFO) << AsText(call,false);
      }
      */
      
      /*
      // if riscv output is tuplenode leave it to the postprocress
      if (device_name == "riscv" && expr_op->IsInstance<TupleNode>() && io_expr_.find(expr_op) != io_expr_.end()){
        flag_output_tuplenode = true;
        continue;
      }
      */
          
      for (StorageToken* tok : kv.second) {
        if (tok->device_type) {
          num_annotated_nodes++;
        }
        num_nodes++;

        if (io_expr_.find(expr_op) != io_expr_.end()){
          size_t io_id=data_.size()+io_expr_[expr_op]+vector_index;
          storage_ids.push_back(io_id);
          io_storage_ids.push_back(io_id);
        }
        else{       
          storage_ids.push_back(tok->storage_id);
        }
        max_bytes.push_back(tok->max_bytes);
        if (vector_index<tensortypes.size()){
          size_t size=GetMemorySize(tensortypes[vector_index], aid_infos[vector_index]);
          memory_sizes.push_back(size);
          vector_index++;
        }
        else{
          LOG(FATAL) << "in plan memory, the output index ( " << vector_index << " ) is  out of expr's output size ( " << tensortypes.size() << " ).";
        }
        /*
        //LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
        if (device_name =="riscv"){
          LOG(INFO)<<"storage_id: " <<storage_ids[vector_index-1] <<" device_type: " <<tok->device_type 
             <<" ref_counter: " <<tok->ref_counter<< " max_bytes: " <<tok->max_bytes<< " size_2: " << memory_sizes[vector_index-1];
        }
        */

      }
      
      smap.Set(expr_op, Array<IntegerArray>({storage_ids, max_bytes, memory_sizes}));
    }

    /////////postprocess
    int total_id = data_.size();
    if (device_name =="func"){
      /*LOG(INFO)<<"***********************************************************************";
      for (auto it : smap){
        for (int i =0 ; i< it.second[0].size(); i++){
          LOG(INFO)<<"storage_id: " <<it.second[0][i] << " max_bytes: " <<it.second[1][i]<< " size_2: " << it.second[2][i];
        }
      }*/
      TuplePlan tupleplan(func,smap,data_.size());
      smap = tupleplan.Run();
      total_id = tupleplan.GetTuoleID();
      /*
      LOG(INFO)<<"***********************************************************************";
      for (auto it : smap){
        Expr expr_op = it.first;
        if (expr_op->IsInstance<CallNode>()){
          const CallNode* call_node = expr_op.as<tvm::relay::CallNode>();
          if (call_node->op.as<OpNode>()){
            auto callop_node = call_node->op.as<OpNode>();
            std::string callop_name = tvm::runtime::GetRef<Op>(callop_node)->name;
            LOG(INFO)<< "op_name: " << callop_name;
          }
          else if(call_node->op.as<FunctionNode>()){
            Function func_node = GetRef<Function>(call_node->op.as<FunctionNode>());
            std::string global_symbol = func_node->GetAttr<String>(tvm::attr::kGlobalSymbol).value();
            LOG(INFO)<< "kComposite: " << global_symbol;
          }
        }
        else{
          LOG(INFO)<< "other node: ";
          if (expr_op->GetTypeKey() == "relay.Var"){
            LOG(INFO)<< AsText(expr_op,false);
          }
          else if (expr_op->GetTypeKey() == "relay.Tuple"){
            LOG(INFO)<< "relay.Tuple";
          }
          else{
            LOG(INFO)<< expr_op->GetTypeKey();
          }
        }
        for (int i =0 ; i< it.second[0].size(); i++){
          LOG(INFO)<<"storage_id: " <<it.second[0][i] << " max_bytes: " <<it.second[1][i]<< " size_2: " << it.second[2][i];
        }
      }
      */
    }

    // if riscv output is tuplenode, deal with it in this part
    /*
    if(flag_output_tuplenode){
      std::vector<Integer> storage_ids;
      std::vector<Integer> max_bytes;  // the max bytes for a storage id
      std::vector<Integer> memory_sizes;  // the memory size for a expr
      if (func_body->IsInstance<TupleNode>()) {
        auto tuple = Downcast<Tuple>(func_body);
        for (auto field : tuple->fields) {
          Expr expr_branch=GetNotQuantizedExpr(field);
          if (io_expr_.find(expr_branch) != io_expr_.end()){
            for (int i =0; i < smap[expr_branch][0].size(); i++){
              storage_ids.push_back(smap[expr_branch][0][i]);
              max_bytes.push_back(smap[expr_branch][1][i]);
              memory_sizes.push_back(smap[expr_branch][2][i]);
              //LOG(INFO) << "storage_ids: "<< storage_ids[i] << " max_bytes: " << max_bytes[i] << " memory_sizes: " << memory_sizes[i];
            }
          }
          else{
            LOG(INFO) << AsText(expr_branch,false);
            LOG(FATAL) << "riscv output branch is not in  io_expr_ , please checkout";
          }
        }
        smap.Set(func_body, Array<IntegerArray>({storage_ids, max_bytes, memory_sizes}));
      }
      else{
        LOG(FATAL) << "riscv output is not tuplenode, its type is " << func_body->GetTypeKey()<< ", please checkout";
      }
    }
    */
    for (auto it : smap) {
      for (size_t i = 0; i< it.second[0].size(); i++) {
        //LOG(INFO)<<"kv"<<kv.size();
        size_t output_size = it.second[1][i];
        int storage_id = it.second[0][i];
        //LOG(INFO) << "storage_id: " << storage_id << " output_size: "<< output_size; 
        if (temporary_data_storage_.find(storage_id) == temporary_data_storage_.end()) {
          temporary_data_storage_.insert(std::pair<int,size_t>(storage_id, output_size));
        }
        else if (output_size > temporary_data_storage_[storage_id]) {
          //LOG(INFO)<< "!!!!!!!!!!!!!!!!!!not max!!!!!!!!!!!!!!!!!!!!!!";
          temporary_data_storage_[storage_id] = output_size;
        }
      }
    }
    /*
    // offset for func 
    if (1){
      for (int i =0 ; i<io_storage_ids.size(); i++){
       temporary_data_offset_.insert(std::pair<int,size_t>(io_storage_ids[i], 0));
    }
    */
    if (device_name =="func"){
      for (int i=0; i< total_id;i++) {
        if (temporary_data_storage_.find(i) != temporary_data_storage_.end()) {
          if (temporary_data_offset_.find(i) == temporary_data_offset_.end()) {
            temporary_data_offset_.insert(std::pair<int,size_t>(i, memory_used_));
            memory_used_+=temporary_data_storage_[i];
          }
        }
      }
    }

    //LOG(INFO) << "*******************************************************************";
    //LOG(INFO) << smap.size();
    //for (auto it : temporary_data_storage_)
    //  LOG(INFO) << "storage_id " << it.first << " data_size " << it.second;

    // Either all or none of the nodes should be annotated.
    if (num_annotated_nodes != 0 && num_annotated_nodes != num_nodes) {
      LOG(FATAL) << num_annotated_nodes << " out of " << num_nodes
                 << "expressions are assigned with virtual device types. Either all "
                    "or none of the expressions are expected to be annotated.";
    }
    
    return smap;
  }

 protected:
  using StorageAllocaBaseVisitor::VisitExpr_;

  // override create token by getting token as prototype requirements.
  void CreateToken(const ExprNode* op, bool can_realloc) final {
    Expr expr = GetRef<Expr>(op);
    Array<IntegerArray> aid_infos;
    if (aid_dtype_map_.count(expr)){
      for (size_t i =0;i<aid_dtype_map_[expr][0].size();i++){
        aid_infos.insert(aid_infos.end(),{aid_dtype_map_[expr][0][i], aid_dtype_map_[expr][1][i]});
      }
    }
    //else if(device_name == "riscv"){ // for ConstantNode
    //  aid_info = {4,1};
    //}
    else{
      LOG(FATAL) << "aid_info does not support : " << AsText(expr, false);
    }
    int index = 0;
    ICHECK(!token_map_.count(op));
    auto it = prototype_.find(op);
    ICHECK(it != prototype_.end());
    std::vector<StorageToken*> tokens;
    for (StorageToken* tok : it->second) {
      if (can_realloc) {
        tokens.push_back(Request(tok, aid_infos[index]));
      } else {
        // Allocate a new token,
        StorageToken* allocated_tok = Alloc(tok, GetMemorySize(tok, aid_infos[index]));
        allocated_tok->device_type = tok->device_type;
        // ensure it never get de-allocated.
        allocated_tok->ref_counter += 1;
        //allocated_tok->expr_type = expr_type_flag;
        tokens.push_back(allocated_tok);
      }
      index++;
    }
    token_map_[op] = tokens;
  }

  // Mark op to reuse the input_token
  // tie the two memories together
  void ReuseInputToken(const ExprNode* op, StorageToken* input_token) {
    ICHECK(!token_map_.count(op));
    auto it = prototype_.find(op);
    ICHECK(it != prototype_.end());
    ICHECK_EQ(it->second.size(), 1U);
    StorageToken* prototype = it->second[0];
    // add the reference counter of the output
    // so the input token can only be deleted after references
    // to both are expired
    input_token->ref_counter += prototype->ref_counter;
    /*
    Expr call = GetRef<Expr>(op);
    IntegerArray aid_info;
    if (aid_dtype_.count(call)){
      aid_info = aid_dtype_[call];
    }
    else{
      LOG(FATAL) << "aid_info does not support : " << AsText(call, false);
    }
    size_t size = GetMemorySize(input_token, aid_info);
    input_token->max_bytes = std::max(size, input_token->max_bytes);
    */
    // reuse the input token
    token_map_[op] = {input_token};
  }

  // if the op/function token is same as its param or not
  // case 1("func") : the function only include quantized ops
  // case 2("riscv"): the op only include "reshape" ,"expand_dims", "squeeze" , and their dtype && device are both same
  bool IsSameToken(const CallNode* op){
    if (device_name == "func" && op->op.as<FunctionNode>()){
      Function funct = GetRef<Function>(op->op.as<FunctionNode>());
      std::string compiler = funct->GetAttr<String>(attr::kCompiler).value();
      if (compiler.substr(0, 3) == "dla")  { // dla func
        return false;
      }
      Expr expr = GetNotQuantizedExpr(funct->body , true);
      //LOG(INFO)<<"expr: "<< expr;
      for (Var param : funct->params) { 
        //auto expr2=GetRef<Expr>(param.operator->());
        //LOG(INFO)<<"expr2: "<< expr2;    
        if (expr == GetRef<Expr>(param.operator->())){
          //LOG(INFO) << "***************************right*****************************";
          return true;
        }
      }
    }
    else if (device_name == "riscv"){
      Expr expr = GetRef<Expr>(op);
      if (const auto* op_node = op->op.as<OpNode>()) { 
        const auto op_name = GetRef<Op>(op_node)->name;
        if (op_name == "reshape" || op_name == "expand_dims" || op_name == "squeeze"){
          //LOG(INFO) << "***************************riscv right*****************************";
          Expr expr_arg = GetNotQuantizedExpr(op->args[0]);
          int type = aid_dtype_map_[expr][0][0];
          int type_arg = aid_dtype_map_[expr_arg][0][0];
          int lable = aid_dtype_map_[expr][1][0];
          int lable_arg = aid_dtype_map_[expr_arg][1][0];
          if ((type == type_arg) && (lable == lable_arg)){
            return true;
          }
        }
      }
    }
    return false;
  }

  void GetIOReuseExprfromArgs(const CallNode* op){
    if (device_name == "riscv"){
      Expr expr_op = GetRef<Expr>(op);
      Expr arg_expr = GetNotQuantizedExpr(op->args[0]);
      if(io_expr_.find(arg_expr) != io_expr_.end()){
        io_expr_.insert(std::pair<Expr, int32_t>(expr_op, io_expr_[arg_expr]));
      }
    }
  }

  void GetIOReuseExprfromOutput(Expr expr){
    if (device_name == "riscv"){
      if (expr->IsInstance<CallNode>()){
        //auto op = Downcast<Call>(expr);
        const CallNode* call_node = expr.as<tvm::relay::CallNode>();
        Expr arg_expr = GetNotQuantizedExpr(call_node->args[0]);
        if (IsSameToken(call_node)){
          if(io_expr_.find(expr) != io_expr_.end()){
            io_expr_.insert(std::pair<Expr, int32_t>(arg_expr, io_expr_[expr]));
          }
          GetIOReuseExprfromOutput(arg_expr);
        }
      }
      else if(expr->IsInstance<TupleNode>()){
        io_expr_.insert(std::pair<Expr, int32_t>(expr, io_num_));
        auto tuple = Downcast<Tuple>(expr);
        for (auto field : tuple->fields) {
          Expr expr_branch = GetNotQuantizedExpr(field);
          io_expr_.insert(std::pair<Expr, int32_t>(expr_branch, io_num_));
          auto type_node = expr_branch->checked_type().as<TupleTypeNode>();
          if (type_node) {
            io_num_ += type_node->fields.size();
          }
          else{
            io_num_++;
          }
          GetIOReuseExprfromOutput(expr_branch);
        }
      } 
      else if (expr->IsInstance<VarNode>()){
        return;
      }
      else {
        LOG(FATAL)<< "The related expr in plan memory does not support the type: " << expr->GetTypeKey();
      }
    }
  }

  // The call map
  void VisitExpr_(const CallNode* op) final {
    // LOG(INFO) << GetRef<Expr>(op);
    std::vector<StorageToken*> args;
    // for each input, visit argument token.
    for (Expr arg : op->args) {
      for (StorageToken* tok : GetToken(arg)) {
        args.push_back(tok);
      }
    }

    // create token for the call node.
    if (IsSameToken(op)) {
      ICHECK_EQ(args.size(), 1U);
      ReuseInputToken(op, args[0]);
      GetIOReuseExprfromArgs(op);
    } else {
      // create token for the call node.
      CreateToken(op, true);
    }

    // check if there is orphaned output that can be released immediately.
    for (StorageToken* tok : token_map_.at(op)) {
      CheckForRelease(tok);
    }
    for (StorageToken* tok : args) {
      tok->ref_counter -= 1;
      CheckForRelease(tok);
    }
  }
  /*!
   * \brief ceil(size/word_size) to get number of words.
   * \param size The original size.
   * \param word_size The element size.
   */
  static size_t DivRoundUp(size_t size, size_t word_size) {
    return (size + word_size - 1) / word_size;
  }

  /*!
   * \brief Get the memory requirement.
   * \param prototype The prototype token.
   * \return The required memory size.
   */
  size_t GetMemorySize(StorageToken* prototype, IntegerArray aid_info) {
    const TensorTypeNode* ttype = prototype->ttype;
    return GetMemorySize(ttype, aid_info);
  }

  size_t GetMemorySize(const TensorTypeNode* ttype, IntegerArray aid_info) {
    size_t size=1;
    size_t type_bytes = GetTypeBytes(aid_info[0]);
    
    if (aid_info[1]==1){  //riscv
      size=GetMemorySize1(ttype, type_bytes);
    }
    else if (aid_info[1]==2){  //dla
      size=GetMemorySize2(ttype);
    }
    else if (aid_info[1]==3){  //host
      size=GetMemorySize3(ttype, type_bytes);
    }
    else{
      LOG(FATAL)<< "GetMemorySize does not support the decive(1 for riscv, 2 for dla, 3 for host): " << aid_info[1];
    }
    return size;
  }

  /*!
   * for riscv 
   */
  size_t GetMemorySize1(StorageToken* prototype , size_t type_bytes) {
    const TensorTypeNode* ttype = prototype->ttype;
    return GetMemorySize1(ttype,type_bytes);
  }
  size_t GetMemorySize1(const TensorTypeNode* ttype ,size_t type_bytes) {
    //const TensorTypeNode* ttype = prototype->ttype;
    ICHECK(ttype != nullptr);
    size_t size = 1;
    
    auto shape = tvm::relay::backend::GetIntShape(ttype->shape);
    //size_t type_bytes=std::ceil(ttype->dtype.bits() * ttype->dtype.lanes()/8);
    //size_t type_bytes=DivRoundUp(ttype->dtype.bits() * ttype->dtype.lanes(), 8);  
    for (auto dim : shape){
      size*=dim;
    }
    //LOG(INFO)<< "type_bytes : " << type_bytes<< " size11 : " << size;
    size=(std::ceil(size*type_bytes/32.0))*32;
    //LOG(INFO)<< "type_bytes : " << type_bytes<< " size12 : " << size;
    return size;
  }
    /*!
   *for dla
   */
  size_t GetMemorySize2 (StorageToken* prototype) {
    const TensorTypeNode* ttype = prototype->ttype;
    return GetMemorySize2(ttype);
  }

  size_t GetMemorySize2 (const TensorTypeNode* ttype) {
    ICHECK(ttype != nullptr);
    size_t size = 1;
    auto shape = tvm::relay::backend::GetIntShape(ttype->shape);
    size_t type_bytes = 1;
    if (shape.size() == 0){
      size = (std::ceil(size/32.0))*32;
    }
    else{
      int shapesize_1 = shape.size()-1;
      for (int i = 0 ; i < shapesize_1; i++){
        size*=shape[i];
      }
      size = size*(std::ceil(type_bytes*shape[shapesize_1]/32.0))*32;
    }
    return size;
  }

  size_t GetMemorySize3(StorageToken* prototype , size_t type_bytes) {
    const TensorTypeNode* ttype = prototype->ttype;
    return GetMemorySize3(ttype,type_bytes);
  }
  size_t GetMemorySize3(const TensorTypeNode* ttype ,size_t type_bytes) {
    //const TensorTypeNode* ttype = prototype->ttype;
    ICHECK(ttype != nullptr);
    size_t size = 1;
    auto shape = tvm::relay::backend::GetIntShape(ttype->shape);

    for (auto dim : shape){
      size*=dim;
    }
    size=size*type_bytes;
    return size;
  }
  
  size_t GetTypeBytes(Integer type_index ){
    size_t type_bytes = 4;
    if (type_index == 3){ // float32
      type_bytes = 4;
    }
    else if(type_index == 5){ // int8
      type_bytes = 1;
    }
    else if(type_index == 1){ // int32
      type_bytes = 4;
    }
    else if(type_index == 2){ // uint32
      type_bytes = 4;
    }
    else if(type_index == 4){ // float16
      type_bytes = 2;
    }
    else{
      LOG(FATAL)<< "GetTypeBytes does not support the type_index: " << type_index;
    }
    return type_bytes;
  }
  /*!
   * \brief Request a storage token for a given prototype.
   * \param prototype. The prototype storage token.
   * \return The result token.
   */
  StorageToken* Request(StorageToken* prototype, IntegerArray aid_info) {
    // calculate the size;
    size_t size = GetMemorySize(prototype, aid_info);
    // search memory block in [size / match_range_, size * match_range_)
    if (match_range_ == 0) {
      return this->Alloc(prototype, size);
    }
    auto begin = free_.lower_bound(size / match_range_);
    auto mid = free_.lower_bound(size);
    auto end = free_.upper_bound(size * match_range_);
    // search for memory blocks larger than requested
    for (auto it = mid; it != end; ++it) {
      StorageToken* tok = it->second;
      if (tok->device_type != prototype->device_type) continue;
      ICHECK_EQ(tok->ref_counter, 0);
      // Use exect matching strategy
      tok->max_bytes = std::max(size, tok->max_bytes);
      tok->ref_counter = prototype->ref_counter;
      // tok->expr_type = expr_type_flag;
      // find a exact match, erase from map and return
      free_.erase(it);
      return tok;
    }
    // then search for memory blocks smaller than requested space
    for (auto it = mid; it != begin;) {
      --it;
      StorageToken* tok = it->second;
      if (tok->device_type != prototype->device_type) continue;
      ICHECK_EQ(tok->ref_counter, 0);
      // Use exect matching strategy
      tok->max_bytes = std::max(size, tok->max_bytes);
      tok->ref_counter = prototype->ref_counter;
      //tok->expr_type = expr_type_flag;
      // erase from map and return
      free_.erase(it);
      return tok;
    }
    // cannot find anything return a new one.
    return this->Alloc(prototype, size);
  }
  /*!
   * \brief Allocate a storage token by consuming prototype
   * \param prototype The prototype token.
   * \param size The size of memory being requested.
   */
  StorageToken* Alloc(StorageToken* prototype, size_t size) {
    prototype->max_bytes = size;
    prototype->storage_id = static_cast<int64_t>(data_.size());
    //prototype->expr_type = expr_type_flag;
    data_.push_back(prototype);
    return prototype;
  }
  /*!
   * \brief Check if we can release token.
   * \param tok The token to be released.
   */
  void CheckForRelease(StorageToken* tok) {
    ICHECK_GE(tok->storage_id, 0);
    ICHECK_GE(tok->ref_counter, 0);
    if (tok->ref_counter == 0) {
      free_.insert({tok->max_bytes, tok});
    }
  }

 private:
  
  // allocator
  support::Arena arena_;
  // scale used for rough match
  size_t match_range_{16};
  // free list of storage entry
  std::multimap<size_t, StorageToken*> free_;
  // all the storage resources available
  std::vector<StorageToken*> data_;
  /*! \brief internal prototype token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*> > prototype_;

  // tmp: map between memory block and its' offset
  std::map<int, size_t> temporary_data_offset_;
  // tmp: map between memory block and its' size
  std::map<int, size_t> temporary_data_storage_;
  size_t memory_used_{0};

  // key: expr , 
  // value: two array are singly ref_dtypes, dtype_names
  Map<Expr, Array<IntegerArray>> aid_dtype_map_;
};

//Map<Expr, Array<IntegerArray> > GraphPlanMemory(const Function& func) {
//  return StorageAllocator().Plan(func);
//}

//TVM_REGISTER_GLOBAL("relay.backend.GraphPlanMemory").set_body_typed(GraphPlanMemory);

}  // namespace aipu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_GRAPH_PLAN_MEMORY_H_
