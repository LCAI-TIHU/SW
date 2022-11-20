#ifndef TVM_RELAY_BACKEND_CONTRIB_GRAPH_AID_DTYPE_H_
#define TVM_RELAY_BACKEND_CONTRIB_GRAPH_AID_DTYPE_H_


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
 * \file src/relay/backend/contrib/aipu/graph_aid_dtype.h
 * \brief get true dtype
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

// TODO. Add "layout" into DtypeInfo. Maybe add "quant scale" into DtypeInfo
struct DtypeInfo {
  /*! \brief Reference dtype */
  /* ref_dtype
  RINT = 1U,
  RUINT = 2U,
  RFLOAT = 3U,
  RBFLOAT = 4U,
  RINT8 = 5U,
  */
  int ref_dtype{0};
  /*! \brief name (1 for riscv ; 2 for dla; 3 for host)  */
  int dtype_name{0};

};

class DtypeInfoBaseVisitor : public ExprVisitor {
 public:
  // run the visitor on a function.
  void Run(const Function& func) {
    c_name_ = "func_";
    for (Var param : func->params) {  
      Expr expr = GetRef<Expr>(param.operator->());   
      CreateDtypes(expr, "func", true); 
    }
    VisitExpr(func);
  }

  void VisitExpr_(const VarNode* op) final {
    // Do nothing.
  }

  //void VisitExpr_(const FunctionNode* op) final {
    // do not recurse into sub function.
  //}

  void VisitExpr_(const GlobalVarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const OpNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const ConstantNode* op) final {
    Expr expr = GetRef<Expr>(op);
    this->CreateDtypes(expr, "host", true); 
  }

  void VisitExpr_(const TupleNode* op) final {
    //LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!! Go into tuple !!!!!!!!!!!!!!!!!!!!!!!" ;
    std::vector<DtypeInfo*> fields;
    for (Expr field : op->fields) {
      auto tokens = GetDtype(field);
      fields.insert(fields.end(), tokens.begin(), tokens.end());
    }
    dtype_map_[op] = fields;
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    const auto& token = GetDtype(op->tuple);
    ICHECK_LT(static_cast<size_t>(op->index), token.size());
    dtype_map_[op] = {token[op->index]};
  }

  void VisitExpr_(const IfNode* op) final { LOG(FATAL) << "if is not supported."; }

  void VisitExpr_(const LetNode* op) final {
    auto token = GetDtype(op->value);
    dtype_map_[op->var.operator->()] = token;
    dtype_map_[op] = GetDtype(op->body);
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
  /*! \brief internal dtype map */
  std::unordered_map<const ExprNode*, std::vector<DtypeInfo*> > dtype_map_;

  std::string c_name_;
  /*!
   * \brief Get the necessary token.
   * \param expr The expression.
   * \return The corresponding token.
   */
  std::vector<DtypeInfo*>& GetDtype(const Expr& expr) {
    if (expr->IsInstance<CallNode>()){
      //auto op = Downcast<Call>(expr);
      const CallNode* call_node = expr.as<tvm::relay::CallNode>();
      if (const auto* op_node = call_node->op.as<OpNode>()) { 
      //const auto* op_node = call->op.as<OpNode>();
      const auto op_name = GetRef<Op>(op_node)->name;
      //LOG(INFO) << "op_name: " << op_name;
      if (op_name == "relay.op.annotation.simulated_quantize" || op_name == "annotation.stop_fusion" || op_name == "annotation.cast_hint")
        return GetDtype(call_node->args[0]);
      }
    }
    this->VisitExpr(expr);
    auto expr_node = expr.as<ExprNode>();
    auto it = dtype_map_.find(expr_node);
    ICHECK(it != dtype_map_.end()) << "error" << AsText(expr, false);
    return it->second;
  }

  void GetOriginType(const TensorTypeNode *type, int& dy_type){
      if (type->dtype.code() == DataType::kInt)
        dy_type = 1;
      else if (type ->dtype.code() == DataType::kUInt)
        dy_type = 2;
      else if (type ->dtype.code() == DataType::kFloat)
        dy_type = 3;
      else if (type ->dtype.code() == DataType::kBFloat && type->dtype.bits() == 16)
        dy_type = 4;
      else
        LOG(FATAL) << "Datatype not supported ";
    }
  int GetRefName(std::string dtype_name){
    int ref = 0;
    if (dtype_name == "func"){
        ref = 1;//default riscv
      }
      else if (dtype_name == "dla"){
        ref = 2;//dla
      }
      else if (dtype_name == "riscv"){
        ref = 1;//riscv
      }
      else if (dtype_name == "host"){
        ref = 3; //host
      }
      else {
        LOG(FATAL) << "dtype_name: " << dtype_name;
      }
    return ref;
  }
  
  /*!
   * \brief Populate the token Dtypes to set op's Dtypes
   * \param op The node to be processed.
   * \param dtype_name the decive name: "func" "dla" "riscv" "host"
   * \param datatype  RINT = 1U, RUINT = 2U, RFLOAT = 3U, RBFLOAT = 4U, RINT8 = 5U,
   * \param can_reuse Whether we can re-use the tok.
   */
  virtual void CreateDtypes(Expr op, std::string dtype_name, bool can_reuse) = 0;
};

class DtypeInfoInit : public DtypeInfoBaseVisitor {
 public:
  explicit DtypeInfoInit(support::Arena* arena) {
    arena_ = arena;
  }

  /*! \return The internal dtype map */
  std::unordered_map<const ExprNode*, std::vector<DtypeInfo*> > GetInitTokenMap(
      const Function& func) {
    this->Run(func);
    return std::move(dtype_map_);
  }

 protected:
  using DtypeInfoBaseVisitor::VisitExpr_;

  void CreateDtypes(Expr op, std::string dtype_name,bool can_reuse) final{
    auto expr_node = op.as<ExprNode>();
    ICHECK(!dtype_map_.count(expr_node));
    
    std::vector<DtypeInfo*> tokens;

    if (const auto* tuple_type = op->checked_type().as<TupleTypeNode>()) {
      for (Type t : tuple_type->fields) {
        const auto* ttype = t.as<TensorTypeNode>();
        ICHECK(ttype);
        DtypeInfo* token = arena_->make<DtypeInfo>();
        //if (dtype_name == "func" || dtype_name == "dla" ){
        if (dtype_name == "dla" ){
          token->ref_dtype = 5;
        }
        else{
          int dy_type = 3;
          GetOriginType(ttype,dy_type);
          token->ref_dtype = dy_type;
        }
        token->dtype_name = GetRefName(dtype_name);
        //if (token->ref_dtype==0){
        //  LOG(FATAL) << "not suport: " << AsText(op,false);
        //}
        tokens.push_back(token);
      }
    } else {
      const auto* ttype = op->checked_type().as<TensorTypeNode>();
      ICHECK(ttype);
      DtypeInfo* token = arena_->make<DtypeInfo>();
      //if (dtype_name == "func" || dtype_name == "dla" ){
      if (dtype_name == "dla" ){
        token->ref_dtype = 5;
      }
      else{
        int dy_type = 3;
        GetOriginType(ttype,dy_type);
        token->ref_dtype = dy_type;
      }
      token->dtype_name = GetRefName(dtype_name);
      //if (token->ref_dtype==0){
      //  LOG(FATAL) << "not suport: " << AsText(op,false);
      //}
      tokens.push_back(token);
    }
    dtype_map_[expr_node] = tokens;
    //LOG(INFO)<< "dtype_map_: " << dtype_map_.size();
  }

  void VisitExpr_(const CallNode* op) final {
    
    Expr expr = GetRef<Expr>(op);
    // for each input, visit argument token.
    for (Expr arg : op->args) {
      arg = GetNotQuantizedExpr(arg);
      VisitExpr(arg);
    }
    // deal with function
    if (c_name_ == "func_"){
      Function func_node;
      if(op->op.as<FunctionNode>()){
        func_node = GetRef<Function>(op->op.as<FunctionNode>());
      } 
      else {
        LOG(FATAL) << "TVM Dtype does not support calls to " << op->op->GetTypeKey();
      }
      std::string compiler = func_node->GetAttr<String>(attr::kCompiler).value();

      if (compiler == "riscv")  // riscv func
      {
        // create token for the call node.
        CreateDtypes(expr, "riscv", true);
        for (auto param : func_node->params){
          Expr param_expr = GetRef<Expr>(param.operator->());   
          CreateDtypes(param_expr, "riscv", true); 
          //LOG(INFO)<< "param_expr: "<< AsText(param_expr,false);
        }
        c_name_ = "riscv_";
        VisitExpr(func_node->body);  //go into riscv func, visit riscv op
        c_name_ = "func_";
      }
      else if(compiler.substr(0, 3) == "dla"){
        // create token for the call node.
        CreateDtypes(expr, "dla", true);
      }
      else {
        LOG(FATAL) << "TVM Dtype does not support compiler name: " << compiler;
      }

    }
    else if (c_name_ == "riscv_"){
      if (const auto* op_node = op->op.as<OpNode>()) { 
        const auto op_name = GetRef<Op>(op_node)->name;
        // quantized op
        if (op_name == "relay.op.annotation.simulated_quantize" || op_name == "annotation.stop_fusion" || op_name == "annotation.cast_hint"){
          //VisitExpr(op->args[0]);
        }
        else{ // normal ops
          CreateDtypes(expr, "riscv", true);
        }
      }
      else{ 
        LOG(FATAL) << "TVM Dtype does not support calls to : " << op->op->GetTypeKey();
      }
    }
    else {
      LOG(FATAL) << "Dtype does not support c_name: " << c_name_;
    }
  }

 private:
  // allocator
  support::Arena* arena_;
};

class AidDtypeExpr : public DtypeInfoBaseVisitor {
 public:
  
  // Run true dtype for a function.
  Map<Expr, Array<IntegerArray> > GetAidDtype(const Function& func) {
    
    prototype_ = DtypeInfoInit(&arena_).GetInitTokenMap(func);
    //LOG(INFO) << "&&&&&&&&&&&&&&&&&&&&&&&&prototype_&&&&&&&&&&&&&&&&&&&&&&&&&&: " << prototype_.size();
    this->Run(func);

    Map<Expr, Array<IntegerArray>> dtypemap;
    for (const auto& kv : dtype_map_) {
      std::vector<Integer> ref_dtypes;
      std::vector<Integer> d_names;  // the max bytes for a storage id
      if (kv.second.size()==0){
        LOG(FATAL) << "not suport: " << AsText(GetRef<Expr>(kv.first),false);
      }      
      for (DtypeInfo* tok : kv.second) {
        //if (tok->ref_dtype==0){
        //  LOG(FATAL) << "not suport: " << AsText(GetRef<Expr>(kv.first),false);
        //}
        ref_dtypes.push_back(tok->ref_dtype);
        d_names.push_back(tok->dtype_name);
      }
      
      dtypemap.Set(GetRef<Expr>(kv.first), Array<IntegerArray>({ref_dtypes, d_names}));
    }
    //debug();
    //LOG(INFO) << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&dtype_map_-debug&&&&&&&&&&&&&&&&&&&&&&&&&&&: " << dtype_map_.size();
    return dtypemap;
  }

 protected:
  using DtypeInfoBaseVisitor::VisitExpr_;

  // override create token by getting token as prototype requirements.
  void CreateDtypes(Expr op, std::string dtype_name, bool can_reuse)  final{
    auto expr_node = op.as<ExprNode>();
    //ICHECK(!dtype_map_.count(expr_node))<< "error" << AsText(op, false);
    auto it = prototype_.find(expr_node);
    if (it == prototype_.end() && dtype_name == "host"){
      return;
    }
    ICHECK(it != prototype_.end());
    std::vector<DtypeInfo*> tokens;
    for (DtypeInfo* tok : it->second) {
      //if (tok->ref_dtype==0){
      //  LOG(FATAL) << "not suport: " << AsText(op,false);
      //}
      tokens.push_back(tok);
    }
    dtype_map_[expr_node] = tokens;
    //LOG(INFO)<< "dtype_map_: " << dtype_map_.size();
  }

  void CorrectDtypes(Expr op, std::string dtype_name, std::vector< int > dy_types) {
    auto expr_node = op.as<ExprNode>();
    ICHECK(dtype_map_.count(expr_node))<< "error" << AsText(op, false);
    auto it = prototype_.find(expr_node);
    ICHECK(it != prototype_.end());
    for (int i =0; i<dtype_map_[expr_node].size();i++) {    
      dtype_map_[expr_node][i]->dtype_name = GetRefName(dtype_name);
      dtype_map_[expr_node][i]->ref_dtype = dy_types[i];
      //if (dtype_map_[expr_node][i]->ref_dtype==0){
      //  LOG(FATAL) << "not suport: " << AsText(op,false);
      //}
    }
  }

  void CopyDtypes(Expr src, Expr dst ) {
    auto src_expr = src.as<ExprNode>();
    auto dst_expr = dst.as<ExprNode>();
    ICHECK(dtype_map_.count(src_expr));
    std::vector<DtypeInfo*> tokens;
    for (DtypeInfo* tok : dtype_map_[src_expr]) {
      //if (tok->ref_dtype==0){
      //  LOG(FATAL) << "not suport: " << AsText(dst,false);
      //}
      tokens.push_back(tok);
    }
    dtype_map_[dst_expr] = tokens;
  }
  
  // The call map
  void VisitExpr_(const CallNode* op) final {
    Expr expr = GetRef<Expr>(op);
    //expr = GetNotQuantizedExpr(expr);
    //auto op = expr.as<CallNode>();
    //if(!op){
    //  LOG(INFO) << "expr type: " << expr->GetTypeKey();
    //  LOG(FATAL) << "not suport: " << AsText(expr,false);
    //}
    // deal with the last function of dla function
    if (c_name_ == "last_func_"){
      //LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!go into last_func_!!!!!!!!!!!!!!!!!!!!!!!!";
      AdjustLastFunction(op);
      return;
    }

    // deal with the last normal riscv op of quantized op
    if (c_name_ == "last_riscv_"){
      //LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!go into last_riscv_!!!!!!!!!!!!!!!!!!!!!!!!";
      AdjustLastRiscvOp(op);
      return;
    }

    if (c_name_ == "func_" && dtype_map_.count(op)){
      return;
    }
    if (c_name_ == "riscv_" && dtype_map_.count(op)){
      return;
    }
    // for each input, visit argument token.
    for (Expr arg : op->args) {
      //arg = GetNotQuantizedExpr(arg);
      VisitExpr(arg);
    }
    // deal with function
    if (c_name_ == "func_"){
      Function func_node;
      if(op->op.as<FunctionNode>()){
        func_node = GetRef<Function>(op->op.as<FunctionNode>());
      } 
      else {
        LOG(FATAL) << "TVM Dtype does not support calls to " << op->op->GetTypeKey();
      }
      std::string compiler = func_node->GetAttr<String>(attr::kCompiler).value();

      if (compiler == "riscv")  // riscv func
      {
        // create token for the call node.
        CreateDtypes(expr, "riscv", true);
        c_name_ = "riscv_";
        Expr func_body_expr = GetNotQuantizedExpr(func_node->body);
        VisitExpr(func_body_expr);  //go into riscv func, visit riscv op
        c_name_ = "func_";
        // the dtype of riscv func 
        auto func_body_node = func_body_expr.as<ExprNode>();
        if (dtype_map_.count(func_body_node)){ //not quantized op
          //dtype_map_[op]=dtype_map_[func_body_node];
          CopyDtypes(func_body_expr,expr);
        }
        else { //quantized op
          std::vector <int> dy_types;
          for (int i =0; i<dtype_map_[op].size();i++) {
            dy_types.push_back(5);//int8
          }
          CorrectDtypes(expr, "dla", dy_types );//riscv
        }

        for (int t=0; t< func_node->params.size(); t++){
          auto arg_node = op->args[t].as<ExprNode>();
          auto param_expr = GetRef<Expr>(func_node->params[t].operator->());
          if (func_body_expr == param_expr){
            // this rsicv function only contains quantized op, its params should be in 8
            auto param_node = func_node->params[t].operator->();
            ICHECK(prototype_.count(param_node)) << AsText(param_expr,false);
            auto it = prototype_.find(op);
            std::vector <int> dy_types;
            //LOG(INFO)<<"it->second.size(): " << it->second.size();
            for (int i =0; i<it->second.size();i++) {
              dy_types.push_back(5);//int8
            }
            CreateDtypes(param_expr, "riscv", true);
            CorrectDtypes(param_expr,"dla", dy_types);//riscv
            //LOG(INFO)<< AsText(op->args[t],false);
            if (op->args[t]->IsInstance<VarNode>()){
              CopyDtypes(param_expr,op->args[t]); //Expr src, Expr dst
            }
            else{
              LOG(FATAL) << "op->args[t] is not VarNode: " << op->args[t]->GetTypeKey();
            }
            
          }
          else{
            // the dtype of riscv input op  (the dtype of riscv function's params) 
            if(dtype_map_.count(arg_node)){
              CopyDtypes(op->args[t],param_expr);
            }
            else{
              LOG(FATAL) << "op->args[t] is not in dtype_map_: " << AsText(op->args[t],false);
            }
          }
        }
      }
      else if(compiler.substr(0, 3) == "dla"){
        // create token for the call node.
        CreateDtypes(expr, "dla", true);
        c_name_ = "last_func_";
        for (auto call_arg : op->args) {
          VisitExpr(call_arg);
        }
        c_name_ = "func_";
      }
      else {
        LOG(FATAL) << "TVM Dtype does not support compiler name: " << compiler;
      }

    }
    else if (c_name_ == "riscv_"){
      if (const auto* op_node = op->op.as<OpNode>()) { 
        const auto op_name = GetRef<Op>(op_node)->name;
        // quantized op
        if (op_name == "relay.op.annotation.simulated_quantize" || op_name == "annotation.stop_fusion" || op_name == "annotation.cast_hint"){
          c_name_ = "last_riscv_";
          Expr riscv_expr = GetNotQuantizedExpr(op->args[0]);
          VisitExpr(riscv_expr);
          c_name_ = "riscv_";
        }
        else{ // normal ops
          CreateDtypes(expr, "riscv", true);
        }
      }
      else{ 
        LOG(FATAL) << "TVM Dtype does not support calls to : " << op->op->GetTypeKey();
      }
    }
    else {
      LOG(FATAL) << "Dtype does not support c_name: " << c_name_;
    }
  }
  
  void VisitExpr(const Expr& expr) {
    using TParent = ExprFunctor<void(const Expr&)>;
    TParent::VisitExpr(expr);
  }

  void AdjustLastRiscvOp(const CallNode* op, std::string dtype_name = "riscv"){
    Expr expr = GetRef<Expr>(op);
    Expr op_expr = GetNotQuantizedExpr(expr);
    if (op_expr->IsInstance<CallNode>()){
      std::vector <int> dy_types;
      auto op_node = op_expr.as<ExprNode>();
      for (int i =0; i<dtype_map_[op_node].size();i++) {
        dy_types.push_back(5);//int8
      }
      CorrectDtypes(op_expr, dtype_name, dy_types );//riscv
    }
    else{
      LOG(FATAL) << "Dtype does not support op_expr to " << op_expr->GetTypeKey();
    }
  }

  void AdjustLastFunction(const CallNode* op, std::string dtype_name = "dla"){  
    Expr expr = GetRef<Expr>(op);
    std::vector <int> dy_types;
    for (int i =0; i<dtype_map_[op].size();i++) {
      dy_types.push_back(5);//int8
    }
    CorrectDtypes(expr, dtype_name, dy_types );//dla

    if(expr->IsInstance<CallNode>()){
      Function func_node;
      if(op->op.as<FunctionNode>()){
        func_node = GetRef<Function>(op->op.as<FunctionNode>());
      } 
      else {
        LOG(FATAL) << "function Dtype does not support node to " << op->op->GetTypeKey();
      }

      if (const auto* call_node = func_node->body.as<CallNode>()){
        Expr op_expr = GetNotQuantizedExpr(func_node->body);
        auto op_node = op_expr.as<ExprNode>();
        //dtype_map_[op_node]=dtype_map_[op]; 
        CopyDtypes(expr,op_expr);
      }
      else if(const auto* tuple_node = func_node->body.as<TupleNode>()){
        Expr op_expr = GetNotQuantizedExpr(func_node->body);
        auto op_node = op_expr.as<ExprNode>();
        //dtype_map_[op_node]=dtype_map_[op];
        CopyDtypes(expr,op_expr);
        //ICHECK(tuple_node->fields.size()== dtype_map_[op_node].size());
        for (auto field : tuple_node->fields){
          Expr expr_branch = GetNotQuantizedExpr(field);
          auto op_branch = expr_branch.as<ExprNode>();
          std::vector <int> tmp_types;
          for (int s = 0; s< dtype_map_[op_branch].size(); s++){
            tmp_types.push_back(5); //int8
          }
          CorrectDtypes(expr_branch, dtype_name, tmp_types );//dla
          CopyDtypes(expr_branch,field);//dla
        }
      }
      else{
        LOG(INFO) << AsText(func_node,false);
        LOG(FATAL) << "riscv op Dtype does not support calls to " << op->op->GetTypeKey();
      }
    }
    else{
      LOG(INFO) << AsText(expr,false);
      LOG(FATAL) << "riscv op Dtype does not support calls to " << op->op->GetTypeKey();
    }

  }

  void debug(){
    LOG(INFO) << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&dtype_map_--new--debug&&&&&&&&&&&&&&&&&&&&&&&&&&&: " << dtype_map_.size();
    for (auto kv: dtype_map_){
      LOG(INFO) << " !!!!!!!!!!!!!!!!!!!!!!!!dtype_map_!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
      for (auto token: kv.second){
        LOG(INFO) <<"dtype: " << token->ref_dtype << " name: " << token->dtype_name << " type: " << kv.first->GetTypeKey();
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
        if (call->IsInstance<VarNode>()){
          LOG(INFO) << AsText(call,false);
        }
      }
      
      LOG(INFO) << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&dtype_map_---new---debug&&&&&&&&&&&&&&&&&&&&&&&&&&&";
    }
  }

  private:
  
  // allocator
  support::Arena arena_;
  /*! \brief internal prototype dtype map */
  std::unordered_map<const ExprNode*, std::vector<DtypeInfo*> > prototype_;

  // IntegerArray : 1st dtype_bytes ; 2ed name (1 for riscv ; 2 for dla; 3 for host) 
  //Map <Expr, IntegerArray> aid_dtype_; 
  Expr aid_expr_;
};

}  // namespace aipu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_GRAPH_AID_DTYPE_H_
