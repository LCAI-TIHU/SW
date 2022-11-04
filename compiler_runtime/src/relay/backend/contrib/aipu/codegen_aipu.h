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
 * \file src/relay/backend/contrib/aipu/codegen_ethosn.h
 * \brief The Relay -> Ethos-N command stream compiler.
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_AIPU_CODEGEN_AIPU_H_
#define TVM_RELAY_BACKEND_CONTRIB_AIPU_CODEGEN_AIPU_H_

#include <dmlc/memory_io.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h> // shenfw add
#include <tvm/relay/dataflow_matcher.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../../runtime/contrib/aipu/aipu_runtime.h"
#include "../codegen_c/codegen_c.h"

// NVDLA headers
#if 1
// #include "../../../../target/source/codegen_source_base.h"
//#define NVDLA_UTILS_ERROR_TAG "DLA"
#include "nvdla_os_inf.h"
#endif

namespace tvm {
namespace relay {
namespace contrib {
namespace aipu {

/*! \brief Store the result of a build. */
  class PatterTableResultNode : public Object {
  public:
    Array<String> pattern_names;
    Array<DFPattern> patterns;
    Array<String> checkfuncname;

    void VisitAttrs(tvm::AttrVisitor* v) {
      v->Visit("pattern_names", &pattern_names);
      v->Visit("patterns", &patterns);
      v->Visit("checkfuncname", &checkfuncname);
    }

    static constexpr const char* _type_key = "relay.ext.aipu.PatterTableResult";
    TVM_DECLARE_FINAL_OBJECT_INFO(PatterTableResultNode, Object);
  };

  // /*!
  //  * \brief Managed reference to BuildResultNode.
  //  * \sa BuildResultNode
  //  */
  class PatterTableResult : public ObjectRef {
  public:

    // PatterTableResult(Array<String> pattern_names, Array<DFPattern> patterns, Array<String> checkfuncname);
    PatterTableResult(Array<String> pattern_names,
                      Array<DFPattern> patterns, Array<String> checkfuncname)
    {
      auto node = make_object<PatterTableResultNode>();
      node->pattern_names = std::move(pattern_names);
      node->patterns = std::move(patterns);
      node->checkfuncname = std::move(checkfuncname);
      data_ = std::move(node);
    }

    TVM_DEFINE_OBJECT_REF_METHODS(PatterTableResult, ObjectRef, PatterTableResultNode);
  };

  TVM_REGISTER_NODE_TYPE(PatterTableResultNode);

/*class AIPUModuleNode : public runtime::ModuleNode {
 public:
  explicit AIPUModuleNode(uint8_t *buf, uint64_t size, std::string riscv_source) :
    buf_(buf), size_(size), riscv_source_(riscv_source) {};
  // destructor
  ~AIPUModuleNode() {
    LOG(INFO)<<"free 0";
    free(buf_);
  }

  const char* type_key() const final { return "aipu"; }

  // 没有添加此函数则编译会报错
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final
  {
    return PackedFunc();
  }

  // 输出loadable文件
  void SaveToFile(const std::string& file_name, const std::string& format) final {
    if (format != "nvdla")
      LOG(FATAL) << "doesn't support " << format << std::endl;
    NvDlaFileHandle file = 0;
    NvDlaError e = NvDlaSuccess;
    std::string fileName = file_name + ".nvdla";
    e = NvDlaFopen(fileName.c_str(), NVDLA_OPEN_WRITE, &file);
    if (e != NvDlaSuccess)
      LOG(FATAL) << "failed to open file" << std::endl;
    e = NvDlaFwrite(file, buf_, size_);
    if (e != NvDlaSuccess)
      LOG(FATAL) << "failed to write file" << std::endl;
    NvDlaFclose(file);
  }

  runtime::Module AipuRuntimeCreate(const std::string& sym_json, const tvm::runtime::Module& m, const std::vector<Device>& devs); // shenfw add
  runtime::Module AipuSetInput(TVMArrayHandle arr); // shenfw add
  runtime::Module AipuRun(); // shenfw add

  std::string GetSource(){
    return riscv_source_;
  }

  uint8_t* GetBuff() {
    return buf_;
  }

  uint64_t GetBuffSize() {
    return size_;
  }

  void SetBuffSize(uint64_t size) {
    size_ = size;
  }

  void SetBuff(uint8_t * buff) {
    buf_ = buff;
  }
  
  void SetRiscVSource(std::string riscv_source) {
    riscv_source_ = riscv_source;
  }

 private:
  // NVDLA flatbuffer
  uint8_t *buf_;
  // buff_size
  uint64_t size_;
  // riscv source code
  std::string riscv_source_;
};*/

}  // namespace aipu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_ETHOSN_CODEGEN_ETHOSN_H_
