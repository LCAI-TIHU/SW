/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*
 * Inspur.
 * This is a new or modified file.
 */

#ifndef NVDLA_PRIV_RELAY_PARSER_H
#define NVDLA_PRIV_RELAY_PARSER_H

#include <iostream>
#include <istream>
#include <vector>
#include <map>
#include <memory>
#include <string>

#include "priv/Type.h"
#include "priv/Network.h"
#include "priv/Layer.h"
#include <tvm/relay/function.h>

namespace tvm {
namespace relay {
namespace contrib {
namespace aipu{

class BlobNameToTensor {
 public:
  virtual void add(const std::string& name, nvdla::ITensor* tensor);
  virtual nvdla::ITensor* find(const char* name) const;
  virtual nvdla::ITensor*& operator[](const std::string& name);
  virtual void setTensorNames();
  virtual ~BlobNameToTensor();
 private:
  std::map<std::string, nvdla::ITensor*> mMap;
};

class RelayParser {
 public:
  RelayParser() :
    mDimsCallback(NULL),
    mBlobNameToTensor(NULL)
    { }

  void parse(const tvm::relay::Function& func, nvdla::INetwork* network);
  virtual int identifyOutputs(nvdla::INetwork * network);
  virtual ~RelayParser();
 private:
  nvdla::INetwork::OutputDimensionsFormula* mDimsCallback;
  BlobNameToTensor* mBlobNameToTensor;
};

}
}
}
}

#endif // NVDLA_PRIV_RELAY_PARSER_H
