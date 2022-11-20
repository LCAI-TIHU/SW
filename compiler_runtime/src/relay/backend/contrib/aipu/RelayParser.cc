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

#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include <unistd.h>

#include "RelayParseCore.h"
#include "ErrorMacros.h"
#include "priv/Check.h"
// #include "Check.h"
#include "RelayParser.h"

#include "half.h"
typedef half_float::half float16;

namespace tvm {
namespace relay {
namespace contrib {
namespace aipu {
/*!
 * \brief Extract shape from expr to vector<int64_t>
 *
 * \param shape
 * \return std::vector<int64_t>
 */
std::vector<int64_t> _ShapeToIntVector(tvm::Array<IndexExpr> shape) {
  std::vector<int64_t> ret;
  for (IndexExpr dim : shape) {
    const int64_t* pval = tir::as_const_int(dim);
    ret.push_back(*pval);
  }
  return ret;
}

void BlobNameToTensor::add(const std::string& name, nvdla::ITensor* tensor) {
  std::cout<<"BlobNameToTensor::add "<<name.c_str();
  mMap[name] = tensor;
}

nvdla::ITensor* BlobNameToTensor::find(const char* name) const {
  std::map<std::string, nvdla::ITensor*>::const_iterator p = mMap.find(name);
  if (p == mMap.end()) {
    return 0;
  }
  return p->second;
}

nvdla::ITensor*& BlobNameToTensor::operator[](const std::string& name) {
  return mMap[name];
}

void BlobNameToTensor::setTensorNames() {
  std::map<std::string, nvdla::ITensor*>::iterator p;
  std::cout<<"mBlobNameToTensor size:"<<mMap.size()<<std::endl;
  for ( p = mMap.begin(); p != mMap.end(); p++) {
    p->second->setName(p->first.c_str());
  }
}

BlobNameToTensor::~BlobNameToTensor() { }

// TODO. Deal with this function
struct RelayParserPoolingDimsCallback : public nvdla::INetwork::OutputDimensionsFormula {
  // FB pooling parameters
  // Use floor((height + 2 * padding - kernel) / stride) + 1
  // instead of ceil((height + 2 * padding - kernel) / stride) + 1
  std::set<std::string> mHasTorchPooling;

  // TODO: mostly duplicated with code in engine
  virtual nvdla::Dims2 compute(nvdla::Dims2 input, nvdla::Dims2 kernel, nvdla::Dims2 stride,
      nvdla::Dims2 tlPadding, nvdla::Dims2 brPadding, const char* layerName) const /* override */
  {
    // check for overflow before we delve into any further computations here...
    assert( input.h + tlPadding.h + brPadding.h >= kernel.h );
    assert( input.w + tlPadding.w + brPadding.w >= kernel.w );
    int pooledH, pooledW;
    if (mHasTorchPooling.find(std::string(layerName)) != mHasTorchPooling.end()
        || kernel.h == kernel.w ==1)
    {
      pooledH = static_cast<int>
        (std::floor(static_cast<float>(input.h + tlPadding.h + brPadding.h - kernel.h) / stride.h)) + 1;
      pooledW = static_cast<int>
        (std::floor(static_cast<float>(input.w + tlPadding.w + brPadding.w - kernel.w) / stride.w)) + 1;

    } else
    {
      pooledH = static_cast<int>
        (std::ceil(static_cast<float>(input.h + tlPadding.h + brPadding.h - kernel.h) / stride.h)) + 1;
      pooledW = static_cast<int>
        (std::ceil(static_cast<float>(input.w + tlPadding.w + brPadding.w - kernel.w) / stride.w)) + 1;
    }

    if (tlPadding.h || tlPadding.w)
    {
      // DS: caffe comment for this (which doesn't work if padding is very large) is:
      // "If we have padding, ensure that the last pooling starts strictly inside the image (instead of at the padding); otherwise clip the last."
      if ((pooledH - 1) * stride.h >= input.h + tlPadding.h)
        --pooledH;
      if ((pooledW - 1) * stride.w >= input.w + tlPadding.w)
        --pooledW;

      assert((pooledH - 1) * stride.h < input.h + tlPadding.h);
      assert((pooledW - 1) * stride.w < input.w + tlPadding.w);
    }
    //int pooledH, pooledW;
    //pooledH = 0;
    //pooledW = 0;
    //LOG(INFO)<<"virtual nvdla::Dims2 compute one: "<<pooledH<<", "<<pooledW<<std::endl;
    return nvdla::Dims2(pooledH, pooledW);
  }

  nvdla::Dims2 compute(nvdla::Dims2 /*input*/, nvdla::Dims2 /*kernel*/, nvdla::Dims2 /*stride*/,
      nvdla::Dims2 /*tlPadding*/, nvdla::Dims2 /*brPadding*/, nvdla::Dims2 /*dilation*/, const char*) const
  {
    gLogInfo<<"virtual nvdla::Dims2 compute two -1 -1 :";;
    return nvdla::Dims2(-1, -1);
  }

};

RelayParser::~RelayParser() {
  delete mBlobNameToTensor;
}

void RelayParser::parse(const tvm::relay::Function& func, nvdla::INetwork* network) {
  ICHECK(func.defined()) << "Input error: expect a Relay function.";
  mDimsCallback = new RelayParserPoolingDimsCallback;
  network->setPoolingOutputDimensionsFormula(mDimsCallback);
  mBlobNameToTensor = new BlobNameToTensor();
  // parse inputs to Tensors and set them in mBlobNameToTensor. Network
  for (auto arg : func->params) {
    if (const auto* tensor_type = arg->checked_type().as<TensorTypeNode>()) {
      std::vector<int64_t> shape = _ShapeToIntVector(tensor_type->shape);
      // the input shape must be NHWC but how to know input shape needs advise
      nvdla::Dims4 dims;
      if (shape.size() == 2) {
        dims.n = 1;
        dims.c = shape[1];
        dims.h = shape[0];
        dims.w = 1;
      } else if (shape.size() == 3) {
        dims.n = 1;
        dims.c = shape[2];
        dims.h = shape[0];
        dims.w = shape[1];
      } else if (shape.size() == 4){
        dims.n = shape[0];
        dims.c = shape[3];
        dims.h = shape[1];
        dims.w = shape[2];
      } else if (shape.size() == 1){
        dims.n = 1;
        dims.c = shape[0];
        dims.h = 1;
        dims.w = 1;
      } else if (!shape.size()){
        dims.n = 1;
        dims.c = 1;
        dims.h = 1;
        dims.w = 1;
      } else {
        LOG(FATAL) << "DLA does not accept shape length: " << shape.size();
      }
      const auto& name = arg->name_hint();
      LOG(INFO) << "func param name: " << name;
      nvdla::ITensor* tensor = network->addInput(name.c_str(), dims);
      mBlobNameToTensor->add((std::string)name.c_str(), tensor);
    } else {
      LOG(FATAL) << "Failed to obtain tensor type of an input of a DLA func";
    }
  }
  auto sid = GetExtSymbol(func);
  LOG(INFO) << "sid: " << sid;
  // parse func->body to Network, BlobNameToTensor.
  AIPURelay2NetworkCore builder(sid, network, mBlobNameToTensor);
  builder.VisitExpr(func->body);
  mBlobNameToTensor->setTensorNames();
  if (network->getNumOutputs() <= 0) {
    int outs = identifyOutputs(network);
    LOG(INFO) << "Marking total " << outs << " outputs";
  }
  return;
}

int RelayParser::identifyOutputs(nvdla::INetwork* network) {
  std::set< nvdla::ITensor* > outputTensors;
  std::set< nvdla::ITensor* > inputTensors;
  for (int l = 0; l < network->getNumLayers(); ++l) {
    nvdla::ILayer* layer = network->getLayer(l);
    if (!layer)
      return -1;
    for (int ii = 0; ii < layer->getNumInputs(); ++ii) {
      inputTensors.insert(layer->getInput(ii));
    }
    for (int oo = 0; oo < layer->getNumOutputs(); ++oo) {
      outputTensors.insert(layer->getOutput(oo));
    }
  }
  for (std::set<nvdla::ITensor*>::iterator oi = outputTensors.begin(); oi != outputTensors.end(); ++oi) {
    // an output tensor which is not an input to any other layers is a network output tensor
    if (inputTensors.find(*oi) == inputTensors.end()) {
      network->markOutput(*oi);
      LOG(INFO) << "Mark output: " << (*oi)->getName();
    }
  }
  return network->getNumOutputs();
}

}//tvm::relay::contrib::aipu
} //tvm::relay::contrib
} // tvm::relay
} // tvm::

