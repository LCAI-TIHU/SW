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

#include "RelayParseCore.h"
#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include "ErrorMacros.h"
#include "priv/Check.h"
// #include "Check.h"
//#include "priv/caffe/CaffeParser.h"
#include "RelayParser.h"

#include "ditcaffe/protobuf-2.6.1/ditcaffe.pb.h"

#include "half.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <unistd.h>
//using namespace nvdla;
namespace dc = ditcaffe;
typedef half_float::half float16;
using nvdla::BiasMode;
using nvdla::ILayer;
//using nvdla::DataType;
using nvdla::PoolingType;
using nvdla::kRELU;
using nvdla::sUNIFORM;
using nvdla::kSUM;
using nvdla::kPROD;
using nvdla::ew_kMAX;
using nvdla::kSIGMOID;
using nvdla::kTANH;
using nvdla::BatchNormMode;
using nvdla::ScaleMode;
using nvdla::ElementWiseOperation;

namespace tvm {
namespace relay {
namespace contrib {
namespace aipu{
/*!
   * \brief Extract shape from expr to vector<int64_t>
   *
   * \param shape
   * \return std::vector<int64_t>
   */
  std::vector<int64_t> _ShapeToJSON(tvm::Array<IndexExpr> shape) {
    std::vector<int64_t> ret;
    for (IndexExpr dim : shape) {
      const int64_t* pval = tir::as_const_int(dim);
      // LOG(INFO)<<"dim:"<<*pval;
      ret.push_back(*pval);
    }
    return ret;
  }
IBlobNameToTensor::~IBlobNameToTensor() { }
IBinaryProtoBlob::~IBinaryProtoBlob() { }
IRelayParser::~IRelayParser() { }
IRelayParser *createRelayParser()
{
    priv::RelayParserFactory::RelayParserPrivPair ppair;
    ppair = priv::RelayParserFactory::newRelayParser();
    return ppair.i();
}
NvDlaError destroyRelayParser(IRelayParser *parser)
{
    NvDlaError e = NvDlaSuccess;
    PROPAGATE_ERROR_FAIL(priv::RelayParserFactory::deleteRelayParser(parser));
fail:
    return e;
}

namespace priv
{

RelayParserFactory::RelayParserPrivPair RelayParserFactory::newRelayParser()
{
    IRelayParser *parser;
    RelayParser *parser_priv;
    parser = parser_priv = new priv::RelayParser();
    if (parser) {
        s_priv.insert(parser,parser_priv);
        s_self.insert(parser, parser);
    }
    return RelayParserPrivPair(parser, parser_priv);
}

NvDlaError RelayParserFactory::deleteRelayParser(IRelayParser *parser)
{
    if (parser != NULL) {
        RelayParser *parser_priv = priv(parser);
        if (parser_priv != NULL) {
            delete(parser_priv);
        }

        s_priv.remove(parser);
        s_self.remove(parser);
    }

    return NvDlaSuccess;
}

RelayParser *RelayParserFactory::priv(IRelayParser *parser)
{
    // gLogError << __func__ << " looking up priv for base_i=" << parser << endl;
    nvdla::compiler::priv::BiMap<IRelayParser *, RelayParser *>::left_iterator f = s_priv.find_left(parser);
    if ( f == s_priv.end_left() ) {
        return NULL;
    }
    return f->second;
}

IRelayParser *RelayParserFactory::i(RelayParser *parser)
{
    nvdla::compiler::priv::BiMap<IRelayParser *, RelayParser *>::right_iterator f = s_priv.find_right(parser);
    if ( f == s_priv.end_right() ) {
        return NULL;
    }
    return f->second;
}


IRelayParser *RelayParserFactory::self(void *s)
{
    nvdla::compiler::priv::BiMap<void *, IRelayParser *>::left_iterator f = s_self.find_left(s);
    if ( f == s_self.end_left() ) {
        return NULL;
    }
    return f->second;
}

nvdla::compiler::priv::BiMap<IRelayParser *, RelayParser*> RelayParserFactory::s_priv;
nvdla::compiler::priv::BiMap<void *, IRelayParser*> RelayParserFactory::s_self;


void BlobNameToTensor::add(const std::string& name, nvdla::ITensor* tensor)
{
    std::cout<<"BlobNameToTensor::add "<<name.c_str();
    mMap[name] = tensor;
}

nvdla::ITensor* BlobNameToTensor::find(const char* name) const
{
    std::map<std::string, nvdla::ITensor*>::const_iterator p = mMap.find(name);
    if (p == mMap.end()) {
        return 0;
    }
    return p->second;
}

nvdla::ITensor*& BlobNameToTensor::operator[](const std::string& name)
{
  //LOG(INFO) <<"BlobNameToTensor::operator[] "<<name.c_str()
  //          <<" mMap[name] size: "<<mMap.size();
    return mMap[name];
}

void BlobNameToTensor::setTensorNames()
{
    std::map<std::string, nvdla::ITensor*>::iterator p;
    std::cout<<"mBlobNameToTensor size:"<<mMap.size()<<std::endl;
    for ( p = mMap.begin(); p != mMap.end(); p++) {
        p->second->setName(p->first.c_str());
    }
}


BlobNameToTensor::~BlobNameToTensor() { }

struct RelayParserPoolingDimsCallback : public nvdla::INetwork::OutputDimensionsFormula
{
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

void RelayParser::shutdownProtobufLibrary()
{
    google::protobuf::ShutdownProtobufLibrary();
}

// There are some challenges associated with importing caffe models. One is that
// a .caffemodel file just consists of layers and doesn't have the specs for its
// input and output blobs.
//
// So we need to read the deploy file to get the input

static bool readBinaryProto(dc::NetParameter* net, const char* file, size_t bufSize)
{
    CHECK_NULL_RET_VAL(net, false);
    CHECK_NULL_RET_VAL(file, false);
    using namespace google::protobuf::io;

    std::ifstream stream(file, std::ios::in | std::ios::binary);
    if (!stream)
    {
        std::cout << "could not open file " << file << std::endl;
        return false;
    }

    IstreamInputStream rawInput(&stream);
    CodedInputStream codedInput(&rawInput);
    codedInput.SetTotalBytesLimit(int(bufSize), -1);

    bool ok = net->ParseFromCodedStream(&codedInput);
    stream.close();

    if (!ok)
    {
        std::cout << "Relay Parser: could not parse binary model file" << std::endl;
        return false;
    }

    return ok;
}

static bool readTextProto(dc::NetParameter* net, const char* file)
{
    CHECK_NULL_RET_VAL(net, false);
    CHECK_NULL_RET_VAL(file, false);
    using namespace google::protobuf::io;

    std::ifstream stream(file, std::ios::in );
    if (!stream)
    {
        std::cout << "could not open file " << file;
        return false;
    }

    IstreamInputStream input(&stream);
    bool ok = google::protobuf::TextFormat::Parse(&input, net);
    stream.close();

    if (!ok)
    {
        std::cout << "Relay Parser: could not parse text file" << std::endl;
        return false;
    }

    return ok;
}

enum /*class*/ WeightType
{
    // types for convolution, deconv, fully connected
    kGENERIC = 0,	// typical weights for the layer: e.g. filter (for conv) or matrix weights (for innerproduct)
    kBIAS = 1,		// bias weights

    kMEAN = 0,
    kVARIANCE = 1,
    kMOVING_AVERAGE = 2
};


class RelayWeightFactory
{
public:
    RelayWeightFactory(const dc::NetParameter& msg, bool convertTo16bit, std::vector<void*>& tmpAllocs)
        : mMsg(msg), mTmpAllocs(tmpAllocs), m16bit(convertTo16bit), mOK(true)
    {
        mRef = new dc::NetParameter;
    }
    virtual ~RelayWeightFactory() { }

    bool is16bit() const
    {
        return m16bit;
    }

    std::vector<void*>& getTmpAllocs()
    {
        return mTmpAllocs;
    }

    virtual nvdla::Weights operator()(const std::string& layerName, WeightType weightType)
    {
        int numLayers = mMsg.layer_size();

        const dc::BlobProto* blobMsg;

        if (numLayers > 0)
        {
            int i = 0;
            for (; i < mMsg.layer_size(); i++)
            {
                std::string n = mMsg.layer(i).name();
                if (mMsg.layer(i).name() == layerName) {
                    break;
                }
            }

            int index = static_cast<int>(weightType);
            blobMsg = &mMsg.layer(i).blobs(index);
        }
        else
        {
            int i = 0;
            for (; i < mMsg.layers_size(); i++)
            {
                std::string n = mMsg.layers(i).name();
                if (mMsg.layers(i).name() == layerName) {
                    break;
                }
            }

            int index = static_cast<int>(weightType);
            blobMsg = &mMsg.layers(i).blobs(index);
        }


        if (!m16bit)
        {
            if (blobMsg->data_size() >0)
            {
                mOK &= checkForNans<float>(blobMsg->data().data(), int(blobMsg->data_size()), layerName);
                std::cout<<"layer name:"<<layerName<<",blob data size:"<<blobMsg->data_size();
                return nvdla::Weights(nvdla::DataType::FLOAT, blobMsg->data().data(), NvS64(blobMsg->data_size()));
            }
            std::cerr << layerName << ": ERROR - 32-bit weights not found for 32-bit model" << std::endl;
            mOK = false;
            return nvdla::Weights(nvdla::DataType::FLOAT, NULL, 0);
        }


        size_t count;
        float16* data;
        if (blobMsg->half_data_size() > 0)
        {
            count = blobMsg->half_data().size();
            data = (float16*)blobMsg->half_data().data();
            for (int i = 0; i < blobMsg->half_data().size(); i++) {
                // 'cos the fp16 data is stored in uint32, luvverly.
                data[i] = data[i * 2];
            }
        }
        else
        {
            count = blobMsg->data().size();
            data = reinterpret_cast<float16*>(malloc(count*sizeof(float16)));
            mTmpAllocs.push_back(data);
            float* data32 = (float*)blobMsg->data().data();
            for (size_t i = 0; i < count; i++)
            {
                if (data32[i]>std::numeric_limits<float16>::max() ||
                    data32[i] < -std::numeric_limits<float16>::max())
                {
                    std::cerr << "error:" << layerName << ": - weights are out"
                        " of range for 16-bit conversion" << std::endl;
                    mOK = false;
                }
                data[i] = data32[i];

            }
        }


        mOK &= checkForNans<float16>(data, count, layerName);
        return nvdla::Weights(nvdla::DataType::HALF, data, NvS64(count));
    }

    bool isOK()
    {
        return mOK;
    }

private:
    template<typename T> bool checkForNans(const void* values, int count, const std::string& layerName)
    {
        const T* v = reinterpret_cast<const T*>(values);
        for (int i = 0; i < count; i++)
        {
            if (std::isnan(float(v[i])))
            {
                NVDLA_UNUSED(layerName);
                // std::cout << layerName << ": Nan detected in weights" << std::endl;
                return false;
            }
        }
        return true;
    }

    const dc::NetParameter& mMsg;
    dc::NetParameter * mRef;
    std::vector<void*>& mTmpAllocs;
    bool m16bit;

    bool mOK;
};

static ILayer* parseConvolution(nvdla::INetwork *network, const dc::LayerParameter& msg,
                                       RelayWeightFactory& weightFactory,
                                       IBlobNameToTensor* tensors)
{
    const dc::ConvolutionParameter& p = msg.convolution_param();
    int numOutputs = p.num_output();
    int numGroups  = p.has_group()? p.group() : 1;
    ILayer* layer = NULL;

    int kernelW = p.has_kernel_w() ? p.kernel_w() : p.kernel_size(0);
    int kernelH = p.has_kernel_h() ? p.kernel_h() : p.kernel_size_size() > 1 ? p.kernel_size(1) : p.kernel_size(0);

    int strideW = p.has_stride_w() ? p.stride_w() : p.stride_size() > 0 ? p.stride(0) : 1;
    int strideH = p.has_stride_h() ? p.stride_h() : p.stride_size() > 1 ? p.stride(1) : p.stride_size() > 0 ? p.stride(0) : 1;

    int padW = p.has_pad_w() ? p.pad_w() : p.pad_size() > 0 ? p.pad(0) : 0;
    int padH = p.has_pad_h() ? p.pad_h() : p.pad_size() > 1 ? p.pad(1) : p.pad_size() > 0 ? p.pad(0) : 0;

    int dilationW = p.dilation_size() > 0 ? p.dilation(0) : 1;
    int dilationH = p.dilation_size() > 1 ? p.dilation(1) : p.dilation_size() > 0 ? p.dilation(0) : 1;

    nvdla::BiasMode biasMode = nvdla::BiasMode::bNONE;

    // TODO: cross-correlation vs convolution
    nvdla::Weights kernelWeights = weightFactory(msg.name(), /*WeightType::*/kGENERIC);
    nvdla::Weights biasWeights =
        (!p.has_bias_term() || p.bias_term()) ?
        weightFactory(msg.name(), /*WeightType::*/kBIAS) :
        nvdla::Weights(nvdla::DataType::FLOAT, NULL, 0);

    if ( biasWeights.count == 0 )
    {
        biasMode = BiasMode::bNONE;
    }
    else if ( biasWeights.count == 1 )
    {
        biasMode = BiasMode::bUNIFORM;
    }
    else if ( biasWeights.count == numOutputs )
    {
        biasMode = BiasMode::bCHANNEL;
    }
    else
    {
        biasMode = BiasMode::bm_ELEMENTWISE;
    }

    nvdla::Dims2 tlPadding = nvdla::Dims2(padH, padW);
    nvdla::Dims2 brPadding = nvdla::Dims2(padH, padW);
    nvdla::Dims2 stride    = nvdla::Dims2(strideH, strideW);
    nvdla::Dims2 dilation  = nvdla::Dims2(dilationH, dilationW);
    nvdla::Dims2 kernelSize= nvdla::Dims2(kernelH, kernelW);

    // TODO: cross-correlation vs convolution
    layer = network->addConvolution((*tensors)[msg.bottom(0)], numOutputs, 0,
                                    kernelSize, tlPadding, brPadding, stride, dilation,
                                    kernelWeights, biasWeights, biasMode, numGroups);
    gLogInfo<<"msg.name():"<<msg.name()<<"msg.bottom(0):"<<msg.bottom(0)<<std::endl;
    return layer;
}

static ILayer* parsePooling(nvdla::INetwork* network, const dc::LayerParameter&msg,
                                   RelayWeightFactory& /*weightFactory*/,
                                   IBlobNameToTensor * tensors)
{
    const dc::PoolingParameter& p = msg.pooling_param();
    if (p.pool() != dc::PoolingParameter::MAX && p.pool() != dc::PoolingParameter::AVE)
    {
        gLogError << "only AVE and MAX pool operations are supported" << std::endl;
        return 0;
    }


    // mandatory
    int kernelH, kernelW;
    if (p.has_global_pooling() && p.global_pooling())
    {
        nvdla::Dims4 dims = (*tensors)[msg.bottom(0)]->getDimensions();
        kernelH = dims.h;
        kernelW = dims.w;
    }
    else
    {
        // mandatory
        kernelH = p.has_kernel_h() ? p.kernel_h() : p.kernel_size();
        kernelW = p.has_kernel_w() ? p.kernel_w() : p.kernel_size();
    }

    int strideH = p.has_stride_h() ? p.stride_h() : p.has_stride() ? p.stride() : 1;
    int strideW = p.has_stride_w() ? p.stride_w() : p.has_stride() ? p.stride() : 1;

    int padH = p.has_pad_h() ? p.pad_h() : p.has_pad() ? p.pad() : 0;
    int padW = p.has_pad_w() ? p.pad_w() : p.has_pad() ? p.pad() : 0;

    nvdla::Dims2 windowSize = nvdla::Dims2(kernelH, kernelW);
    nvdla::Dims2 stride     = nvdla::Dims2(strideH, strideW);
    nvdla::Dims2 tlPadding  = nvdla::Dims2(padH, padW);
    nvdla::Dims2 brPadding  = nvdla::Dims2(padH, padW);

    PoolingType type = p.has_pool() && p.pool() ==
        dc::PoolingParameter::AVE ? PoolingType::kAVERAGE : PoolingType::kMAX;
    gLogInfo<<"pool compute"<<std::endl;
    ILayer *layer = network->addPooling((*tensors)[msg.bottom(0)], type,
                                        windowSize, stride, tlPadding, brPadding);

    if (layer)
    {
        layer->setName(msg.name().c_str());
        if (p.has_torch_pooling() ? p.torch_pooling() : false) {
            gLogInfo<<"nvdla network pool"<<std::endl;
            static_cast<RelayParserPoolingDimsCallback &>
                (network->getPoolingOutputDimensionsFormula()).mHasTorchPooling.insert(msg.name());
        }

        (*tensors)[msg.top(0)] = layer->getOutput(0);
    }
    return layer;
}

static ILayer* parseInnerProduct(nvdla::INetwork* network, const dc::LayerParameter&msg,
                                        RelayWeightFactory& weightFactory,
                                        IBlobNameToTensor * tensors)
{
    const dc::InnerProductParameter& p = msg.inner_product_param();
    int numOutputs = p.num_output();

    nvdla::Weights kernelWeights = weightFactory(msg.name(), /*WeightType::*/kGENERIC);
    nvdla::Weights biasWeights = !p.has_bias_term() || p.bias_term() ? weightFactory(msg.name(), /*WeightType::*/kBIAS) : nvdla::Weights(nvdla::DataType::FLOAT, NULL, 0);
    BiasMode biasMode = BiasMode::bNONE;

    if ( biasWeights.count == 0 )
    {
        biasMode = BiasMode::bNONE;
    }
    else if ( biasWeights.count == 1 )
    {
        biasMode = BiasMode::bUNIFORM;
    }
    else if ( biasWeights.count == numOutputs )
    {
        biasMode = BiasMode::bCHANNEL;
    }
    else
    {
        biasMode = BiasMode::bm_ELEMENTWISE;
    }

    return network->addFullyConnected((*tensors)[msg.bottom(0)], numOutputs,
                                      kernelWeights, biasWeights, biasMode);

}


static ILayer* parseReLU(nvdla::INetwork* network, const dc::LayerParameter&msg,
                            RelayWeightFactory& /*weightFactory*/,
                            IBlobNameToTensor * tensors)
{
    return network->addActivation((*tensors)[msg.bottom(0)], /*ActivationType::*/kRELU);
}

static ILayer* parseSoftMax(nvdla::INetwork * network, const dc::LayerParameter&msg,
                                   RelayWeightFactory& /*weightFactory*/, IBlobNameToTensor * tensors)
{
    return network->addSoftMax((*tensors)[msg.bottom(0)]);
}

static ILayer* parseLRN(nvdla::INetwork * network, const dc::LayerParameter&msg,
                               RelayWeightFactory& /*weightFactory*/, IBlobNameToTensor* tensors)
{
    const dc::LRNParameter& p = msg.lrn_param();
    int localSize = p.has_local_size() ? p.local_size() : 5;
    float alpha = p.has_alpha() ? p.alpha() : 1;
    float beta = p.has_beta() ? p.beta() : 5;
    float k = p.has_k() ? p.k() : 1;

    return network->addLRN((*tensors)[msg.bottom(0)], localSize, alpha, beta, k);
}


static ILayer* parsePower(nvdla::INetwork * network, const dc::LayerParameter&msg,
                                 RelayWeightFactory& weightFactory, IBlobNameToTensor *tensors)
{
    const dc::PowerParameter& p = msg.power_param();

    float shift = p.has_shift() ? p.shift() : 0.0f;
    float scale = p.has_scale() ? p.scale() : 1.0f;
    float power = p.has_power() ? p.power() : 1.0f;

    if (power != 1.0f || shift != 0.0f)
    {
        //std::cout << "Relay Parser: shift and power not supported in scale layers" << std::endl;
        return 0;
    }

    bool is16bit = weightFactory.is16bit();
    nvdla::Weights wShift, wScale, wPower;
    if (is16bit)
    {
        float16* t = reinterpret_cast<float16*>(malloc(3 * sizeof(float16)));
        t[0] = float16(shift), t[1] = float16(scale), t[2] = float16(power);
        wShift = nvdla::Weights(nvdla::DataType::HALF, &t[0], 1);
        wScale = nvdla::Weights(nvdla::DataType::HALF, &t[1], 1);
        wPower = nvdla::Weights(nvdla::DataType::HALF, &t[2], 1);
        weightFactory.getTmpAllocs().push_back(t);
    }
    else
    {
        float* t = reinterpret_cast<float*>(malloc(3 * sizeof(float)));
        t[0] = shift, t[1] = scale, t[2] = power;
        wShift = nvdla::Weights(nvdla::DataType::FLOAT, &t[0], 1);
        wScale = nvdla::Weights(nvdla::DataType::FLOAT, &t[1], 1);
        wPower = nvdla::Weights(nvdla::DataType::FLOAT, &t[2], 1);
        weightFactory.getTmpAllocs().push_back(t);
    }


    return network->addScale((*tensors)[msg.bottom(0)], /*ScaleMode::*/sUNIFORM, wShift, wScale, wPower);
}


static ILayer* parseEltwise(nvdla::INetwork * network, const dc::LayerParameter&msg,
                                   RelayWeightFactory& /*weightFactory*/, IBlobNameToTensor * tensors)
{
    const dc::EltwiseParameter& p = msg.eltwise_param();

    ElementWiseOperation op = /*ElementWiseOperation::*/kSUM;
    switch (p.operation())
    {
    case dc::EltwiseParameter_EltwiseOp_SUM: op = /*ElementWiseOperation::*/kSUM; break;
    case dc::EltwiseParameter_EltwiseOp_PROD: op = /*ElementWiseOperation::*/kPROD; break;
    case dc::EltwiseParameter_EltwiseOp_MAX: op = /*ElementWiseOperation::*/ew_kMAX; break;
    }

    return network->addElementWise((*tensors)[msg.bottom(0)], (*tensors)[msg.bottom(1)], op);
}


static ILayer* parseConcat(nvdla::INetwork * network, const dc::LayerParameter&msg,
                                  RelayWeightFactory& /*weightFactory*/, IBlobNameToTensor * tensors)
{
    //const dc::ConcatParameter& p = msg.concat_param(); // TODO: unused

    std::vector<nvdla::ITensor*> ptrs;
    for (unsigned int i = 0, n = msg.bottom_size(); i < n; i++) {
        ptrs.push_back((*tensors)[msg.bottom().Get(i)]);
    }

    return network->addConcatenation(&ptrs[0], msg.bottom_size());
}


static ILayer* parseDeconvolution(nvdla::INetwork * network, const dc::LayerParameter& msg,
                                         RelayWeightFactory& weightFactory, IBlobNameToTensor * tensors)
{
    const dc::ConvolutionParameter& p = msg.convolution_param();
    int numOutputs = p.num_output();

    BiasMode biasMode = BiasMode::bNONE;

    int kernelW = p.has_kernel_w() ? p.kernel_w() : p.kernel_size(0);
    int kernelH = p.has_kernel_h() ? p.kernel_h() : p.kernel_size_size() > 1 ? p.kernel_size(1) : p.kernel_size(0);

    int strideW = p.has_stride_w() ? p.stride_w() : p.stride_size() > 0 ? p.stride(0) : 1;
    int strideH = p.has_stride_h() ? p.stride_h() : p.stride_size() > 1 ? p.stride(1) : p.stride_size() > 0 ? p.stride(0) : 1;

    int padW = p.has_pad_w() ? p.pad_w() : p.pad_size() > 0 ? p.pad(0) : 0;
    int padH = p.has_pad_h() ? p.pad_h() : p.pad_size() > 1 ? p.pad(1) : p.pad_size() > 0 ? p.pad(0) : 0;

    int dilationW = p.dilation_size() > 0 ? p.dilation(0) : 1;
    int dilationH = p.dilation_size() > 1 ? p.dilation(1) : p.dilation_size() > 0 ? p.dilation(0) : 1;

    int numGroups = p.has_group()? p.group() : 1;

    nvdla::Weights kernelWeights = weightFactory(msg.name(), /*WeightType::*/kGENERIC);
    nvdla::Weights biasWeights =
        !p.has_bias_term() || p.bias_term() ?
        weightFactory(msg.name(), /*WeightType::*/kBIAS) :
        nvdla::Weights(nvdla::DataType::FLOAT, NULL, 0);

    if ( biasWeights.count == 0 )
    {
        biasMode = BiasMode::bNONE;
    }
    else if ( biasWeights.count == 1 )
    {
        biasMode = BiasMode::bUNIFORM;
    }
    else if ( biasWeights.count == numOutputs )
    {
        biasMode = BiasMode::bCHANNEL;
    }
    else
    {
        biasMode = BiasMode::bm_ELEMENTWISE;
    }

    nvdla::Dims2 stride = nvdla::Dims2(strideH, strideW);
    nvdla::Dims2 dilation  = nvdla::Dims2(dilationH, dilationW);
    nvdla::Dims2 tlPadding = nvdla::Dims2(padH, padW);
    nvdla::Dims2 brPadding = nvdla::Dims2(padH, padW);
    nvdla::Dims2 kernelSize = nvdla::Dims2(kernelH, kernelW);

    ILayer *layer = network->addDeconvolution((*tensors)[msg.bottom(0)], numOutputs, 0,
                                              kernelSize, tlPadding, brPadding, stride, dilation,
                                              kernelWeights, biasWeights, biasMode, numGroups);

    if (numGroups != 1)
    {
        // std::cout << "Deconvolution layer: groups not supported" << std::endl;
        return 0;
    }

    return layer;
}

static ILayer* parseSigmoid(nvdla::INetwork * network, const dc::LayerParameter&msg,
                                   RelayWeightFactory& /*weightFactory*/, IBlobNameToTensor * tensors)
{
    return network->addActivation((*tensors)[msg.bottom(0)], /*ActivationType::*/kSIGMOID);
}

static ILayer* parseTanH(nvdla::INetwork * network, const dc::LayerParameter&msg,
                                   RelayWeightFactory& /*weightFactory*/, IBlobNameToTensor * tensors)
{
    return network->addActivation((*tensors)[msg.bottom(0)], /*ActivationType::*/kTANH);
}

static ILayer* parseBatchNormalization(nvdla::INetwork * network, const dc::LayerParameter &msg,
                                              RelayWeightFactory& weightFactory, IBlobNameToTensor *tensors)
{
    const dc::BatchNormParameter& p = msg.batch_norm_param();

    nvdla::Weights mean = weightFactory(msg.name(), /*WeightType::*/kMEAN);
    nvdla::Weights variance = weightFactory(msg.name(), /*WeightType::*/kVARIANCE);
    nvdla::Weights movingAverage = weightFactory(msg.name(), /*WeightType::*/kMOVING_AVERAGE);
    float eps = p.eps();
    float scaleFactor = 1.0f;
    float average = 0.0f;
    int i;

    average = *(static_cast<const float*>(movingAverage.values));
    if ( average == 0.0f )
    {
        gLogError << "Batch Normalization moving average is zero " << std::endl;
        return 0;
    }
    scaleFactor /= average;

    if (mean.count != variance.count)
    {
        gLogError << "Mean and variance have differing number of elements " << mean.count << " & " << variance.count << std::endl;
        return 0;
    }

    float *meanBlob = (float *)mean.values;
    float *varianceBlob = (float *)variance.values;

    nvdla::Dims4 inputDims = (*tensors)[msg.bottom(0)]->getDimensions();
    BatchNormMode mode;

    if (mean.count == 1)
    {
        mode = BatchNormMode::bnUNIFORM;
        meanBlob[0] = meanBlob[0] * scaleFactor;
        varianceBlob[0] = varianceBlob[0] * scaleFactor;
    }
    else if (mean.count == inputDims.c)
    {
        mode = BatchNormMode::bnm_CHANNEL;
        for (i = 0; i < mean.count; i++)
        {
            meanBlob[i] = meanBlob[i] * scaleFactor;
            varianceBlob[i] = varianceBlob[i] * scaleFactor;
        }
    }
    else
    {
        gLogError << "Unknown batch norm mode" << std::endl;
        return 0;
    }

    /* DLA hardware expects mean and variance and not scale and shift */
    return network->addBatchNorm((*tensors)[msg.bottom(0)], mode, mean, variance, eps);
}

static ILayer* parseScale(nvdla::INetwork* network, const dc::LayerParameter& msg,
                   RelayWeightFactory& weightFactory, IBlobNameToTensor* tensors)
{
    const dc::ScaleParameter& p = msg.scale_param();

    nvdla::Weights scale = weightFactory(msg.name(), WeightType::kGENERIC);
    nvdla::Weights shift = p.has_bias_term() ? weightFactory(msg.name(), WeightType::kBIAS) : nvdla::Weights(scale.type, NULL, 0);
    nvdla::Weights power = nvdla::Weights(scale.type, NULL, 0);
    nvdla::Dims4 inputDims = (*tensors)[msg.bottom(0)]->getDimensions();
    ScaleMode mode;

    if (msg.bottom_size() > 1)
    {
        gLogError << "Parser can't handle more than 1 inputs to scale op" << std::endl;
        return 0;
    }

    if ( scale.count == 1 )
    {
        mode = ScaleMode::sUNIFORM;
    }
    else if ( scale.count == inputDims.c )
    {
        mode = ScaleMode::sCHANNEL;
    }
    else if ( scale.count == (inputDims.c * inputDims.h * inputDims.w) )
    {
        mode = ScaleMode::sm_ELEMENTWISE;
    }
    else
    {
        gLogError << "Unknown scale mode" << std::endl;
        return 0;
    }

    if ( shift.count > 0 )
    {
        if ( shift.count != scale.count )
        {
            gLogError << "Bias dims not same as scale dims" << std::endl;
            return 0;
        }
    }

    return network->addScale((*tensors)[msg.bottom(0)], mode, shift, scale, power);
}


typedef ILayer*(*LayerParseFn)(nvdla::INetwork *,
                                      const dc::LayerParameter&,
                                      RelayWeightFactory&,
                                      IBlobNameToTensor *);


typedef std::map<std::string, LayerParseFn> LayerParseFnMap;

LayerParseFnMap::value_type gParseTableData[] =
    {
        LayerParseFnMap::value_type("Convolution", parseConvolution),
        LayerParseFnMap::value_type("Pooling", parsePooling),
        LayerParseFnMap::value_type("InnerProduct", parseInnerProduct),
        LayerParseFnMap::value_type("ReLU", parseReLU),
        LayerParseFnMap::value_type("Softmax", parseSoftMax),
        LayerParseFnMap::value_type("SoftmaxWithLoss", parseSoftMax),
        LayerParseFnMap::value_type("LRN", parseLRN),
        LayerParseFnMap::value_type("Power", parsePower),
        LayerParseFnMap::value_type("Eltwise", parseEltwise),
        LayerParseFnMap::value_type("Concat", parseConcat),
        LayerParseFnMap::value_type("Deconvolution", parseDeconvolution),
        LayerParseFnMap::value_type("Sigmoid", parseSigmoid),
        LayerParseFnMap::value_type("TanH", parseTanH),
        LayerParseFnMap::value_type("BatchNorm", parseBatchNormalization),
        LayerParseFnMap::value_type("Scale", parseScale)
    };
const int nelems = sizeof gParseTableData / sizeof gParseTableData[0];
LayerParseFnMap gParseTable( gParseTableData, gParseTableData + nelems);

RelayParser::~RelayParser()
{

    std::vector<void*>::iterator v;
    gLogInfo<<"RelayParser::~RelayParser mTmpAllocs size："<<mTmpAllocs.size()<<std::endl;
    for (v = mTmpAllocs.begin(); v!= mTmpAllocs.end(); v++) {
        free(*v);
    }

    delete mBlobNameToTensor;
}

const IBlobNameToTensor* RelayParser::parse(const char* deployFile,
                                            const char* modelFile,
                                            const tvm::relay::Function& func,
                                            nvdla::INetwork * network)
{

    CHECK_NULL_RET_NULL(deployFile);
    CHECK_NULL_RET_NULL(modelFile);
    assert(mDimsCallback == 0);
    gLogInfo<<"relay parse"<<std::endl;
    if (!mDimsCallback) {
        mDimsCallback = new RelayParserPoolingDimsCallback;//
    }

    network->setPoolingOutputDimensionsFormula(mDimsCallback);

    bool ok = true;
    mBlobNameToTensor = new BlobNameToTensor();
    for (auto arg : func->params) {
        const auto& name = arg->name_hint();
        //LOG(INFO)<<"fucn param name:"<<name;
        //Expr expr = GetRef<Expr>((const VarNode)arg);
        auto checked_type = arg->checked_type(); 
        if (const auto* tensor_type = checked_type.as<TensorTypeNode>()) {
            ShapeVector shape;
            shape = _ShapeToJSON(tensor_type->shape);
            //for(auto dimshape:shape)
            //{
            //    LOG(INFO)<<"relaynetwork input shape:"<<dimshape;
            //}
            //input shape must nhwc but how to know inputshape need advise
            nvdla::Dims4 dims;
            if (shape.size() == 2) {
              dims.n = 1;
              dims.c = shape[1];
              dims.h = shape[0];
              dims.w = 1;
            }
            else if (shape.size() == 3) {
              dims.n = 1;
              dims.c = shape[2];
              dims.h = shape[0];
              dims.w = shape[1];
            }
            else if (shape.size() == 4){
              dims.n = shape[0];
              dims.c = shape[3];
              dims.h = shape[1];
              dims.w = shape[2];
            }
            else if (shape.size() == 1){
              dims.n = 1;
              dims.c = shape[0];
              dims.h = 1;
              dims.w = 1;
            }
	    else if (!shape.size()){
              dims.n = 1;
              dims.c = 1;
              dims.h = 1;
              dims.w = 1;
            }
            else{
                LOG(FATAL) << "riscv not suppose shape size: " << shape.size();
            }
            nvdla::ITensor* tensor = network->addInput(name.c_str(), dims);
            mBlobNameToTensor->add((std::string)name.c_str()/*"data"*/, tensor);
        }
    }
    ICHECK(func.defined()) << "Input error: expect a Relay function.";
    auto sid = GetExtSymbol(func);
    // Record the external symbol for runtime lookup.
    AIPURelay2NetworkCore builder(sid,network,mBlobNameToTensor, &mTmpAllocs);
    auto out = builder.VisitExpr(func->body);
    mBlobNameToTensor->setTensorNames();
    std::cout<<"mtmp data:"<<mTmpAllocs.size()<<std::endl;
    
    return ok? mBlobNameToTensor : 0;
}

int RelayParser::identifyOutputs(nvdla::INetwork * network)
{  ////////
    std::set< nvdla::ITensor* > outputTensors;
    std::set< nvdla::ITensor* > inputTensors;

    for (int l = 0; l < network->getNumLayers(); ++l)
    {
        ILayer* layer = network->getLayer(l);
        if (!layer)
            return -1;

        for (int ii = 0; ii < layer->getNumInputs(); ++ii) {
            inputTensors.insert(layer->getInput(ii));
        }

        for (int oo = 0; oo < layer->getNumOutputs(); ++oo)
        {
            outputTensors.insert(layer->getOutput(oo));
        }
    }

    for (std::set<nvdla::ITensor*>::iterator oi = outputTensors.begin(); oi != outputTensors.end(); ++oi)
    {
        // an output tensor which is not an input to any other layers is a network output tensor
        if (inputTensors.find(*oi) == inputTensors.end())
        {
            network->markOutput(*oi);
            gLogInfo << "mark " << (*oi)->getName() << std::endl;
        }
    }

    return network->getNumOutputs();
}

BinaryProtoBlob::BinaryProtoBlob(void* memory, nvdla::DataType type, nvdla::Dims4 dimensions) :
    mMemory(memory), mDataType(type), mDimensions(dimensions)
{
}

nvdla::Dims4 BinaryProtoBlob::getDimensions()
{
    return mDimensions;
}

const void* BinaryProtoBlob::getData()
{
    return mMemory;
}

void BinaryProtoBlob::destroy()
{
    delete this;
}

BinaryProtoBlob::~BinaryProtoBlob()
{
    free(mMemory);
}

} // tvm::relay::contrib::aipu::priv
}//tvm::relay::contrib::aipu
} //tvm::relay::contrib
} // tvm::relay
} // tvm::

