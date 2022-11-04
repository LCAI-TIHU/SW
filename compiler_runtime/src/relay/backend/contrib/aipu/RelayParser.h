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
#include "IRelayParser.h"
#include <tvm/relay/function.h>

namespace ditcaffe{
class NetParameter;
}

namespace tvm {
namespace relay {
namespace contrib {
namespace aipu{
namespace priv {

class BlobNameToTensor : public IBlobNameToTensor
{
public:
    virtual void add(const std::string& name, nvdla::ITensor* tensor);

    virtual nvdla::ITensor* find(const char* name) const;
    virtual nvdla::ITensor*& operator[](const std::string& name);

    virtual void setTensorNames();
    virtual ~BlobNameToTensor();

private:
    std::map<std::string, nvdla::ITensor*> mMap;
};

class BinaryProtoBlob : public IBinaryProtoBlob
{
public:
    BinaryProtoBlob(void* memory, nvdla::DataType type, nvdla::Dims4 dimensions);

    const void*	getData();
    nvdla::Dims4 getDimensions();
    void	destroy();
protected:
    void* mMemory;
    nvdla::DataType mDataType;
    nvdla::Dims4 mDimensions;
    virtual ~BinaryProtoBlob();
};

class RelayParser;

class RelayParserFactory
{
public:
    typedef nvdla::compiler::priv::PrivPair<IRelayParser *, RelayParser *> RelayParserPrivPair;

    static RelayParserPrivPair newRelayParser();
    static NvDlaError deleteRelayParser(IRelayParser *parser);

    static RelayParser *priv(IRelayParser *);
    static IRelayParser *i(RelayParser *);
    static IRelayParser *self(void *);

protected:
    static nvdla::compiler::priv::BiMap<IRelayParser *, RelayParser *> s_priv;
    static nvdla::compiler::priv::BiMap<void *, IRelayParser *> s_self;
};


class RelayParser : public IRelayParser
{
public:
    RelayParser() :
        IRelayParser(),
        mDeploy(NULL),
        mModel(NULL),
        mTmpAllocs(),
        mDimsCallback(NULL),
        mBlobNameToTensor(NULL),
        mProtobufBufferSize(1024 << 20)
    { }

    virtual const IBlobNameToTensor* parse(const char* deploy,
                                           const char* model,
                                           const tvm::relay::Function& func,
                                           nvdla::INetwork* network);
    virtual int identifyOutputs(nvdla::INetwork * network);
    virtual ~RelayParser();

    void setProtobufBufferSize(size_t size) { mProtobufBufferSize = size; }

    // read a blob from a protobuf file (typically a mean blob)
    static BinaryProtoBlob* parseBinaryProto(const char* fileName);

    static void shutdownProtobufLibrary();
private:
    ditcaffe::NetParameter * mDeploy;
    ditcaffe::NetParameter * mModel;
    std::vector<void*> mTmpAllocs;
    nvdla::INetwork::OutputDimensionsFormula* mDimsCallback;
    IBlobNameToTensor* mBlobNameToTensor;
    size_t mProtobufBufferSize;
};

}
}
}
}
}

#endif // NVDLA_PRIV_RELAY_PARSER_H
