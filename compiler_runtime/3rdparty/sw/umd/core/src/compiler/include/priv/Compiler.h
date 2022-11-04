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

#ifndef NVDLA_PRIV_COMPILER_H
#define NVDLA_PRIV_COMPILER_H

#include <vector>
#include <map>
#include <algorithm>
#include <fstream>

#include "priv/Type.h"
#include "priv/Check.h"

#include "nvdla/ICompiler.h"

#include "priv/Memory.h"
#include "priv/Loadable_compiler.h"
#include "priv/CanonicalAST.h"
#include "priv/EngineAST.h"


namespace nvdla
{

namespace priv
{

class Compiler;
class Loadable;

class CompilerFactory
{
public:
    typedef nvdla::compiler::priv::PrivPair<ICompiler *, Compiler*> CompilerPrivPair;

    static CompilerPrivPair newCompiler();
    static NvDlaError deleteCompiler(ICompiler *compiler);

    static Compiler *priv(ICompiler *);
    static ICompiler *i(Compiler *);
    static ICompiler *self(void *s);

protected:
    static nvdla::compiler::priv::BiMap<ICompiler *, Compiler *> s_priv;
    static nvdla::compiler::priv::BiMap<void *, ICompiler *> s_self;
};


class DLAInterface;

class Compiler : public ICompiler
{
public: // externally facing
    virtual IWisdom *wisdom() const;

    virtual NvDlaError getDataType(DataType::UnderlyingType *d) const;
    virtual NvDlaError compile(const char *profile_name, const char *target_config_name, ILoadable **); // "" := default
    virtual NvDlaError getLoadableImage(const char *profile_name, NvU8 *flatbuf);
    virtual NvDlaError getLoadableImageSize(const char *profile_name, NvU64 *size);

public: // internally facing

    NvDlaError emit(engine_ast::Graph * g, LoadableFactory::LoadablePrivPair &);

    Compiler();
    virtual ~Compiler();

    void setWisdom(Wisdom *w) { m_wisdom = w; }

    virtual NvU16 getFactoryType() const;

    inline bool debugVersions() const { return false; }
    inline bool debugTasks() const { return false; }
    inline bool debugMemoryLayout() const { return false; }
    inline bool debugGraphs() const { return false; }
    inline bool debugProfile() const {  return false; }

protected:

    friend class Wisdom;
    friend class CompilerFactory;

    Wisdom *m_wisdom;


    /**
     * @Purpose:-> //xxx: mostly can this phase as the duty is spread out to other ops
     *      As a result of splits, fusions and introducing multiple ROIs,
     *      the graph might have several nodes that need special treatment:
     *          - intra-roi nodes might need joins
     *          - nodes around a split node might have to be split themselves
     *          - etc (as we find)
     *      Massage the graph to bound/tighten the nodes from either sides.
     */
    // not implemented
    engine_ast::Graph *boundGraph(engine_ast::Graph *);

    /**
     * @Purpose:->
     *      # Resolve all types of dependencies:
     *          - data dependencies between nodes exchanging tensor
     *          - compute dependencies between nodes of same engine
     *          - software dependencies around software (no h/w required) ops like
     *            split & concat
     *      # Determine task boundaries (DLA/EMU/DLA/etc)
     *      # And generate node annotation order within each task so that it
     *        represents functional data-flow among nodes and allows chronological
     *        memory allocation for each of them
     *      # Finally, inherit the dependency graph state generated for 1 batch
     *        into that of the 'N' multiple batches (if N > 1).
     */
    engine_ast::Graph *generateDependencyParams(engine_ast::Graph *, engine_ast::NodeSequence &);

    /**
     * @Purpose:->
     *      Schedule/reserve memory resources.
     */
    engine_ast::Graph *resolveMemory(engine_ast::Graph *, const engine_ast::NodeSequence &);

    /**
     * @Purpose:->
     *      This compilation phase adds debug bdmas to copy intermediate
     *      surfaces to sysmem.  Each such surface is presented
     *      as a bindable debug buffer to the runtime.
     */
    engine_ast::Graph *enableCopyOutDebugSurfaces(engine_ast::Graph *);

    DLAInterface *getTargetDLAInterface(Profile *);
    EMUInterface *getTargetEMUInterface(Profile *);

    /**
     * Internal functions which are unsafe and external interfaces wraps them
     * to catch possible exception thrown.
     **/
    NvDlaError compileInternal(Profile *, TargetConfig *, ILoadable **);
    NvDlaError getLoadableImageInternal(const char *profile_name, NvU8 *flatbuf);
    NvDlaError getLoadableImageSizeInternal(const char *profile_name, NvU64 *size);

private:
    NvDlaError getLoadableFromWisdom(const char *test_point_name,
                                    ILoadable **i);
};


class DumpGraphBase
{
public:
    DumpGraphBase(const std::string &filename, const std::string &graphId) :
        _m_filename(filename), _m_graph_id(graphId) { } // don't write file by default

    virtual ~DumpGraphBase() { }

    virtual void setFilename(const std::string s) { _m_filename = s; }
    virtual void setGraphId(const std::string i)  { _m_graph_id = i; }
    virtual std::ofstream &out() { return _m_file; }

protected:
    std::string _m_filename;
    std::string _m_graph_id;
    std::ofstream _m_file;
    std::string _m_delim;
};


} // nvdla::priv

} // nvdla

#endif // NVDLA_PRIV_COMPILER_H
