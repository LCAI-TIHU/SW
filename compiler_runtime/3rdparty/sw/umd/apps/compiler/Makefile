# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

ROOT := $(TOP)

MODULE := nvdla_compiler

include $(ROOT)/make/macros.mk

BUILDOUT ?= $(ROOT)/out/apps/compiler
BUILDDIR := $(BUILDOUT)/$(MODULE)
TEST_BIN := $(BUILDDIR)/$(MODULE)

INCLUDES :=
MODULE_COMPILEFLAGS := -W -Wall -Wno-multichar -Wno-unused-parameter -Wno-unused-function -Werror-implicit-function-declaration
MODULE_CFLAGS := --std=c99
MODULE_CPPFLAGS := --std=c++11 -fexceptions -fno-rtti
# PROTOBUF_LIBRARY := $(ROOT)/apps/compiler/libprotobuf.a
PROTOBUF_LIBRARY := $(ROOT)/external/protobuf-2.6/src/.libs/libprotobuf.so

all:: $(TEST_BIN)

include rules.mk

# the logic to compile and link stuff is in here
$(TEST_BIN): $(ALLMODULE_OBJS) $(SHARED_LIBS)
	@echo building $(MODULE)  $@
	# $(info LOCAL_DIR $(LOCAL_DIR))
	g++ $(ALLMODULE_OBJS) -L$(ROOT)/out/core/src/compiler/libnvdla_compiler -lnvdla_compiler ${PROTOBUF_LIBRARY} -pthread -o $@ ${PROTOBUF_LIBRARY} -Wl,-rpath=.
