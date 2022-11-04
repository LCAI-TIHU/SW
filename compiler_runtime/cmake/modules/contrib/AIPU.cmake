# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_AIPU STREQUAL "ON")
  find_library(EXTERN_LIBRARY_PROTOBUF
    NAMES protobuf
    HINTS ${NVDLASW_PATH}/umd/external/protobuf-2.6/src/.libs)
  if(NOT EXTERN_LIBRARY_PROTOBUF)
    message(FATAL_ERROR "can't find protobuf library, need to install NVDLA UMD")
  endif()
  find_library(EXTERN_LIBRARY_JPEG 
    NAMES jpeg)
  if(NOT EXTERN_LIBRARY_JPEG)
    message(FATAL_ERROR "can't find libjpeg")
  endif()
  set(NVDLA_INCLUDE_DIRS ${NVDLASW_PATH}/umd/core/src/compiler/caffe ${NVDLASW_PATH}/umd/core/src/compiler/include ${NVDLASW_PATH}/umd/core/include ${NVDLASW_PATH}/umd/port/linux/include ${NVDLASW_PATH}/umd/apps/compiler ${NVDLASW_PATH}/umd/external/include ${NVDLASW_PATH}/umd/external/protobuf-2.6/src ${NVDLASW_PATH}/umd/apps/runtime)
  #include_directories(${NVDLA_INCLUDE_DIRS})
  file(GLOB AIPU_RELAY_CONTRIB_SRC src/relay/backend/contrib/aipu/*.cc)
  list(APPEND COMPILER_SRCS ${AIPU_RELAY_CONTRIB_SRC})
  message(STATUS "Build with aipu codegen" )
  file(GLOB AIPU_CONTRIB_SRC src/runtime/contrib/aipu/*.cc)
  list(APPEND RUNTIME_SRCS ${AIPU_CONTRIB_SRC})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_LIBRARY_PROTOBUF} ${EXTERN_LIBRARY_JPEG})
  message(STATUS "Build with aipu runtime" )

  set(COMPILE_NVDLA "ON")
  if(COMPILE_NVDLA STREQUAL "ON")
    # new approach: compile nvdla lib from scratch
    add_subdirectory(./3rdparty/sw/umd)
  else()
    # original approach: use the precompiled nvdla lib
    find_library(EXTERN_LIBRARY_NVDLACOMPILER
      NAMES nvdla_compiler
      HINTS ${NVDLASW_PATH}/umd/out/core/src/compiler/libnvdla_compiler)
    if(NOT EXTERN_LIBRARY_NVDLACOMPILER)
      message(FATAL_ERROR "can't find NVDLA compiler library, need to install NVDLA UMD")
    endif()
    find_library(EXTERN_LIBRARY_NVDLARUNTIME
      NAMES nvdla_runtime
      HINTS ${NVDLASW_PATH}/umd/out/core/src/runtime/libnvdla_runtime)
    if(NOT EXTERN_LIBRARY_NVDLARUNTIME)
      message(FATAL_ERROR "can't find NVDLA runtime library, need to install NVDLA UMD")
    endif()
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_LIBRARY_NVDLACOMPILER} ${EXTERN_LIBRARY_NVDLARUNTIME})
  endif()
endif()

