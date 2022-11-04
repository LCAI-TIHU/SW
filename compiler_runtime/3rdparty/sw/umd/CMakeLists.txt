cmake_minimum_required(VERSION 3.2)
project(nvdla VERSION 1.0)
#set(CMAKE_C_COMPILER gcc)
#set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

set(NVDLA_TOP .)


find_library(NVDLA_PROTOBUF
    NAMES protobuf
    HINTS ./external/protobuf-2.6/src/.libs)

# nvdla sw compiler
#######################################
set(NVDLA_SW_COMPILER_INCLUDES 
    ${NVDLA_TOP}/core/src/compiler/include  
    ${NVDLA_TOP}/core/include 
    ${NVDLA_TOP}/external/include 
    ${NVDLA_TOP}/port/linux/include
    ${NVDLA_TOP}/core/src/common/include 
    ${NVDLA_TOP}/external/protobuf-2.6/include
    )

file(GLOB NVDLA_SW_COMPILER_SRCS_CPP
    ${NVDLA_TOP}/core/src/compiler/*.cpp
    ${NVDLA_TOP}/core/src/compiler/engine-ast/*.cpp
    ${NVDLA_TOP}/core/src/compiler/caffe/CaffeParser.cpp
    ${NVDLA_TOP}/core/src/compiler/caffe/ditcaffe/protobuf-2.6.1/ditcaffe.pb.cpp
    ${NVDLA_TOP}/core/src/common/EMUInterface.cpp
    ${NVDLA_TOP}/core/src/common/EMUInterfaceA.cpp
    ${NVDLA_TOP}/core/src/common/Loadable_compiler.cpp
    )
file(GLOB NVDLA_SW_COMPILER_SRCS_C
    ${NVDLA_TOP}/utils/*.c
    ${NVDLA_TOP}/port/linux/nvdla_os.c
    )
# intermediate c target
add_library(nvdla_sw_compiler_c STATIC ${NVDLA_SW_COMPILER_SRCS_C})
#target_compile_options(nvdla_sw_compiler_c BEFORE PRIVATE -g -finline -W -Wall -Wno-multichar -Wno-unused-parameter -Wno-unused-function -Werror-implicit-function-declaration)
target_compile_options(nvdla_sw_compiler_c BEFORE PRIVATE -g -finline)
target_include_directories(nvdla_sw_compiler_c PUBLIC ${NVDLA_SW_COMPILER_INCLUDES})
target_compile_options(nvdla_sw_compiler_c PRIVATE -DNVDLA_UTILS_ERROR_TAG="DLA")
target_compile_options(nvdla_sw_compiler_c PRIVATE -fPIC)

# compiler target
add_library(EXTERN_LIBRARY_NVDLACOMPILER STATIC ${NVDLA_SW_COMPILER_SRCS_CPP})
target_include_directories(EXTERN_LIBRARY_NVDLACOMPILER PUBLIC ${NVDLA_SW_COMPILER_INCLUDES})
#target_compile_options(EXTERN_LIBRARY_NVDLACOMPILER BEFORE PRIVATE -g -finline -W -Wall -Wno-multichar -Wno-unused-parameter -Wno-unused-function -Werror-implicit-function-declaration)
target_compile_options(EXTERN_LIBRARY_NVDLACOMPILER BEFORE PRIVATE -g -finline)
target_compile_options(EXTERN_LIBRARY_NVDLACOMPILER BEFORE PRIVATE -DNVDLA_UTILS_ERROR_TAG="DLA")
target_compile_options(EXTERN_LIBRARY_NVDLACOMPILER BEFORE PRIVATE -DGOOGLE_PROTOBUF_NO_RTTI)
target_compile_options(EXTERN_LIBRARY_NVDLACOMPILER BEFORE PRIVATE -DNVDLA_COMPILER_OUTPUT_FOR_PROTOTEST)
target_link_libraries(EXTERN_LIBRARY_NVDLACOMPILER PUBLIC nvdla_sw_compiler_c PUBLIC ${NVDLA_PROTOBUF})

# nvdla sw runtime
#######################################
set(NVDLA_SW_RUNTIME_INCLUDES 
    ${NVDLA_TOP}/core/include 
    ${NVDLA_TOP}/core/src/common/include 
    ${NVDLA_TOP}/core/src/runtime/include 
    ${NVDLA_TOP}/port/linux/include
    ${NVDLA_TOP}/external/include
    ${NVDLA_TOP}/external/protobuf-2.6/include
    ${NVDLA_TOP}/apps/runtime
    )

file(GLOB NVDLA_SW_RUNTIME_SRCS_CPP
    ${NVDLA_TOP}/core/src/common/Check.cpp
    ${NVDLA_TOP}/core/src/common/EMUInterface.cpp
    ${NVDLA_TOP}/core/src/common/EMUInterfaceA.cpp
    ${NVDLA_TOP}/core/src/common/Loadable_runtime.cpp
    ${NVDLA_TOP}/core/src/runtime/Emulator.cpp
    ${NVDLA_TOP}/core/src/runtime/Runtime.cpp
    ${NVDLA_TOP}/apps/runtime/DlaImage.cpp
    ${NVDLA_TOP}/apps/runtime/DlaImageUtils.cpp
    ${NVDLA_TOP}/apps/runtime/RuntimeTest.cpp
    ${NVDLA_TOP}/apps/runtime/TestUtils.cpp
    )
file(GLOB NVDLA_SW_RUNTIME_SRCS_C
    ${NVDLA_TOP}/core/src/common/ErrorLogging.c
    ${NVDLA_TOP}/port/linux/nvdla.c
    ${NVDLA_TOP}/port/linux/nvdla_os.c
    )

# intermediate c target
add_library(nvdla_sw_runtime_c STATIC ${NVDLA_SW_RUNTIME_SRCS_C})
target_include_directories(nvdla_sw_runtime_c PUBLIC ${NVDLA_SW_RUNTIME_INCLUDES})
#target_compile_options(nvdla_sw_runtime_c BEFORE PRIVATE -g -finline -W -Wall -Wno-multichar -Wno-unused-parameter -Wno-unused-function -Werror-implicit-function-declaration)
target_compile_options(nvdla_sw_runtime_c BEFORE PRIVATE -g -finline)
target_compile_options(nvdla_sw_runtime_c PRIVATE -DNVDLA_UTILS_ERROR_TAG="DLA_RUNTIME")
target_compile_options(nvdla_sw_runtime_c PRIVATE -fPIC)

# runtime target
add_library(EXTERN_LIBRARY_NVDLARUNTIME STATIC ${NVDLA_SW_RUNTIME_SRCS_CPP})
target_include_directories(EXTERN_LIBRARY_NVDLARUNTIME PUBLIC ${NVDLA_SW_RUNTIME_INCLUDES})
#target_compile_options(EXTERN_LIBRARY_NVDLARUNTIME BEFORE PRIVATE -g -finline -W -Wall -Wno-multichar -Wno-unused-parameter -Wno-unused-function -Werror-implicit-function-declaration)
target_compile_options(EXTERN_LIBRARY_NVDLARUNTIME BEFORE PRIVATE -g -finline)
target_compile_options(EXTERN_LIBRARY_NVDLARUNTIME PRIVATE -DNVDLA_UTILS_ERROR_TAG="DLA_RUNTIME")
target_link_libraries(EXTERN_LIBRARY_NVDLARUNTIME PUBLIC nvdla_sw_runtime_c PUBLIC ${NVDLA_PROTOBUF})
