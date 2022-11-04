/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include <tvm/runtime/container.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include "aipu_runtime.h" 
#include "aipu_ioctl.h" 
#include <iostream>
#include <cstdlib> // system
#include <math.h>
#include "half.h"
#include <stdio.h> // snprintf
#include <map> 

extern "C" {
#include "jpeglib.h"
}

#include <sstream>
#include <fstream>
#include <algorithm>
#include <malloc.h>
#include <sys/ioctl.h> 
#include <cstdio> // snprintf, fopen
#include <string.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include <vector> 
#include <time.h>
#include <sys/time.h>
#include <iomanip>

#define DEBUG_RUNTIME 0

namespace tvm {
namespace runtime {
namespace contrib {

using namespace std;
using namespace half_float;
using namespace tvm::runtime;
using namespace tvm::runtime::contrib;

AIPUModuleNode *aipu_node;
void *image_dat = NULL; 
// std::vector<std::pair<uint64_t, uint8_t*>> mac_tasks;
// std::vector<std::pair<std::vector<size_t>, std::vector<cpu_param *>>> riscv_tasks;
// std::vector<int> execute_order;
// std::map<std::string, std::pair<size_t, size_t>> input_param;
// std::map<std::string, std::pair<size_t, size_t>> output_param;
// std::vector<Fused_function_offsets> task_io_param;
Loadable_pair mac_tasks;
Riscv_vector riscv_tasks;
Execution_order execute_order;
Riscv_addr_list riscv_addr_list;
Riscv_wt_list riscv_wt_list;
Network_io input_param, output_param;
std::vector<Fused_function_offsets> task_io_param;

struct nvdla::IRuntime::AipuConfig aipu_config;// used as mac parameters to call UMD funtions;

struct AipuTask aipu_task; // aipu task need to write to firmware

    
// Get all devices for the host and other runtime devices.
std::vector<Device> GetAllDevice(const TVMArgs& args, int dev_start_arg) {
	// Reserve the first item as the fallback device.
	std::vector<Device> ret;
	Device dev;
	for (int i = dev_start_arg; i < args.num_args; i += 2) {
        if(DEBUG_RUNTIME) {
		    LOG(INFO)<<"the args index is: "<<i;
        }
		// LOG(INFO)<<"the dev_type is: "<<args[2];
		int dev_type = args[i];
		dev.device_type = static_cast<DLDeviceType>(dev_type);
		dev.device_id = args[i + 1];
        if(DEBUG_RUNTIME) {
            LOG(INFO)<<"the args index is "<<(i+1);
            LOG(INFO)<<"the device_id is "<<(dev.device_id);
        }
		ret.push_back(dev);
	}
	return ret;
}

static inline size_t GetDataSize(const DLTensor* t) {
  size_t size = 1;
  if(DEBUG_RUNTIME) {
    LOG_INFO << "\n TVM tensor ndim is " << t->ndim
      << "\n TVM tensor shape is " << t->shape[0]
      << " x " << t->shape[1]
      << " x " << t->shape[2]
      << " x " << t->shape[3];
  }
  for (tvm_index_t i = 0; i < t->ndim; ++i) {
    size *= t->shape[i];
  }
  size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
  return size;
}



runtime::Module AipuRuntimeCreate(const std::string& sym_json, const tvm::runtime::Module& m, const std::vector<Device>& devs) 
{
    aipu_node =     static_cast<AIPUModuleNode*>(Module(m).operator->());
    if(DEBUG_RUNTIME) {
        LOG_INFO << "\n1 Got aipu module node" << "\n";
        LOG_INFO << "====================COMPILER DEBUG INFO BEGIN =====================================";
    }
	
    mac_tasks =     aipu_node->GetLoadable();// mac task
    if(DEBUG_RUNTIME) {
        LOG_INFO << "\n\tGot mac_tasks, mac tasks number is "
            << mac_tasks.size() << ". Loadable address and size:";
        for (uint32_t i = 0; i < mac_tasks.size(); i++) {
            printf("%d  --  loadable address:    0x%08x, loadable size: %ld.\n", i, (uint64_t)(mac_tasks[i].second), mac_tasks[i].first);
        }
    }
	
	riscv_tasks =   aipu_node->GetRiscv();// riscv task
    if(DEBUG_RUNTIME) {
        LOG_INFO << "\n\tGot riscv_tasks, task number is "
            << riscv_tasks.size() << "\n";
    }

	execute_order = aipu_node->GetExecutionOrder();// task execute order
    if(DEBUG_RUNTIME) {
        LOG_INFO << "\n\tGot execute_order, size is "
            << execute_order.size();
        for (uint32_t i = 0; i < execute_order.size(); i++) {
            printf("    %d", execute_order[i]);
        }
        printf("\n");
    }

    riscv_addr_list = aipu_node->GetRiscvAddrList();
    if(DEBUG_RUNTIME) {
        LOG_INFO << "\n\tGot riscv_addr_list, size is "
            << riscv_addr_list.size() << "\n";
        for (uint32_t i = 0; i < riscv_addr_list.size(); i++) {
            printf("%d  task address list is [wt_index, offset]:\n\t{", i);
            for (uint32_t j = 0; j < riscv_addr_list[i].size(); j++) {
                printf("[%d, 0x%08x], ", riscv_addr_list[i][j].first, riscv_addr_list[i][j].second);
            }
            printf(" }\n");
        }
    }

    riscv_wt_list = aipu_node->GetRiscvWtList();
    if(DEBUG_RUNTIME) {
        LOG_INFO << "\n\tGot riscv_wt_list, size is "
            << riscv_wt_list.size() << "\n";
        printf("task wt list is [size, address]:\n\t{");
        for (uint32_t i = 0; i < riscv_wt_list.size(); i++) {
             printf("[%ld, 0x%08x], ", riscv_wt_list[i].first, (uint64_t)(riscv_wt_list[i].second));
        }
        printf(" }\n");
    }
	
	task_io_param = aipu_node->GetFusedOffset(); // task io offset
    if (task_io_param.size() != execute_order.size()) {
        LOG_ERROR << "\nAIPU MODULE NODE ERROR";
    } else {
        if(DEBUG_RUNTIME) {
            LOG_INFO << "\tGot task_io_param, size is "
                << task_io_param.size() << "Fused_function_offsets is:";
        }
    }
    if(DEBUG_RUNTIME) {
        for (uint32_t i = 0; i < task_io_param.size(); i++) {
            printf("%d -- Fused_funtion_offsets {\n", i);
            printf("\tinput_offsets:");
            for (uint32_t j = 0; j < task_io_param[i].input_offsets.size(); j++) {
                printf(" 0x%08x,", task_io_param[i].input_offsets[j]);
            }
            printf("\n\toutput_offsets:");
            for (uint32_t j = 0; j < task_io_param[i].output_offsets.size(); j++) {
                printf(" 0x%08x,", task_io_param[i].output_offsets[j]);
            }
            //printf("\n\tinput_fused:");
            //for (uint32_t j = 0; j < task_io_param[i].input_fused.size(); j++) {
            //    printf(" %d,", task_io_param[i].input_fused[i]);
            //}
            printf("\n\t}\n");

        }
    }

    input_param =   aipu_node->GetInputParam();
    if (input_param.size() == 0) {
        LOG_ERROR << "Can not get input parameters.";
    } else {
        if(DEBUG_RUNTIME) {
            for (auto i = input_param.begin(); i != input_param.end(); i++) {
                LOG_INFO << "\nInput: " << "\n\tkey = " << i->first << "\n\toffset = 0x" << hex << i->second.first << "\n\tsize = 0x" 
                    << hex << i->second.second; 
            }
        }
    }

	output_param =  aipu_node->GetOutputParam(); 
    if (output_param.size() == 0) {
        LOG_ERROR << "\nError： Can not get output parameters.";
    } else {
        if(DEBUG_RUNTIME) {
            for (auto i = output_param.begin(); i != output_param.end(); i++) {
                LOG_INFO << "\nOutput: " << "\n\tkey = " << i->first << "\n\toffset = 0x" << hex << i->second.first << "\n\tsize = 0x" 
                    << hex << i->second.second; 
            }
        }
    }
    if(DEBUG_RUNTIME) {
        LOG_INFO << "====================COMPILER DEBUG INFO END =====================================";
    }
    aipu_config.net_parse_done = false;
    
    return Module(m);
}

runtime::Module AipuSetInput(const tvm::runtime::Module& m, string str, TVMArrayHandle arr) {
    size_t input_offset = 0;
    ssize_t input_size = 0;
    // test input key
    if (str.length() == 0) {
        LOG_ERROR << "\nGot wrong input data paramters." << "\n"; 
    } else {
        if(DEBUG_RUNTIME) {
            LOG_INFO << "\nSet input with input string : " << str;
        }
    }
    // tesk input data
    if (arr->data == NULL) {
        LOG_ERROR << "\nGot wrong input data paramters" << "\n"; 
    } else {
        if(DEBUG_RUNTIME) {
            LOG_INFO << "\nInput data shape is " << arr->shape[0] << " x " << arr->shape[1] << " x " 
                << arr->shape[2] << " x " << arr->shape[3] 
                << "\nTVMArray data address 0x" << hex << arr->data
                << "\nTVMArray data offset is " << arr->byte_offset; 
        }
    }
    // Get input offset and size
    bool find_input = false;
    for (auto i = input_param.begin(); i != input_param.end(); i++) {
        if (i->first == str) {
            input_offset = i->second.first;
            input_size = i->second.second;
            find_input = true;
            if(DEBUG_RUNTIME) {
                LOG_INFO << "\nInput: " << "key = " << i->first << ", offset = " << dec << i->second.first 
                    << ", size = " << dec << i->second.second;
            }
            break;
        } else {
            find_input = false;
            if(DEBUG_RUNTIME) {
                LOG_INFO << "\nInput str " << str << " != " << i->first; 
            }
        }
    }
    
    // check find_input
    if (!find_input) {
        LOG_ERROR << "Can not find input, check input parameters.\n";
    }
    size_t arr_size = GetDataSize(arr);
    if(DEBUG_RUNTIME) {
        LOG_INFO << "arr_size is " << arr_size << ", input_size is " << input_size;
    }
    if (arr_size >= input_size) {
        input_size = arr_size;
    }
    void *src = (void *)((uint64_t)arr->data + (uint64_t)arr->byte_offset);
    void *dst = (void *)(COMPILER_BASE_ADDR + input_offset);
    
    //if mac is first task, DlaSetInput should be used to load image
    //image load occur in AipuRun()
    if (execute_order[0] > 0) {
        // if first task is mac task, the input_parm size should be 1
        if (input_param.size() != 1) {
            LOG_ERROR << "Check network configuration, input size should be 1.\n";
        }
        image_dat = (char *)malloc(arr_size);
        memcpy((void *)image_dat, (void *)src, arr_size);
        // attention: only set the first mac, do not forget to set following mac parameters.
        aipu_config.mac_is_first_task = true;
        aipu_config.image_addr        = image_dat;
        aipu_config.input_num         = 1;
        aipu_config.input_addr[0]     = (uint64_t)dst;
    } else {
        aipu_config.mac_is_first_task = false;
        aipu_config.image_addr = NULL;
        int32_t rc =  NvDlaWrite(dst, src, input_size);
        if(DEBUG_RUNTIME) {
            LOG_INFO << std::endl << "Write from host " << hex << src << " to target " << hex << dst << ", size is 0x" << input_size << std::endl;
        }
        if (rc != input_size) {
            LOG_ERROR << "AipuSetInput Error!!!" << " Write from 0x" << hex 
                << src << " to 0x" << hex << dst << ", size is 0x" << input_size << "\n";
        }
    }
    return Module(m);
}

runtime::Module AipuGetOutput(const tvm::runtime::Module& m, int index, DLTensor* data_out) { 
    auto i = output_param.begin();
    for (int j = 0; j < index; j++) {
        i++;
    }
    size_t arr_size = GetDataSize(data_out);
    size_t output_offset = i->second.first;
    size_t output_size = i->second.second;
    if(DEBUG_RUNTIME) {
        LOG_INFO << "\nTvm data bits is " << data_out->dtype.bits 
            << "\nTvm data size is " << dec << arr_size
            << "\nTvm data dim is " << dec << data_out->ndim
            << "\nTvm data shape is " << data_out->shape[0] 
            << " x " <<  data_out->shape[1] 
            << " x " << data_out->shape[2] 
            << " x " << data_out->shape[3];
    
        LOG_INFO << "\nOutput index is " << index
            << "\nOutput size is " << dec <<output_size
            << "\nOutput offset is 0x" << hex << output_offset;
    }
    
    /* if (arr_size >= output_size) { */
    /*     output_size = arr_size; */
    /* } */
    ssize_t shape_size = (data_out->ndim == 4) ? data_out->shape[0] * data_out->shape[1] * data_out->shape[2] * data_out->shape[3] :
                        (data_out->ndim == 3) ? data_out->shape[0] * data_out->shape[1] * data_out->shape[2] :
                        (data_out->ndim == 2) ? data_out->shape[0] * data_out->shape[1] : data_out->shape[0];
    void *dst = (void *)malloc(output_size);
    void *src = (void *)(COMPILER_BASE_ADDR + output_offset);
    int32_t rc =  NvDlaRead(dst, src, output_size);
    if (rc != output_size) {
        LOG_ERROR << "AipuGetOutput Error!!!" << " Read from 0x" << hex 
            << src << " to 0x" << hex << dst << ", size is " << dec << output_size << "\n";
    } else {
        void *tvm_dst = (void *)((uint64_t)data_out->data + (uint64_t)data_out->byte_offset); 
        memcpy(tvm_dst, dst, arr_size);//shold be arr_size
        /*
        if(DEBUG_RUNTIME) {
            printf("Output data is :\n\t[");
            for (int i = 0; i < shape_size; i++) {
                if (arr_size/shape_size >= 4.0)
                    printf(" %f,", ((float *)dst)[i]);
                else 
                    printf(" %d,", ((char *)dst)[i]);
            }
            printf("]\n");
        LOG_INFO << "\nAipuGetOutput successed.\n"
            << "\tRead from 0x" << hex << (uint64_t)src
            << "\n\tSize is " << dec << arr_size << " bytes"
            <<"\n========Runtime done.===========\n";
        }
        */
    }
    free(dst);
    return Module(m);
}

static int32_t NetworkParse() {
    int32_t rc = 0;
    /*init aipu config for mac */
    {
        aipu_config.aipu_mode = true;
        aipu_config.base_addr = DLA_BASE_ADDR; 
        aipu_config.addr_offset = 0;
        aipu_config.loadable.first = NULL;
        aipu_config.loadable.second = 0;

        aipu_task.next = (uint64_t)AIPU_TASK_ADDR;
    }

    uint32_t mac_task_cnt = 0;
    uint32_t cpu_task_cnt = 0;

    for (uint32_t exe_order = 0; exe_order < execute_order.size(); exe_order++) {
        /* size_t input_offset = task_io_param[exe_order].input_offsets[0]; */ 
        size_t output_offset = task_io_param[exe_order].output_offsets[0];    
        // mac task parse
        /* printf("exe_order = %d.\n", exe_order); */
        if (execute_order[exe_order] > 0) {
            if(DEBUG_RUNTIME) {
                LOG_INFO << "\n=== Execute_order : " << exe_order << ": =================== MAC TASK===============\n";
            }
            uint64_t *addr_list = (uint64_t *)malloc(DLA_MAX_BUFFERS_PER_TASK * sizeof(uint64_t));
            // Set the output of mac task parsing
            aipu_config.nvdla_task.num_addresses = 0;
            aipu_config.nvdla_task.address_list = addr_list;

            if (exe_order != 0) {
                aipu_config.mac_is_first_task = false;
                aipu_config.image_addr = NULL;
            }
            
            // Set aipu_config DLA's input/output address, image address is set in AipuSetInput;
            aipu_config.input_num = task_io_param[exe_order].input_offsets.size();
            for (uint32_t j = 0; j < task_io_param[exe_order].input_offsets.size(); j++) {
                size_t input_offset = task_io_param[exe_order].input_offsets[j];
                aipu_config.input_addr[j] = (uint64_t)(COMPILER_BASE_ADDR + input_offset);
            }
            aipu_config.output_addr = (uint64_t)(COMPILER_BASE_ADDR + output_offset);
            // set loadable address and size
            aipu_config.loadable.second = mac_tasks[mac_task_cnt].first;
            aipu_config.loadable.first = mac_tasks[mac_task_cnt].second;
            if(DEBUG_RUNTIME) {
                LOG_INFO << "\naipu_config {";
                printf("\taipu_mode          = %d,\n"
                        "\tbase_addr         = 0x%08x,\n"
                        "\taddr_offset       = 0x%08x,\n"
                        "\tmac_is_first_task = %d,\n"
                        "\tinput_num         = 0x%08x,\n"
                        "\tinput_addr0       = 0x%08x,\n"
                        "\tinput_addr1       = 0x%08x,\n"
                        "\tinput_addr2       = 0x%08x,\n"
                        "\toutput_addr       = 0x%08x,\n"
                        "\tloadable address  = 0x%08x, size = %d,\n"
                        "\timage_address     = 0x%08x,\n"
                        "\tnet_parse_done    = %d\n",
                        aipu_config.aipu_mode,
                        aipu_config.base_addr,
                        aipu_config.addr_offset,
                        aipu_config.mac_is_first_task,
                        aipu_config.input_num,
                        aipu_config.input_addr[0],
                        aipu_config.input_addr[1],
                        aipu_config.input_addr[2],
                        aipu_config.output_addr,
                        (uint64_t)(aipu_config.loadable.first), aipu_config.loadable.second,
                        (uint64_t)(aipu_config.image_addr),
                        aipu_config.net_parse_done
                        );
                printf("\n----------------------mac parse begin---------------------------\n");
            }
            rc = mac_task_parse(&aipu_config);
            if(DEBUG_RUNTIME) {
                printf("\n----------------------mac parse end-----------------------------\n");
            }
            /* if network parse has been done, follow parse should be pass, but reading image should not be pass */ 
            if (aipu_config.net_parse_done) {
                free(addr_list);
                return rc;
            }
            if (rc > 0) {
                LOG_ERROR << "/nMAC task parse failed.\n";
                return rc;
            } else {
                if(DEBUG_RUNTIME) {
                    LOG_INFO << "/nMAC task parse done.\n";
                }
                // parse done, write dla address list and task to firmware
                void *dla_addr_list_dst = (void *)(aipu_config.base_addr + aipu_config.addr_offset);
                size_t addr_list_size = aipu_config.nvdla_task.num_addresses * sizeof(uint64_t);
                aipu_config.addr_offset += ALIGN_32B(addr_list_size);

                // write address list to firmware mac memeory start from DLA_BASE_ADDR
                rc =  NvDlaWrite(dla_addr_list_dst, (void *)aipu_config.nvdla_task.address_list, addr_list_size);
                if (rc != addr_list_size) {
                    LOG_ERROR << "Dla address list write error.\n";
                } else {
                    if(DEBUG_RUNTIME) {
                        LOG_INFO << "Dla write address list done.\n";
                    }
                }

                aipu_task.dev_type = 0;
                aipu_task.num_addresses = (uint64_t)aipu_config.nvdla_task.num_addresses;
                aipu_task.address_list = (uint64_t)dla_addr_list_dst;
                aipu_task.task_pointer = (uint64_t)NULL;
                void *task_src = (void *)(&aipu_task);
                void *task_dst = (void *)aipu_task.next;
                // write task to firmware task memory start from AIPU_TASK_ADDR
                if (exe_order == (execute_order.size() - 1))
                    aipu_task.next = NULL;
                else 
                    aipu_task.next += ALIGN_32B(sizeof(aipu_task));
                rc =  NvDlaWrite(task_dst, task_src, sizeof(aipu_task));             
                if (rc != sizeof(aipu_task)) {
                    LOG_ERROR << "Dla task write error.\n";              
                } else {
                    if(DEBUG_RUNTIME) {
                        LOG_INFO << "\n========================FIRWMARE DEBUG INFO=====================\n" 
                                << exe_order <<" --- DLA TASK:"
                                << "\n\tdev_type:\t\t\t" << aipu_task.dev_type
                                << "\n\tnum_address:\t0x" << hex << aipu_task.num_addresses 
                                << "\n\taddress_list_addr:\t0x" << hex <<aipu_task.address_list
                                << "\n\tnext:\t\t\t\t0x" <<hex << aipu_task.next
                                << "\nDla address list is :";
                        printf("\n\t[");
                        for (uint32_t i = 0; i < aipu_task.num_addresses; i++) {
                            //attention: Can not use aipu_task.address_list, that address is on firmware
                            printf(" 0x%08x,", ((uint64_t *)(aipu_config.nvdla_task.address_list))[i]);
                        }
                        printf("]\n");                  
                    }
                }
            }
            mac_task_cnt++;
            free(addr_list);
        } else {// cpu task parse
            if(DEBUG_RUNTIME) {
                LOG_INFO << "\n=== Execute_order : " << exe_order << ": =================== RISCV TASK===============\n";
            }
            aipu_task.task_pointer = (uint64_t)(aipu_config.base_addr + aipu_config.addr_offset);

            if(DEBUG_RUNTIME) {
                LOG_INFO << "\n1.write op_param to DLA memory space of aipu from DLA_BASE_ADDR.\n";
            }
            uint32_t op_size_of_task = riscv_tasks[cpu_task_cnt].size();
            if (op_size_of_task == 0) {
                LOG_ERROR << "\nError: can not get op parameter in " << exe_order << " task.";
            }
            for (uint32_t i = 0; i < op_size_of_task; i++) {
                void *op_param_src = (void *)riscv_tasks[cpu_task_cnt][i];
                void *op_param_dst = (void *)(aipu_config.base_addr + aipu_config.addr_offset);
                size_t op_param_size = sizeof(struct cpu_param);
                rc = NvDlaWrite(op_param_dst, op_param_src, op_param_size);
                if (rc != op_param_size) {
                    LOG_ERROR << cpu_task_cnt << " : " << i << "op_param write error.\n";
                }
                aipu_config.addr_offset += ALIGN_32B(sizeof(struct CpuTaskPackage));
                uint64_t op_param_next = NULL;
                if (i < (op_size_of_task - 1))
                    op_param_next = (uint64_t)(aipu_config.base_addr + aipu_config.addr_offset);
                if(DEBUG_RUNTIME) {
                    LOG_INFO << std::endl << 
                        "op_param_dst is 0x" << std::setw(8) << std::setfill('0') << std::hex << (uint64_t)op_param_dst << std::endl <<
                        "op_param_size is 0x" << std::setw(8) << std::setfill('0') << std::hex << (uint64_t)op_param_size << std::endl <<
                        "next address is 0x" << std::setw(8) << std::setfill('0') << std::hex << ((uint64_t)op_param_dst + (uint64_t)op_param_size) << std::endl; 
                }
                rc = NvDlaWrite((void *)((uint64_t)op_param_dst + (uint64_t)op_param_size), (void *)(&op_param_next), sizeof(uint64_t));
                if(DEBUG_RUNTIME) {
                    LOG_INFO << std::endl <<
                        "\t" << op_size_of_task << " -- " << i << ": (USED TO CHECK FIRMWARE OP_TASK ADDRESS)" << std::endl <<
                        "\t\tcurrent cpu_param address is 0x" << std::setw(8) << std::setfill('0') << std::hex << (uint64_t)op_param_dst << "," << std::endl <<
                        "\t\tnext cpu_param address is 0x" << std::setw(8) << std::setfill('0') << std::hex << (uint64_t)op_param_next << "," << std::endl;
                }
            }
            if(DEBUG_RUNTIME) {
                LOG_INFO << "\n2.write address list to DLA memory space of aipu from DLA_BASE_ADDR.\n";
            }
            void *riscv_addr_list_dst = (void *)(aipu_config.base_addr + aipu_config.addr_offset);
            aipu_task.address_list = (uint64_t)riscv_addr_list_dst;
            uint32_t addr_list_size = riscv_addr_list[cpu_task_cnt].size();
            aipu_task.num_addresses = addr_list_size;
            if (aipu_task.num_addresses == 0) {
                LOG_ERROR << "\nError, address list is empty.";
            }
            uint64_t *riscv_addr_list_tmp = (uint64_t *)malloc(addr_list_size * sizeof(uint64_t));
            aipu_config.addr_offset += ALIGN_32B(addr_list_size * sizeof(uint64_t));
            for (uint32_t i = 0; i < addr_list_size; i++) {
                //2.1 transform offsets to address
                riscv_addr_list_tmp[i] = (uint64_t)(COMPILER_BASE_ADDR + riscv_addr_list[cpu_task_cnt][i].second);
                //2.2 check op has weight or not
                int32_t wt_index = riscv_addr_list[cpu_task_cnt][i].first;
                if (wt_index >= 0) {
                    void *wt_src = (void *)(riscv_wt_list[wt_index].second);
                    size_t wt_size = riscv_wt_list[wt_index].first;
                    // wrong address, weight address in the address list
                    // void *wt_dst = (void *)(aipu_config.base_addr + aipu_config.addr_offset); 
                    void *wt_dst = (void *)riscv_addr_list_tmp[i];
                    rc = NvDlaWrite(wt_dst, wt_src, wt_size);
                    if (rc != wt_size) {
                        LOG_ERROR << "\n\tWrite op weight error.\n";
                    } else {
                        if(DEBUG_RUNTIME) {
                            LOG_INFO << "\n\tWrite " << i<< "th op weight success, address is 0x" << hex << (uint64_t)wt_dst << ", size is " << wt_size << "\n"
                                << "\tThe first weight datas is: \n\t\t";
                            // TODO: is the following print correct?
                            for (int32_t j = 0; j < 8; j++) {
                                printf(" %d,", *(volatile char *)wt_src);
                            }
                            printf("\n");
                            //printf("%f\n", *(float *)wt_src);
                        }
                    }
                }
            }
            //2.3 write address list
            rc = NvDlaWrite(riscv_addr_list_dst, (void *)riscv_addr_list_tmp, (size_t)(addr_list_size * sizeof(uint64_t)));
            if (rc != (size_t)(addr_list_size * sizeof(uint64_t))) {
                LOG_ERROR << "\nWrite riscv address list error.\n";
            }

            if(DEBUG_RUNTIME) {
                LOG_INFO << "\n3.Write task to firmware task memory start from AIPU_TASK_ADDR.\n"; 
            }
            aipu_task.dev_type = 1;
            void *task_src = (void *)(&aipu_task);
            void *task_dst = (void *)aipu_task.next;
            if (exe_order == (execute_order.size() - 1))
                aipu_task.next = NULL;
            else 
                aipu_task.next += ALIGN_32B(sizeof(aipu_task));
            rc =  NvDlaWrite(task_dst, task_src, sizeof(aipu_task));             
            if (rc != sizeof(aipu_task)) {
                LOG_ERROR << "\nRiscv task write error.\n";              
            } else {
                if(DEBUG_RUNTIME) {
                    LOG_INFO << "\n=====================FIRMWARE DEBUG INFO ====================="
                            << "\n\t" << exe_order <<" --- Riscv TASK:\n"
                            << "\n\tdev_type:\t\t\t" << aipu_task.dev_type
                            << "\n\tnum_address:\t\t0x" << hex << aipu_task.num_addresses 
                            << "\n\taddress_list_addr:\t0x" << hex << aipu_task.address_list
                            << "\n\triscv_task_pointer:\t0x" << hex << aipu_task.task_pointer
                            << "\n\tnext:\t\t\t0x" <<hex << aipu_task.next
                            << "\n\tRiscv address list is :";
                    printf("\n\t[");
                    for (uint32_t i = 0; i < aipu_task.num_addresses; i++) {
                        printf(" 0x%08x,", (uint64_t)(riscv_addr_list_tmp[i]));
                    }
                    printf("]\n");                  
                }
            }

            cpu_task_cnt++;
            free(riscv_addr_list_tmp);
        }
    }
    aipu_config.net_parse_done = true;
    if (image_dat != NULL) {
        free(image_dat);
        image_dat = NULL;
   }
    return rc;
}

double get_elapsed_time(struct timespec *before, struct timespec *after)
{
	double deltat_s  = (after->tv_sec - before->tv_sec) * 1000000;
	double deltat_ns = (after->tv_nsec - before->tv_nsec) / 1000;
	return deltat_s + deltat_ns;
}

runtime::Module AipuRun(const tvm::runtime::Module& m) {
    struct timespec before, after, pre_infer;
    clock_gettime(CLOCK_MONOTONIC, &before);
    if ((!aipu_config.net_parse_done) | (aipu_config.mac_is_first_task)) {
        NetworkParse();
    }
    clock_gettime(CLOCK_MONOTONIC, &pre_infer);
    int fpga_fd = open(NVDLA_DEVICE_WRITE_NODE, O_RDWR);
    if (ioctl(fpga_fd, IOCTL_XDMA_RISCV_IRQ, 1) < 0) {
        LOG_ERROR << "Error: AipuRun failed!!!\n";
    }
    close(fpga_fd);
    clock_gettime(CLOCK_MONOTONIC, &after);
    if(DEBUG_RUNTIME) {
        LOG_INFO << "Whole interface time included network parse is " << get_elapsed_time(&before, &after) << endl;
        LOG_INFO << "Hardware interface time without network parse is " << get_elapsed_time(&pre_infer, &after) << endl;
    }
    return Module(m);
}

TVM_REGISTER_GLOBAL("tvm.nvdla_runtime_create").set_body([](TVMArgs args, TVMRetValue* rv) {
		ICHECK_GE(args.num_args, 4) << "The expected number of arguments for graph_executor.create is "
		"at least 4, but it has "
		<< args.num_args;
		PackedFunc lookup_linked_param_func;
		int dev_start_arg = 2;
		if (args[2].type_code() == kTVMPackedFuncHandle) {
		lookup_linked_param_func = args[2];
		dev_start_arg++;
		}
		const auto& devices = GetAllDevice(args, dev_start_arg);
		*rv = AipuRuntimeCreate(args[0], args[1], devices);
		});

TVM_REGISTER_GLOBAL("runtime.set_input").set_body([](TVMArgs args, TVMRetValue* rv) {
		ICHECK_GE(args.num_args, 2) << "The expected number of arguments for graph_executor.create is "
		"at least 2, but it has, later should at leaset one "
		<< args.num_args;

		*rv = AipuSetInput(args[0], args[1], args[2]); 
		});

TVM_REGISTER_GLOBAL("runtime.run").set_body([](TVMArgs args, TVMRetValue* rv) {
		ICHECK_GE(args.num_args, 1) << "The expected number of arguments for graph_executor.create is "
		"at least 1, but it has "
		<< args.num_args;
		const tvm::runtime::Module& m = args[0];
		*rv = AipuRun(args[0]);
		});


TVM_REGISTER_GLOBAL("runtime.get_output").set_body([](TVMArgs args, TVMRetValue* rv) {
		ICHECK_GE(args.num_args, 3) << "The expected number of arguments for graph_executor.create is "
		"at least 3, but it has "
		<< args.num_args;

		const tvm::runtime::Module& m = args[0];

		AipuGetOutput(args[0], args[1], args[2]);


		    });

		}  // namespace contrib
	}  // namespace runtime
}  // namespace tvm

