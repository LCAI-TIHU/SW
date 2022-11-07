# LCAI-TIHU SOFTWARE

## Introduction
TIHU is designed based on a variety of open source projects, including cva6 (https://github.com/openhwgroup/cva6), ara (https://github.com/pulp-platform/ara), nvlda (https://github.com/nvdla), xdma (https://github.com/Xilinx/dma_ip_drivers/tree/master/XDMA/linux-kernel) and TVM (https://github.com/apache/tvm), to explore the current AI open source ecology and accelerate the implementation of AI algorithms. In this project, we can explore RISC-V instruction set, deep-learning accelerator, AI compiler and runtime, AI algorithms and AI frameworks.

### TIHU hardware
TIHU hardware is comprised of RISC-V cpu, nvdla, NoC bus, PCIe module, DDR, SRAM, bootROM, DMA and peripherals. Parameters:  
* Support RISC-V instruction set: RV64gcv0p10;  
* Nvdla config:  
* Memory: DDR -- 2GB, SRAM -- 4MB, ROM -- 128KB;
* SoC frequency: 20MHz;  
* SoC systerm: baremetal;
* Debug: uart;

<div align=center>
<img src="./doc/AIPU_structure.png" width="600" height="300" alt="TIHU"/><br/>
</div>
                                                                                                                                                                                                
### TIHU software
TIHU software is designed based on TVM.

<div align=center>
<img src="./doc/compiler_structure.png" width="600" height="300" alt="TIHU"/><br/>
</div>

## Code structure

├── compiler_runtime: TIHU software, include compiler and runtime;  
├── doc: TIHU user's guide;  
├── firmware: TIHU SoC firmware;  
├── xdma: driver and firmware download tools;  
├── README.md  
└── LICENSE  

## Build procedure
1. Build without dockfile  

2. build with dockefile  

## Build FPGA

# Run reference design example

# Road Map   

# License

TIHU is licensed under the Apache-2.0 license. You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

