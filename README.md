# LCAI-TIHU SOFTWARE

## Introduction

TIHU is an AI computing platform based on RISC-V instruction set. It provides software stack and hardware stack, and is a complete set of solutions that can verify different AI frameworks and AI algorithms. TIHU is designed based on a variety of open source projects, including cva6 (https://github.com/openhwgroup/cva6), ara (https://github.com/pulp-platform/ara), nvlda (https://github.com/nvdla), xdma (https://github.com/Xilinx/dma_ip_drivers/tree/master/XDMA/linux-kernel) and TVM (https://github.com/apache/tvm), to explore the current AI open source ecology and accelerate the implementation of AI algorithms. In this project, we can explore RISC-V instruction set, deep-learning accelerator, AI compiler &  runtime, AI algorithms and AI frameworks.

Currently, TIHU only supports tensorflow. Pytorch will be supported very soon. Several CNN models have been tested.


| num | category | CNN model | TIHU(int8) accuracy | llvm(float) accuracy|
|---|----------------|-------|--------|------|
| 1 | classification | LeNet | 0.9884 | 0.992|
| 2 | classification | ResNet50 | 0.748 | 0.752|
| 3 | classification | MobileNetv2 | 0.71 | 0.72|
| 4 | detection | YOLOv3 | 0.548 | 0.562|
| 5 | NLP | bert-base | - | - |

## TIHU software structure and workflow
TIHU software include compiler, runtime, xdma driver and firmware.

<div align=center>
<img src="./doc/compiler_structure.png" width="600" height="500" alt="TIHU"/><br/>
</div>

<div align=center>
<img src="./doc/TIHU_workflow.png" width="600" height="400" alt="TIHU"/><br/>
</div>

## Code structure

├── compiler_runtime: TIHU software, include compiler and runtime  
├── doc: TIHU user's guide  
├── firmware: TIHU SoC firmware  
├── xdma: driver and firmware download tools  
├── docker: docker file  
├── README.md  
└── LICENSE  

## Build procedure on ubuntu  

1. Build compiler and runtime without dockfile   
Install docker and then:  
```
sudo docker run -it --network=host -v /your/project/path/on/host:/your/project/path/on/docker -v /dev:/dev --privileged=true --name your_docker_name docker_image_name:version /bin/bash # run docker and load ubuntu image
apt-get update
apt-get install -y python3 python3-dev python3-setuptools python3-pip python3-venv python-scipy gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev llvm-dev libjpeg-turbo8-dev git python3-pip autoconf # install prerequisites
pip3 install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple # update pip
pip3 install numpy decorator attrs pytest scipy  opencv-python-headless tqdm pycocotools -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install tensorflow==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple  # install tensorflow
git clone https://github.com/LCAI-TIHU/SW.git
cd SW/compiler_runtime && source env.sh
source clean.sh && source build.sh

```

2. build compiler and runtime with dockefile   



# Run reference design example

There are some samples in xxx/SW/compiler_runtime/AIPU_demo, before run you should:   
* Make sure the FPGA bitstream has been load, you can refer to HW project;  
* Make sure firmware has been compiled and downloaded, you can refer to ./firmware README;  
* Run docker and build compiler and runtime, make sure env.sh has been sourced;  
```
cd xxx/SW/compiler_runtime/AIPU_demo
python3 from_tensorflow_quantize_lenet.py 2>&1 | tee lenet.log

```   

# Road Map   

# License

TIHU is licensed under the Apache-2.0 license. You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

