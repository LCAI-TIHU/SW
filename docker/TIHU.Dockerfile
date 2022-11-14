#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

FROM ubuntu:18.04 AS download-tarballs

LABEL maintainer="Inspur LCAI-TIHU"

ARG SIFIVE_TOOLS_URL=https://static.dev.sifive.com/dev-tools/freedom-tools/v2020.12
ARG RISCV_TOOLS_TARBALL=riscv64-unknown-elf-toolchain-10.2.0-2020.12.8-x86_64-linux-ubuntu14.tar.gz

RUN apt-get update && \
apt-get upgrade -y && \
apt-get install -y \
bzip2 \
rsync \
kmod  \
wget

# Install RISC-V Toolchain
RUN wget  ${SIFIVE_TOOLS_URL}/${RISCV_TOOLS_TARBALL} && \
tar xzf ${RISCV_TOOLS_TARBALL} && \
mkdir -p /tools && \
rsync -a ${RISCV_TOOLS_TARBALL%.tar.gz}/* /tools/


FROM ubuntu:18.04

# Install python3
RUN apt-get update && \
apt-get install -y python3 \
python3-dev \
python3-setuptools \
python3-pip \
python3-venv \
python-scipy 

# Install requried libraries
RUN apt-get update && \
apt-get install -y gcc \
libtinfo-dev \
zlib1g-dev \
build-essential \
cmake \
libedit-dev \
libxml2-dev \
llvm-10-dev \
libjpeg-turbo8-dev \
git \
autoconf \
bzip2 \
rsync \
wget \
device-tree-compiler \
git \
jq \
libfdt-dev \
pciutils \
linux-headers-$(uname -r)

# Install PyPI packages
RUN pip3 install -U pip   && \
pip3 install numpy decorator attrs pytest scipy  opencv-python-headless tqdm pycocotools  && \
pip3 install keras==2.6.0 tensorflow==2.6.1   

COPY --from=download-tarballs /tools /

ENV PATH=/tools/bin:$PATH
