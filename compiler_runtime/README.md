<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# TIHU Open Deep Learning Compiler Stack  
_Based on tvm_  

[Documentation](https://tvm.apache.org/docs) |
[Contributors](CONTRIBUTORS.md) |
[Community](https://tvm.apache.org/community) |
[Release Notes](NEWS.md)

[![Build Status](https://ci.tlcpack.ai/buildStatus/icon?job=tvm/main)](https://ci.tlcpack.ai/job/tvm/job/main/)
[![WinMacBuild](https://github.com/apache/tvm/workflows/WinMacBuild/badge.svg)](https://github.com/apache/tvm/actions?query=workflow%3AWinMacBuild)

Apache TVM is a compiler stack for deep learning systems. It is designed to close the gap between the
productivity-focused deep learning frameworks, and the performance- and efficiency-focused hardware backends.
TVM works with deep learning frameworks to provide end to end compilation to different backends.

TIHU compiler stack integrates the nvdla compiler and runtime into tvm, and necessary modifications are made to
improve the efficiency of hardware computing.

# What we have done

<div align=center>
<img src="../doc/compiler_structure.png" width="600" height="400" alt="TIHU"/><br/>
</div>

Based on TVM [BYOC](https://tvm.apache.org/docs/dev/how_to/relay_bring_your_own_codegen.html) we integrate our backend into tvm, as the above shows,
we developed specific memory plan and operator format for our backend, for MAC unit, we convert tvm relay IR to NVDLA compiler's network, for riscv
unit, we transport tvm relay operators' information to manually implemented operators library.

# What will be done

Right now, our backend can't cooperate with other backends, we will make some changes in runtime to add this feature. Next we will explore other codegen strategy
for riscv unit, since it is a general computing device, we may use tvm's llvm codegen and auto tune to get better performance.

License
-------
© Contributors Licensed under an [Apache-2.0](LICENSE) license.
