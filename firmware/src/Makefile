# Copyright 2019 Inspur Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#    
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

PROGRAM ?= main

override ASFLAGS +=  #-menable-experimental-extensions -mllvm -riscv-v-vector-bits-min=512 -ffast-math #-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize

override CFLAGS +=  # -menable-experimental-extensions -mllvm -riscv-v-vector-bits-min=512 -ffast-math #-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize

override CXXFLAGS += # -menable-experimental-extensions  -mllvm -riscv-v-vector-bits-min=512 -ffast-math #-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize


$(PROGRAM): $(wildcard *.c ./dla/*.c ) $(wildcard *.h ./dla/*.h ./include/*.h) $(wildcard *.S)
	$(CC) $(CFLAGS) $(LDFLAGS) $(filter %.c %.S,$^) $(LOADLIBES) $(LDLIBS) -o $@

clean:
	rm -f $(PROGRAM) $(PROGRAM).hex
