# Copyright 2019 SiFive, Inc #
# SPDX-License-Identifier: Apache-2.0 #

PROGRAM ?= main

override ASFLAGS +=  #-menable-experimental-extensions -mllvm -riscv-v-vector-bits-min=512 -ffast-math #-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize

override CFLAGS +=  # -menable-experimental-extensions -mllvm -riscv-v-vector-bits-min=512 -ffast-math #-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize

override CXXFLAGS += # -menable-experimental-extensions  -mllvm -riscv-v-vector-bits-min=512 -ffast-math #-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize


$(PROGRAM): $(wildcard *.c ./dla/*.c ) $(wildcard *.h ./dla/*.h ./include/*.h) $(wildcard *.S)
	$(CC) $(CFLAGS) $(LDFLAGS) $(filter %.c %.S,$^) $(LOADLIBES) $(LDLIBS) -o $@

clean:
	rm -f $(PROGRAM) $(PROGRAM).hex
