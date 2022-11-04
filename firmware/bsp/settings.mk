# Copyright (C) 2020 SiFive Inc
# SPDX-License-Identifier: Apache-2.0

RISCV_ARCH = rv64imafdcv
#RISCV_ARCH = rv64imafdcv_zba_zbb_zfh_xsfvqmaccqoq_xsfvfhbfmin
#RISCV_ABI = lp64
RISCV_ABI = lp64d
RISCV_CMODEL = medany
RISCV_SERIES = sifive-3-series

TARGET_TAGS = board openocd
TARGET_DHRY_ITERS = 20000000
TARGET_CORE_ITERS = 5000
TARGET_FREERTOS_WAIT_MS = 1000
TARGET_INTR_WAIT_CYCLE  = 0
