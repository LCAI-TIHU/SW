#!/usr/bin/env python3
# Copyright (c) 2019 SiFive Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This program generates CMSIS SVD xml for riscv clint0
"""

def generate_registers_riscv_clint0(dts):
    """Generate xml string for registers for riscv_clint0 peripheral"""
    cpus = dts.get_by_path("/cpus")
    txt = """\
              <registers>
"""

    hart_num = 0
    addr_num = 0x0000
    for cpu in cpus.child_nodes():
        if cpu.get_field("device_type") is not None:
            if cpu.get_fields("device_type")[0] == "cpu":
                hart = str(hart_num)
                addr = "0x{:X}".format(addr_num)
                txt += generate_registers_riscv_clint0_msip(hart, addr)
                hart_num += 1
                addr_num += 0x4

    hart_num = 0
    addr_num = 0x4000
    for cpu in cpus.child_nodes():
        if cpu.get_field("device_type") is not None:
            if cpu.get_fields("device_type")[0] == "cpu":
                hart = str(hart_num)
                addr = "0x{:X}".format(addr_num)
                txt += generate_registers_riscv_clint0_mtimecmp(hart, addr)
                hart_num += 1
                addr_num += 0x8

    addr_num = 0xBFF8
    addr = "0x{:X}".format(addr_num)
    txt += generate_registers_riscv_clint0_mtime(addr)

    txt += """\
              </registers>
"""
    return txt

def generate_registers_riscv_clint0_msip(hart, addr):
    """Generate xml string for riscv_clint0 msip register for specific hart"""
    return """\
                <register>
                  <name>msip_""" + hart + """</name>
                  <description>MSIP Register for hart """ + hart + """</description>
                  <addressOffset>""" + addr + """</addressOffset>
                </register>
"""

def generate_registers_riscv_clint0_mtimecmp(hart, addr):
    """Generate xml string for riscv_clint0 mtimecmp register for specific hart"""
    return """\
                <register>
                  <name>mtimecmp_""" + hart + """</name>
                  <description>MTIMECMP Register for hart """ + hart + """</description>
                  <addressOffset>""" + addr + """</addressOffset>
                  <size>64</size>
                </register>
"""

def generate_registers_riscv_clint0_mtime(addr):
    """Generate xml string for riscv_clint0 mtime register"""
    return """\
                <register>
                  <name>mtime</name>
                  <description>MTIME Register</description>
                  <addressOffset>""" + addr + """</addressOffset>
                  <size>64</size>
                </register>
"""
