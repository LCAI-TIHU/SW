#!/usr/bin/env python3
# Copyright (c) 2019 SiFive Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This program generates CMSIS SVD xml for sifive clic0
"""

from scripts.riscv_clint0_control import generate_registers_riscv_clint0_msip
from scripts.riscv_clint0_control import generate_registers_riscv_clint0_mtimecmp
from scripts.riscv_clint0_control import generate_registers_riscv_clint0_mtime

def generate_registers_sifive_clic0(dts, peripheral):
    """Generate xml string for registers for sifive_clic0 peripheral"""
    numints = peripheral.get_fields("sifive,numints")[0]
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

    addr_num = 0x800000
    for intr_num in range(numints):
        intr = str(intr_num)
        addr = "0x{:X}".format(addr_num + intr_num)
        txt += generate_registers_sifive_clic0_clicintip(intr, addr)

    addr_num = 0x800400
    for intr_num in range(numints):
        intr = str(intr_num)
        addr = "0x{:X}".format(addr_num + intr_num)
        txt += generate_registers_sifive_clic0_clicintie(intr, addr)

    addr_num = 0x800800
    for intr_num in range(numints):
        intr = str(intr_num)
        addr = "0x{:X}".format(addr_num + intr_num)
        txt += generate_registers_sifive_clic0_clicintctl(intr, addr)

    addr_num = 0x800C00
    addr = "0x{:X}".format(addr_num)
    txt += generate_registers_sifive_clic0_cliccfg(addr)

    txt += """\
              </registers>
"""
    return txt

def generate_registers_sifive_clic0_clicintip(intr, addr):
    """Generate xml string for riscv_clic0 intip register for specific interrupt id"""
    return """\
                <register>
                  <name>clicintip_""" + intr + """</name>
                  <description>CLICINTIP Register for interrupt id """ + intr + """</description>
                  <addressOffset>""" + addr + """</addressOffset>
                  <size>8</size>
                </register>
"""

def generate_registers_sifive_clic0_clicintie(intr, addr):
    """Generate xml string for riscv_clic0 intie register for specific interrupt id"""
    return """\
                <register>
                  <name>clicintie_""" + intr + """</name>
                  <description>CLICINTIE Register for interrupt id """ + intr + """</description>
                  <addressOffset>""" + addr + """</addressOffset>
                  <size>8</size>
                </register>
"""

def generate_registers_sifive_clic0_clicintctl(intr, addr):
    """Generate xml string for riscv_clic0 intctl register for specific interrupt id"""
    return """\
                <register>
                  <name>clicintctl_""" + intr + """</name>
                  <description>CLICINTCTL Register for interrupt id """ + intr + """</description>
                  <addressOffset>""" + addr + """</addressOffset>
                  <size>8</size>
                </register>
"""

def generate_registers_sifive_clic0_cliccfg(addr):
    """Generate xml string for riscv_clic0 cfg register"""
    return """\
                <register>
                  <name>cliccfg</name>
                  <description>CLICCFG Register</description>
                  <addressOffset>""" + addr + """</addressOffset>
                  <size>8</size>
                  <fields>
                    <field>
                      <name>nvbits</name>
                      <description>When set, selective hardware vectoring is enabled.</description>
                      <bitRange>[0:0]</bitRange>
                      <access>read-write</access>
                    </field>
                    <field>
                      <name>nlbits</name>
                      <description>Determines the number of Level bits available in clicintctl.</description>
                      <bitRange>[4:1]</bitRange>
                      <access>read-write</access>
                    </field>
                    <field>
                      <name>nmbits</name>
                      <description>Determines the number of Mode bits available in clicintctl.</description>
                      <bitRange>[6:5]</bitRange>
                      <access>read-only</access>
                    </field>
                  </fields>
                </register>
"""
