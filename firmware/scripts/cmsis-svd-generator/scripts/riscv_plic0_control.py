#!/usr/bin/env python3
# Copyright (c) 2019 SiFive Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This program generates CMSIS SVD xml for riscv plic0
"""

def generate_registers_riscv_plic0(dts, peripheral):
    """Generate xml string for registers for riscv_plic0 peripheral"""
    ndev = peripheral.get_fields("riscv,ndev")[0]
    cpus = dts.get_by_path("/cpus")
    txt = """\
              <registers>
"""

    addr_num = 0x4
    for intr_num in range(ndev):
        intr = str(intr_num + 1)
        addr = "0x{:X}".format(addr_num)
        txt += generate_registers_riscv_plic0_priority(intr, addr)
        addr_num += 4

    addr_num = 0x1000
    intr_num = 0
    while intr_num <= ndev:
        inta = str(intr_num >> 5)
        intl = str(intr_num)
        inth = str(intr_num + 32 - 1)
        if intr_num + 32 - 1 > ndev:
            inth = str(ndev)
        addr = "0x{:X}".format(addr_num + (intr_num >> 3))
        txt += generate_registers_riscv_plic0_pending(inta, inth, intl, addr)
        intr_num += 32

    hart_num = 0
    addr_num = 0x2000
    for cpu in cpus.child_nodes():
        if cpu.get_field("device_type") is not None:
            if cpu.get_fields("device_type")[0] == "cpu":
                hart = str(hart_num)
                intr_num = 0
                while intr_num <= ndev:
                    inta = str(intr_num >> 5)
                    intl = str(intr_num)
                    inth = str(intr_num + 32 - 1)
                    if intr_num + 32 - 1 > ndev:
                        inth = str(ndev)
                    addr = "0x{:X}".format(addr_num + (intr_num >> 3))
                    txt += generate_registers_riscv_plic0_enable(inta, inth, intl, hart, addr)
                    intr_num += 32
                hart_num += 1
                addr_num += 0x80

    hart_num = 0
    addr_num = 0x200000
    for cpu in cpus.child_nodes():
        if cpu.get_field("device_type") is not None:
            if cpu.get_fields("device_type")[0] == "cpu":
                hart = str(hart_num)
                addr = "0x{:X}".format(addr_num)
                txt += generate_registers_riscv_plic0_threshold(hart, addr)
                addr = "0x{:X}".format(addr_num + 0x4)
                txt += generate_registers_riscv_plic0_claimplete(hart, addr)
                hart_num += 1
                addr_num += 0x1000

    txt += """\
              </registers>
"""
    return txt

def generate_registers_riscv_plic0_priority(intr, addr):
    """Generate xml string for riscv_plic0 priority register for specific interrupt id"""
    return """\
                <register>
                  <name>priority_""" + intr + """</name>
                  <description>PRIORITY Register for interrupt id """ + intr + """</description>
                  <addressOffset>""" + addr + """</addressOffset>
                </register>
"""

def generate_registers_riscv_plic0_pending(inta, inth, intl, addr):
    """Generate xml string for riscv_plic0 pending register for specific interrupt ids"""
    temp = inth + " to " + intl
    return """\
                <register>
                  <name>pending_""" + inta + """</name>
                  <description>PENDING Register for interrupt ids """ + temp + """</description>
                  <addressOffset>""" + addr + """</addressOffset>
                </register>
"""

def generate_registers_riscv_plic0_enable(inta, inth, intl, hart, addr):
    """Generate xml string for riscv_plic0 enable register for specific interrupt ids per hart"""
    temp = inth + " to " + intl + " for hart " + hart
    return """\
                <register>
                  <name>enable_""" + inta + """_""" + hart + """</name>
                  <description>ENABLE Register for interrupt ids """ + temp + """</description>
                  <addressOffset>""" + addr + """</addressOffset>
                </register>
"""

def generate_registers_riscv_plic0_threshold(hart, addr):
    """Generate xml string for riscv_plic0 threshold register for hart"""
    return """\
                <register>
                  <name>threshold_""" + hart + """</name>
                  <description>PRIORITY THRESHOLD Register for hart """ + hart + """</description>
                  <addressOffset>""" + addr + """</addressOffset>
                </register>
"""

def generate_registers_riscv_plic0_claimplete(hart, addr):
    """Generate xml string for riscv_plic0 claim and complete register for hart"""
    return """\
                <register>
                  <name>claimplete_""" + hart + """</name>
                  <description>CLAIM and COMPLETE Register for hart """ + hart + """</description>
                  <addressOffset>""" + addr + """</addressOffset>
                </register>
"""
