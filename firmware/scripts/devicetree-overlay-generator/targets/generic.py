#!/usr/bin/env python3
# Copyright (c) 2019 SiFive Inc.
# SPDX-License-Identifier: Apache-2.0

"""
These constants and functions are useful for all target types
"""

import itertools

import pydevicetree

PORT_PROTOCOLS = [
    "ahb",
    "apb",
    "axi4",
    "tl",
]

PORT_TYPES = [
    "periph",
    "sys",
    "mem",
]

# The resulting order of the contents of PORTS is sensitive and should not be modified without
# first consulting the RTL testbench developers or RTL simulation will break.
PORTS = ["sifive,%s-%s-port" % (protocol, port_type) \
             for port_type, protocol in itertools.product(PORT_TYPES, PORT_PROTOCOLS)]

CAP_SIZE_FOR_VCS = 0x1fffffff

STDOUT_DEVICES = [
    "sifive,uart0",
    "sifive,trace",
    "ucb,htif0",
]

SRAM_DEVICES = ["sifive,sram0"]

def get_reference(node):
    """Get a Devicetree reference to a node

    For example:
     - With label: "serial0: uart@10013000 {};" -> "&serial0"
     - Without label: "uart@10013000 {};" -> "&{/path/to/uart@10013000}"
    """
    if node.label != "":
        return "&%s" % node.label
    return "&{%s}" % node.get_path()

def set_entry(overlay, node, tuple_index, offset):
    """Set entry vector in overlay"""
    chosen = overlay.get_by_path("/chosen")
    entry_prop = "metal,entry = <%s %d %d>;" % (get_reference(node), tuple_index, offset)
    chosen.properties.append(pydevicetree.Property.from_dts(entry_prop))

def get_itim(tree, hart):
    """Get the ITIM associated with the hart"""
    itim_ref = hart.get_field("sifive,itim")
    if itim_ref:
        itim = tree.get_by_reference(itim_ref)
        return itim
    return None

def get_dtim(tree, hart):
    """Get the DTIM associated with the hart"""
    dtim_ref = hart.get_field("sifive,dtim")
    if dtim_ref:
        dtim = tree.get_by_reference(dtim_ref)
        return dtim
    return None

def get_rams(tree):
    """Get the (ram, itim) tuple for the target"""
    boot_hart = get_boot_hart(tree)

    # Get any available srams in case we don't have an itim, dtim or memory
    srams = []
    for compatible in SRAM_DEVICES:
        srams += tree.match(compatible)

    memory = tree.get_by_path("/memory")
    dtim = get_dtim(tree, boot_hart)

    if memory:
        ram = memory
    elif dtim:
        ram = dtim
    elif len(srams) > 0:
        ram = srams[0]
    else:
        ram = None

    itim = get_itim(tree, boot_hart)
    if not itim:
        if len(srams) > 1:
            itim = srams[1]
        else:
            itim = ram

    return (ram, itim)

def set_rams(overlay, ram, itim):
    """Set the metal,ram and metal,itim properties"""
    if itim:
        set_itim(overlay, itim, 0, 0)
    if ram:
        set_ram(overlay, ram, 0, 0)

def set_itim(overlay, node, tuple_index, offset):
    """Set itim in overlay"""
    chosen = overlay.get_by_path("/chosen")
    itim_prop = "metal,itim = <%s %d %d>;" % (get_reference(node), tuple_index, offset)
    chosen.properties.append(pydevicetree.Property.from_dts(itim_prop))

def set_ram(overlay, node, tuple_index, offset):
    """Set ram in overlay"""
    chosen = overlay.get_by_path("/chosen")
    ram_prop = "metal,ram = <%s %d %d>;" % (get_reference(node), tuple_index, offset)
    chosen.properties.append(pydevicetree.Property.from_dts(ram_prop))

def get_boot_hart(tree):
    """Given a tree, return the node which should be used as the boot hart"""
    riscv_harts = tree.match("^riscv$")
    for hart in riscv_harts:
        if hart.get_reg()[0][0] == 1:
            return hart
    return riscv_harts[0]

def set_boot_hart(tree, overlay):
    """Set boot hart in overlay"""
    chosen = overlay.get_by_path("/chosen")
    boot_hart = get_boot_hart(tree)
    chosen.properties.append(pydevicetree.Property.from_dts("metal,boothart = <%s>;" % \
                                                            get_reference(boot_hart)))

def set_ecc_scrub(tree, overlay):
    """Set ecc scrub bit in overlay"""
    chosen = overlay.get_by_path("/chosen")
    if tree.match("sifive,buserror0"):
        ecc_scrub = 1
    else:
        ecc_scrub = 0
    chosen.properties.append(pydevicetree.Property.from_dts("metal,eccscrub = <%d>;" % \
                                                            ecc_scrub))

def get_spi_flash(tree):
    """Get the SPI Flash node"""
    spi_nors = tree.match("jedec,spi-nor")
    if len(spi_nors) == 0:
        return None
    return spi_nors[0].parent

def get_spi_region(spi_node):
    """Get which reg tuple should be used for memory"""
    tuples = spi_node.get_reg().tuples
    for i, tup in enumerate(tuples):
        if tup[2] == "mem":
            return i
    return 0

def number_to_cells(num, num_cells):
    """Convert an integer into 32-bit cells"""
    cells = []
    for i in range(num_cells):
        cells.insert(0, (0xFFFFFFFF & (num >> (32 * i))))
    return " ".join(["0x%x" % x for x in cells])

def get_stdout(tree):
    """Given a tree, return teh node which should be used as stdout"""
    for compat in STDOUT_DEVICES:
        nodes = tree.match(compat)
        if len(nodes) > 0:
            return nodes[0]
    return None

def set_stdout(tree, overlay, baudrate):
    """Set the stdout path and baud rate"""
    chosen = overlay.get_by_path("/chosen")
    stdout = get_stdout(tree)
    if stdout is not None:
        chosen.properties.append(
            pydevicetree.Property.from_dts("stdout-path = \"%s:%d\";" % \
                    (stdout.get_path(), baudrate)))
