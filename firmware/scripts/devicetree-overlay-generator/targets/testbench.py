#!/usr/bin/env python3
# Copyright (c) 2019 SiFive Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This is a python script for generating RTL testbench Devicetree overlays from the Devicetree
for the RTL DUT.
"""

import sys

import pydevicetree

from targets.generic import PORTS, CAP_SIZE_FOR_VCS
from targets.generic import number_to_cells, set_boot_hart, set_stdout, set_entry, get_rams, set_rams, set_ecc_scrub, get_reference

def get_testram(port, label):
    ranges = port.get_ranges()

    # The memory port may describe itself with the `reg` property, in which case there's no
    # need to attach a testram.
    if ranges is not None:
        address = ranges[0][0]
        size = min(ranges[0][2], CAP_SIZE_FOR_VCS)

        num_address_cells = port.get_field("#address-cells")
        num_size_cells = port.get_field("#size-cells")

        address_cells = number_to_cells(address, num_address_cells)
        size_cells = number_to_cells(size, num_size_cells)

        testram = pydevicetree.Node.from_dts("""
            %s: testram@%x {
                compatible = "sifive,testram0";
                reg = <%s %s>;
                reg-names = "mem";
            };
        """ % (label, address, address_cells, size_cells))

        return testram


def attach_testrams(tree, overlay):
    """Generate testrams attached to ports in the overlay

    Attached rams are also created in the in-memory tree so that they can be queried as if the
    overlay has been applied.
    """
    for count, port in enumerate(tree.match("sifive,.*port")):
        label = "testram%d" % count

        testram = get_testram(port, label)

        if testram is not None:
            port.add_child(testram)
            overlay.children.append(pydevicetree.Node.from_dts("&%s { %s };" % (port.label, testram.to_dts())))

def get_boot_rom(tree):
    """Given a tree with attached testrams, return the testram which contains the default reset
    vector"""
    port_compatibles = list(map(lambda n: n.get_field("compatible"), tree.match("sifive,.*-port")))
    for port in PORTS:
        if port in port_compatibles:
            matching_ports = tree.match(port)
            if matching_ports:
                testrams = matching_ports[0].match("sifive,testram0")
                if testrams:
                    return testrams[0]

    # Fall back to /memory node if no port exists
    memory = tree.get_by_path("/memory")
    if memory is not None:
        return memory

    sys.stderr.write("%s: Unable to determine test bench reset vector\n" % sys.argv[0])
    sys.exit(1)

def generate_overlay(tree, overlay):
    """Generate the overlay"""
    attach_testrams(tree, overlay)

    bootrom = get_boot_rom(tree)
    if bootrom is not None:
        set_entry(overlay, bootrom, 0, 0)

    set_boot_hart(tree, overlay)

    ram, itim = get_rams(tree)

    # If no RAM exists, put everything in the testram
    if ram is None:
        ram = bootrom

    # Do scrub If ROM and RAM is not the same node
    if get_reference(bootrom) != get_reference(ram):
        set_ecc_scrub(tree, overlay)

    set_rams(overlay, ram, itim)
