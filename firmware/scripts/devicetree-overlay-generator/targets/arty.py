#!/usr/bin/env python3
# Copyright (c) 2019 SiFive Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This is a python script for generating RTL testbench Devicetree overlays from the Devicetree
for the RTL DUT.
"""

from targets.generic import set_boot_hart, set_stdout, set_entry, get_spi_flash, get_spi_region, get_rams, set_rams, set_ram, set_itim

SRAM_SPLIT_MIN_SIZE = 0x10000

def generate_overlay(tree, overlay):
    """Generate the overlay"""
    bootrom = get_spi_flash(tree)
    if bootrom is not None:
        region = get_spi_region(bootrom)
        set_entry(overlay, bootrom, region, 0x400000)

    set_boot_hart(tree, overlay)
    set_stdout(tree, overlay, 115200)

    ram, itim = get_rams(tree)

    # If the ram and itim are the same sram node
    ram_compat = ram.get_field("compatible")
    if ram_compat is None:
        ram_compat = ""
    if ram == itim and "sram" in ram_compat:
        # get the size of the first tuple
        size = ram.get_reg()[0][1]
        # if the size is above the threshold
        if size >= SRAM_SPLIT_MIN_SIZE:
            # split the memory into separate ram and itim
            set_ram(overlay, ram, 0, 0)
            set_itim(overlay, ram, 0, size / 2)
            return

    set_rams(overlay, ram, itim)
