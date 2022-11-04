#!/usr/bin/env python3
# Copyright (c) 2020 SiFive Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate Freedom E SDK settings.mk from devicetree source files"""

import argparse
import sys

import pydevicetree

SUPPORTED_TYPES = ["rtl", "arty", "qemu", "hifive", "spike", "vc707", "vcu118"]


def parse_arguments(argv):
    """Parse the arguments into a dictionary with argparse"""
    arg_parser = argparse.ArgumentParser(
        description="Generate Freedom E SDK settings.mk from Devicetrees")

    arg_parser.add_argument("-t", "--type", required=True,
                            help="The type of the target to generate settings.mk for. \
                                Supported types include: %s" % ", ".join(SUPPORTED_TYPES))
    arg_parser.add_argument("-d", "--dts", required=True,
                            help="The path to the Devicetree for the target")
    arg_parser.add_argument("-o", "--output",
                            type=argparse.FileType('w'),
                            help="The path of the settings.mk file to output")

    parsed_args = arg_parser.parse_args(argv)

    if not any([t in parsed_args.type for t in SUPPORTED_TYPES]):
        print("Type '%s' is not supported, please choose one of: %s" % (parsed_args.type,
                                                                        ', '.join(SUPPORTED_TYPES)))
        sys.exit(1)

    return parsed_args


def get_boot_hart(tree):
    """Get the boot hart if one is specified, otherwise just gets the first hart"""
    metal_boothart = tree.chosen("metal,boothart")
    if metal_boothart:
        return tree.get_by_reference(metal_boothart[0])
    return tree.get_by_path("/cpus").children[0]


def get_all_arch(tree):
    """Get a list of architecture strings from all harts"""
    return [cpu.get_field("riscv,isa") for cpu in tree.get_by_path("/cpus").children]

def get_greatest_common_arch(archs):
    """Get the RISC-V ISA string which contains as many extensions as are supported
       by all harts in the design"""
    if len(archs) == 1:
        return archs[0]

    # Get all ISA extensions implemented by any hart
    extensions = ''.join(set(''.join([arch[4:] for arch in archs])))

    # Get a list of any extensions which aren't supported by all harts
    disallowed_extensions = ""
    for extension in extensions:
        if not all([extension in arch[4:] for arch in archs]):
            disallowed_extensions += extension

    # Get the longest arch from the list
    arch = max(archs, key=len)

    # Filter out any disallowed extensions
    for extension in disallowed_extensions:
        base = arch[:4]
        extensions = arch[4:].replace(extension, "")
        arch = base + extensions

    return arch


def arch2arch(arch):
    """Remap certain arch strings which are known to not be supportable"""
    # pylint: disable=too-many-return-statements
    if arch == "rv32ea":
        return "rv32e"
    if arch in ["rv32ema", "rv32emc"]:
        return "rv32em"

    if arch == "rv32ia":
        return "rv32i"
    if arch == "rv32ima":
        return "rv32im"

    if arch == "rv64ia":
        return "rv64i"
    if arch == "rv64ima":
        return "rv64im"

    return arch


def arch2abi(arch):
    """Map arch to abi"""
    # pylint: disable=too-many-return-statements
    if "rv32e" in arch:
        if "d" in arch:
            return "ilp32ed"
        if "f" in arch:
            return "ilp32ef"
        return "ilp32e"
    if "rv32i" in arch:
        if "d" in arch:
            return "ilp32d"
        if "f" in arch:
            return "ilp32f"
        return "ilp32"
    if "rv64i" in arch:
        if "d" in arch:
            return "lp64d"
        if "f" in arch:
            return "lp64f"
        return "lp64"

    raise Exception("Unknown arch %s" % arch)


def type2tag(target_type):
    """Given the target type, return the list of TARGET_TAGS to parameterize Freedom E SDK"""
    if "arty" in target_type:
        tags = "fpga openocd arty"
    if "vc707" in target_type:
        tags = "fpga openocd vc707"
    if "vcu118" in target_type:
        tags = "fpga openocd vcu118"
    elif "hifive1-revb" in target_type:
        tags = "board jlink"
    elif "rtl" in target_type:
        tags = "rtl"
    elif "spike" in target_type:
        tags = "spike"
    elif "qemu" in target_type:
        tags = "qemu"
    else:
        tags = "board openocd"
    return tags


def get_port_width(tree):
    """Get the width of the RTL port, if the entry node specifies it"""
    metal_entry = tree.chosen("metal,entry")
    port_width = None
    if metal_entry:
        entry_node = tree.get_by_reference(metal_entry[0])

        # If the entry node is a testram, the parent node
        # is the port and has the port width
        port_width_bytes = entry_node.parent.get_field(
            "sifive,port-width-bytes")

        # If the entry node is /memory, the node itself
        # has the port width
        if port_width_bytes is None:
            port_width_bytes = entry_node.get_field(
                "sifive,port-width-bytes")

        if port_width_bytes is not None:
            port_width = 8 * port_width_bytes

    return port_width


def get_series(boot_hart, bitness):
    """Given the boot hart and the bitness, get the SiFive core series name"""
    hart_compat = boot_hart.get_field("compatible")
    series = None
    if "mallard" in hart_compat:
        series = "sifive-8-series"
    elif "bullet" in hart_compat:
        series = "sifive-7-series"
    elif "caboose" in hart_compat:
        series = "sifive-2-series"
    elif "rocket" in hart_compat:
        series = "sifive-3-series" if bitness == 32 else "sifive-5-series"
    return series


def main(argv):
    """Parse arguments, extract data, and render the settings.mk to file"""
    # pylint: disable=too-many-locals
    parsed_args = parse_arguments(argv)

    tree = pydevicetree.Devicetree.parseFile(
        parsed_args.dts, followIncludes=True)

    boot_hart = get_boot_hart(tree)

    archs = get_all_arch(tree)
    arch = arch2arch(get_greatest_common_arch(archs))
    bitness = 32 if "32" in arch else 64
    abi = arch2abi(arch)
    codemodel = "medlow" if bitness == 32 else "medany"

    series = get_series(boot_hart, bitness)
    intr_wait_cycle = 0
    if series is not None:
        intr_wait_cycle = 5000 if "8" in series else 0

    tags = type2tag(parsed_args.type)

    if "rtl" in parsed_args.type:
        dhry_iters = 2000
        core_iters = 5
        freertos_wait_ms = 10
    else:
        dhry_iters = 20000000
        core_iters = 5000
        freertos_wait_ms = 1000

    settings = """# Copyright (C) 2020 SiFive Inc
# SPDX-License-Identifier: Apache-2.0

RISCV_ARCH = %s
RISCV_ABI = %s
RISCV_CMODEL = %s
RISCV_SERIES = %s

TARGET_TAGS = %s
TARGET_DHRY_ITERS = %d
TARGET_CORE_ITERS = %d
TARGET_FREERTOS_WAIT_MS = %d
TARGET_INTR_WAIT_CYCLE  = %d""" %  (arch, abi, codemodel, series, tags, dhry_iters, core_iters,
                                    freertos_wait_ms, intr_wait_cycle)

    port_width = get_port_width(tree)
    if port_width is not None:
        settings += "\n\nCOREIP_MEM_WIDTH = %d" % port_width

    if parsed_args.output:
        parsed_args.output.write(settings)
        parsed_args.output.close()
    else:
        print(settings)


if __name__ == "__main__":
    main(sys.argv[1:])
