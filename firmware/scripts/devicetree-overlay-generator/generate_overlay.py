#!/usr/bin/env python3
# Copyright (c) 2019 SiFive Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This program generates Devicetree overlays given the devicetree for the core
"""

import argparse
import os
import sys

import pydevicetree

import targets

SUPPORTED_TYPES = ["rtl", "arty", "qemu", "hifive", "spike", "vc707", "vcu118"]

# pylint: disable=too-many-branches
def main(argv):
    """Parse arguments and generate overlay"""
    arg_parser = argparse.ArgumentParser(description="Generate Devicetree overlays")

    arg_parser.add_argument("-t", "--type", required=True,
                            help="The type of the target to generate an overlay for. \
                                Supported types include: %s" % ", ".join(SUPPORTED_TYPES))
    arg_parser.add_argument("-o", "--output",
                            help="The name of the output file. If not provided, \
                                  the overlay is printed to stdout.")
    arg_parser.add_argument("--rename-include", help="Rename the path of the include file in \
                                the generated overlay to the provided value.")
    arg_parser.add_argument("dts", help="The devicetree for the target")

    parsed_args = arg_parser.parse_args(argv)

    if not any([t in parsed_args.type for t in SUPPORTED_TYPES]):
        print("Type '%s' is not supported, please choose one of: %s" % (parsed_args.type,
                                                                        ', '.join(SUPPORTED_TYPES)))
        sys.exit(1)

    try:
        os.stat(parsed_args.dts)
    except FileNotFoundError:
        print("Could not find file '%s'" % parsed_args.dts)
        sys.exit(1)

    tree = pydevicetree.Devicetree.parseFile(parsed_args.dts)

    if parsed_args.rename_include:
        include_path = parsed_args.rename_include
    else:
        include_path = parsed_args.dts

    overlay = pydevicetree.Devicetree.from_dts("""
    /include/ "%s"
    / {
        chosen {};
    };
    """ % include_path)

    if "rtl" in parsed_args.type:
        targets.testbench.generate_overlay(tree, overlay)
    elif "arty" in parsed_args.type:
        targets.arty.generate_overlay(tree, overlay)
    elif "vc707" in parsed_args.type:
        targets.vc707.generate_overlay(tree, overlay)
    elif "vcu118" in parsed_args.type:
        targets.vcu118.generate_overlay(tree, overlay)
    elif "qemu" in parsed_args.type:
        targets.qemu.generate_overlay(tree, overlay)
    elif "hifive" in parsed_args.type:
        targets.hifive.generate_overlay(tree, overlay)
    elif "spike" in parsed_args.type:
        targets.spike.generate_overlay(tree, overlay)

    if parsed_args.output:
        with open(parsed_args.output, "w") as output_file:
            output_file.write(overlay.to_dts())
    else:
        print(overlay)

if __name__ == "__main__":
    main(sys.argv[1:])
