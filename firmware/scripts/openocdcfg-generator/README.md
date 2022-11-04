# openocdcfg-generator

This is a python tool based on pydevicetree
([GitHub](https://github.com/sifive/pydevicetree)/[PyPI](https://pypi.org/project/pydevicetree/))
which generates OpenOCD Configuration Files for Freedom Metal applications.

## Usage

```
usage: generate_openocdcfg.py [-h] -d DTS -b BOARD [-o OUTPUT] [-p PROTOCOL]
                              [-t]

Generate OpenOCD configuration files from Devicetrees

optional arguments:
  -h, --help            show this help message and exit
  -d DTS, --dts DTS     The path to the Devicetree for the target
  -b BOARD, --board BOARD
                        The board of the target to generate an OpenOCD config
                        for. Supported boards include: arty, vc707, hifive
  -o OUTPUT, --output OUTPUT
                        The path of the linker script file to output
  -p PROTOCOL, --protocol PROTOCOL
                        Supported protocols include: jtag, cjtag
  -t, --tunnel          Enable JTAG tunneling (Xilinx BSCAN)
```

## Required Devicetree Properties

This linker script generator expects that the Devicetree has annotated the desired memory map
for the resulting linker script through properties in the `/chosen` node. Those properties are:

  - `metal,entry`, which describes which memory region read-only data should be placed in
  - `metal,ram`, which describes which memory region should be treated as ram

Each of these properties is a `prop-encoded-array` with the following triplet of values:

  1. A reference to a node which describes memory with the `reg` property
  2. An integer describing which 0-indexed tuple in the `reg` property should be used
  3. An integer describing the offset into the memory region described by the requested `reg` tuple

For example, the chosen node may include the following properties:
```
chosen {
    metal,entry = <&testram0 0 0>;
    metal,ram = <&L28 0 0>;
};
```

## Copyright and License

Copyright (c) 2020 SiFive Inc.

The contents of this repository are distributed according to the terms described in the LICENSE
file.
