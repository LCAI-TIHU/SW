# ldscript-generator

This is a python tool based on pydevicetree
([GitHub](https://github.com/sifive/pydevicetree)/[PyPI](https://pypi.org/project/pydevicetree/))
which generates linker scripts for Freedom Metal applications.

## Usage

```
usage: generate_ldscript.py [-h] -d DTS -o OUTPUT [--scratchpad | --ramrodata]

Generate linker scripts from Devicetrees

optional arguments:
  -h, --help            show this help message and exit
  -d DTS, --dts DTS     The path to the Devicetree for the target
  -o OUTPUT, --output OUTPUT
                        The path of the linker script file to output
  --scratchpad          Emits a linker script with the scratchpad layout
  --ramrodata           Emits a linker script with the ramrodata layout
  --freertos            Emits a linker script with specific layout for freertos
```

## Required Devicetree Properties

This linker script generator expects that the Devicetree has annotated the desired memory map
for the resulting linker script through properties in the `/chosen` node. Those properties are:

  - `metal,entry`, which describes which memory region read-only data should be placed in
  - `metal,itim`, which describes which memory region should be treated as instruction
    tightly-integrated memory for low-latency instruction fetch
  - `metal,ram`, which describes which memory region should be treated as ram

Each of these properties is a `prop-encoded-array` with the following triplet of values:

  1. A reference to a node which describes memory with the `reg` property
  2. An integer describing which 0-indexed tuple in the `reg` property should be used
  3. An integer describing the offset into the memory region described by the requested `reg` tuple

For example, the chosen node may include the following properties:
```
chosen {
    metal,entry = <&testram0 0 0>;
    metal,itim = <&L11 0 0>;
    metal,ram = <&L28 0 0>;
};
```

## Example Invocation

```
$ ./generate_ldscript.py -d e31.dts -o metal.default.lds
Generating linker script with default layout
Selected memories in design:
        RAM:  0x80000000-0x8000ffff (/soc/dtim@80000000)
        ITIM: 0x01800000-0x01801fff (/soc/itim@1800000)
        ROM:  0x20000000-0x3ffffffe (/soc/ahb-periph-port@20000000/testram@20000000)

$ head -n 20 metal.default.lds
/* Copyright (c) 2020 SiFive Inc. */
/* SPDX-License-Identifier: Apache-2.0 */
OUTPUT_ARCH("riscv")


/* Default Linker Script
 *
 * This is the default linker script for all Freedom Metal applications.
 */


ENTRY(_enter)

MEMORY
{
    ram (rwai!x) : ORIGIN = 0x80000000, LENGTH = 0x10000
    itim (rwxai) : ORIGIN = 0x1800000, LENGTH = 0x2000
    rom (rxai!w) : ORIGIN = 0x20000000, LENGTH = 0x1fffffff
    
}
```

## Copyright and License

Copyright (c) 2020 SiFive Inc.

The contents of this repository are distributed according to the terms described in the LICENSE
file.
