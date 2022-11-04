# devicetree-overlay-generator

This is a python tool based on pydevicetree
([GitHub](https://github.com/sifive/pydevicetree)/[PyPI](https://pypi.org/project/pydevicetree/))
which generates Devicetree overlays. The purpose of this tool is to bridge the gap between the
output of the hardware generator (for example, [Rocket Chip](https://github.com/chipsalliance/rocket-chip/))
and what [freedom-devicetree-tools](https://github.com/sifive/freedom-devicetree-tools/) expects
by describing implicit things like:

* Memories implemented by the RTL testbench
* The reset vector for the RTL testbench

as well as sane defaults for things like:

* The "boot hart" which does BSS initialization, etc.
* The standard out path

## Usage

```
usage: generate_overlay.py [-h] -t TYPE [-o OUTPUT]
                           [--rename-include RENAME_INCLUDE]
                           dts

Generate Devicetree overlays

positional arguments:
  dts                   The devicetree for the target

optional arguments:
  -h, --help            show this help message and exit
  -t TYPE, --type TYPE  The type of the target to generate an overlay for.
                        Supported types include: rtl, arty, qemu, hifive,
                        spike
  -o OUTPUT, --output OUTPUT
                        The name of the output file. If not provided, the
                        overlay is printed to stdout.
  --rename-include RENAME_INCLUDE
                        Rename the path of the include file in the generated
                        overlay to the provided value.
```

## Example

Given the Devicetree for the SiFive E31 CoreIP, you can run the following:

`$ ./generate_overlay.py --type rtl e31.dts`

and it will output the following Devicetree overlay:

```
/include/ "e31.dts"
/ {
        chosen {
                metal,boothart = <&L7>;
                metal,entry = <&testram0 0>;
        };
};
&L12 {
        testram0: testram@20000000 {
                compatible = "sifive,testram0";
                reg = <0x20000000 0x1fffffff>;
                reg-names = "mem";
        };
};
&L11 {
        testram1: testram@40000000 {
                compatible = "sifive,testram0";
                reg = <0x40000000 0x1fffffff>;
                reg-names = "mem";
        };
};
```

Note that the resulting overlay attaches the "testram" devices which are present in the RTL testbench
but not in the CoreIP Device-Under-Test, as well as describing the default reset vector with the 
`metal,entry` property in the `chosen` node.

## Copyright and License

Copyright (c) 2019 SiFive Inc.

The contents of this repository are distributed according to the terms described in the LICENSE
file.
