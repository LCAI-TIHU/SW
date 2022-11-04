#!/usr/bin/env python3
# Copyright (c) 2020 SiFive Inc.
# SPDX-License-Identifier: Apache-2.0

import unittest

import pydevicetree

from targets.generic import get_rams
from targets.testbench import get_testram, attach_testrams, get_boot_rom

class TestRTL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e20_tree = pydevicetree.Devicetree.parseFile("tests/e20.dts")
        cls.e31_tree = pydevicetree.Devicetree.parseFile("tests/e31.dts")
        cls.u54_tree = pydevicetree.Devicetree.parseFile("tests/u54.dts")
        cls.u54mc_tree = pydevicetree.Devicetree.parseFile("tests/u54mc.dts")

        cls.trees = [cls.e20_tree, cls.e31_tree, cls.u54_tree, cls.u54mc_tree]

        cls.e20_overlay = pydevicetree.Devicetree.from_dts("/{chosen{};};")
        cls.e31_overlay = pydevicetree.Devicetree.from_dts("/{chosen{};};")
        cls.u54_overlay = pydevicetree.Devicetree.from_dts("/{chosen{};};")
        cls.u54mc_overlay = pydevicetree.Devicetree.from_dts("/{chosen{};};")

        cls.overlays = [cls.e20_overlay, cls.e31_overlay, cls.u54_overlay, cls.u54mc_overlay]

        for tree, overlay in zip(cls.trees, cls.overlays):
            attach_testrams(tree, overlay)

        cls.e20_boot_port_path = "/soc/ahb-sys-port@20000000"
        cls.e31_boot_port_path = "/soc/ahb-periph-port@20000000"
        cls.u54_boot_port_path = "/soc/axi4-periph-port@20000000"
        cls.u54mc_boot_port_path = "/soc/axi4-periph-port@20000000"

    def test_get_testram(self):
        for tree in self.trees:
            for port in tree.match("sifive,.*port"):
                testram = get_testram(port, "testram")
                self.assertIsInstance(testram, pydevicetree.Node)

    def test_attach_testrams(self):
        for tree, overlay in zip(self.trees, self.overlays):
            ports = tree.match("sifive,.*port")
            testrams = tree.match("sifive,testram0")

            overlay_testrams = overlay.match("sifive,testram0")

            self.assertEqual(len(ports), len(testrams))
            self.assertEqual(len(testrams), len(overlay_testrams))

            for port in ports:
                for child in port.children:
                    if child.get_field("compatible") == "sifive,testram0":
                        self.assertIn(child, testrams)

    def test_get_bootrom(self):
        e20_boot_rom = get_boot_rom(self.e20_tree)
        self.assertIs(e20_boot_rom.parent, self.e20_tree.get_by_path(self.e20_boot_port_path))

        e31_boot_rom = get_boot_rom(self.e31_tree)
        self.assertIs(e31_boot_rom.parent, self.e31_tree.get_by_path(self.e31_boot_port_path))

        u54_boot_rom = get_boot_rom(self.u54_tree)
        self.assertIs(u54_boot_rom.parent, self.u54_tree.get_by_path(self.u54_boot_port_path))

        u54mc_boot_rom = get_boot_rom(self.u54mc_tree)
        self.assertIs(u54mc_boot_rom.parent, self.u54mc_tree.get_by_path(self.u54mc_boot_port_path))

    def test_rtl_rams(self):
        e20_ram, e20_itim = get_rams(self.e20_tree)
        self.assertIs(e20_ram, None)
        self.assertIs(e20_itim, None)

        e31_ram, e31_itim = get_rams(self.e31_tree)
        self.assertIs(e31_ram, self.e31_tree.get_by_path("/soc/dtim@80000000"))
        self.assertIs(e31_itim, self.e31_tree.get_by_path("/soc/itim@1800000"))

        u54_ram, u54_itim = get_rams(self.u54_tree)
        self.assertIs(u54_ram, self.u54_tree.get_by_path("/memory"))
        self.assertIs(u54_itim, self.u54_tree.get_by_path("/soc/itim"))

        u54mc_ram, u54mc_itim = get_rams(self.u54mc_tree)
        self.assertIs(u54mc_ram, self.u54mc_tree.get_by_path("/memory"))
        self.assertIs(u54mc_itim, self.u54mc_tree.get_by_path("/soc/itim@1820000"))

if __name__ == "__main__":
    unittest.main()
