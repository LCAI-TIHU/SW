#!/usr/bin/env python3
# Copyright (c) 2020 SiFive Inc.
# SPDX-License-Identifier: Apache-2.0

import unittest

import pydevicetree

from memory_map import *


def add_property(chosen, prop_s):
    chosen.properties.append(pydevicetree.Property.from_dts(prop_s))


class TestMemoryMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = pydevicetree.Devicetree.parseFile("tests/e31_no_chosen.dts")
        cls.chosen = cls.tree.get_by_path("/chosen")

        cls.testram_path = "/soc/ahb-periph-port@20000000/testram@20000000"
        cls.testram_base = 0x20000000
        cls.testram_length = 0x1fffffff

        cls.dtim_path = "/soc/dtim@80000000"
        cls.dtim_base = 0x80000000
        cls.dtim_length = 0x10000

        cls.itim_path = "/soc/itim@1800000"
        cls.itim_base = 0x1800000
        cls.itim_length = 0x2000

    def tearDown(self):
        # Clean up chosen properties between tests
        delete_names = ["metal,entry", "metal,ram", "metal,itim"]
        new_properties = [
            p for p in self.chosen.properties if p.name not in delete_names]
        self.chosen.properties = new_properties

    def test_get_chosen_region(self):
        add_property(self.chosen, "metal,entry = <&testram0 0 0>;")

        result = get_chosen_region(self.tree, "metal,entry")

        self.assertTrue(isinstance(result["node"], pydevicetree.Node))
        self.assertEqual(result["node"].get_path(
        ), "/soc/ahb-periph-port@20000000/testram@20000000")
        self.assertEqual(result["region"], 0)
        self.assertEqual(result["offset"], 0)

    def test_get_chosen_regions(self):
        add_property(self.chosen, "metal,entry = <&testram0 0 0>;")
        add_property(self.chosen, "metal,ram = <&L6 0 0>;")
        add_property(self.chosen, "metal,itim = <&L5 0 0>;")

        regions = get_chosen_regions(self.tree)

        self.assertTrue(isinstance(
            regions["entry"]["node"], pydevicetree.Node))
        self.assertEqual(
            regions["entry"]["node"].get_path(), self.testram_path)
        self.assertEqual(regions["entry"]["region"], 0)
        self.assertEqual(regions["entry"]["offset"], 0)
        self.assertTrue(isinstance(regions["ram"]["node"], pydevicetree.Node))
        self.assertEqual(regions["ram"]["node"].get_path(), self.dtim_path)
        self.assertEqual(regions["ram"]["region"], 0)
        self.assertEqual(regions["ram"]["offset"], 0)
        self.assertTrue(isinstance(regions["itim"]["node"], pydevicetree.Node))
        self.assertEqual(regions["itim"]["node"].get_path(), self.itim_path)
        self.assertEqual(regions["itim"]["region"], 0)
        self.assertEqual(regions["itim"]["offset"], 0)

    def test_compute_address_range(self):
        region = {
            "node": self.tree.get_by_path(self.testram_path),
            "region": 0,
            "offset": 0,
        }
        compute_address_range(region)

        self.assertEqual(region["base"], self.testram_base)
        self.assertEqual(region["length"], self.testram_length)

    def test_compute_address_ranges(self):
        regions = {
            "entry": {
                "node": self.tree.get_by_path(self.testram_path),
                "region": 0,
                "offset": 0,
            },
            "ram": {
                "node": self.tree.get_by_path(self.dtim_path),
                "region": 0,
                "offset": 0,
            },
            "itim": None,
        }
        compute_address_ranges(regions)

        self.assertEqual(regions["entry"]["base"], self.testram_base)
        self.assertEqual(regions["entry"]["length"], self.testram_length)
        self.assertEqual(regions["ram"]["base"], self.dtim_base)
        self.assertEqual(regions["ram"]["length"], self.dtim_length)

    def test_compute_address_ranges_with_overlap(self):
        regions = {
            "entry": {
                "node": self.tree.get_by_path(self.testram_path),
                "region": 0,
                "offset": 0,
            },
            "ram": {
                "node": self.tree.get_by_path(self.testram_path),
                "region": 0,
                "offset": 0x10000,
            },
            "itim": {
                "node": self.tree.get_by_path(self.testram_path),
                "region": 0,
                "offset": 0x20000,
            },
        }
        compute_address_ranges(regions)

        self.assertEqual(regions["entry"]["base"], self.testram_base)
        self.assertEqual(regions["entry"]["length"], 0x10000)
        self.assertEqual(regions["ram"]["base"], self.testram_base + 0x10000)
        self.assertEqual(regions["ram"]["length"], 0x10000)
        self.assertEqual(regions["itim"]["base"], self.testram_base + 0x20000)
        self.assertEqual(regions["itim"]["length"],
                         self.testram_length - 0x20000)

    def test_regions_overlap(self):
        regions = {
            "entry": {
                "base": 0x20000000,
                "length": 0x1fffffff,
            },
            "ram": {
                "base": 0x20000000,
                "length": 0x1fffffff,
            },
            "itim": {
                "base": 0x1800000,
                "length": 0x2000,
            },
        }

        self.assertTrue(regions_overlap(regions["entry"], regions["ram"]))
        self.assertFalse(regions_overlap(regions["entry"], regions["itim"]))
        self.assertFalse(regions_overlap(regions["ram"], regions["itim"]))

    def test_get_memories_ram_rom_itim(self):
        add_property(self.chosen, "metal,entry = <&testram0 0 0>;")
        add_property(self.chosen, "metal,ram = <&L6 0 0>;")
        add_property(self.chosen, "metal,itim = <&L5 0 0>;")

        memories = get_memories(self.tree)

        self.assertTrue("entry" in memories["rom"]["contents"])
        self.assertTrue("ram" in memories["ram"]["contents"])
        self.assertTrue("itim" in memories["itim"]["contents"])

    def test_get_memories_ram_rom_1(self):
        add_property(self.chosen, "metal,entry = <&testram0 0 0>;")
        add_property(self.chosen, "metal,ram = <&L6 0 0>;")
        add_property(self.chosen, "metal,itim = <&L6 0 0>;")

        memories = get_memories(self.tree)

        self.assertTrue("entry" in memories["rom"]["contents"])
        self.assertTrue("ram" in memories["ram"]["contents"])
        self.assertTrue("itim" in memories["ram"]["contents"])

    def test_get_memories_ram_rom_2(self):
        add_property(self.chosen, "metal,entry = <&testram0 0 0>;")
        add_property(self.chosen, "metal,ram = <&L6 0 0>;")
        add_property(self.chosen, "metal,itim = <&testram0 0 0>;")

        memories = get_memories(self.tree)

        self.assertTrue("entry" in memories["rom"]["contents"])
        self.assertTrue("ram" in memories["ram"]["contents"])
        self.assertTrue("itim" in memories["rom"]["contents"])

    def test_get_memories_testram_itim(self):
        add_property(self.chosen, "metal,entry = <&testram0 0 0>;")
        add_property(self.chosen, "metal,ram = <&testram0 0 0>;")
        add_property(self.chosen, "metal,itim = <&L5 0 0>;")

        memories = get_memories(self.tree)

        self.assertTrue("entry" in memories["testram"]["contents"])
        self.assertTrue("ram" in memories["testram"]["contents"])
        self.assertTrue("itim" in memories["itim"]["contents"])

    def test_get_memories_testram(self):
        add_property(self.chosen, "metal,entry = <&testram0 0 0>;")
        add_property(self.chosen, "metal,ram = <&testram0 0 0>;")
        add_property(self.chosen, "metal,itim = <&testram0 0 0>;")

        memories = get_memories(self.tree)

        self.assertTrue("entry" in memories["testram"]["contents"])
        self.assertTrue("ram" in memories["testram"]["contents"])
        self.assertTrue("itim" in memories["testram"]["contents"])

    def test_attributes_from_contents(self):
        self.assertEqual(attributes_from_contents(["entry"]), "irx!wa")
        self.assertEqual(attributes_from_contents(["ram"]), "arw!xi")
        self.assertEqual(attributes_from_contents(["itim"]), "airwx")
        self.assertEqual(attributes_from_contents(["entry", "ram"]), "airwx")
        self.assertEqual(attributes_from_contents(["ram", "itim"]), "airwx")
        self.assertEqual(attributes_from_contents(["entry", "itim"]), "airwx")
        self.assertEqual(attributes_from_contents(
            ["entry", "ram", "itim"]), "airwx")

    def test_format_hex(self):
        regions = {
            "entry": {
                "base": 0x20000000,
                "length": 0x1fffffff,
            },
            "ram": {
                "base": 0x80000000,
                "length": 0x10000,
            },
            "itim": {
                "base": 0x1800000,
                "length": 0x2000,
            },
        }
        format_hex(regions)

        self.assertEqual(regions["entry"]["base_hex"], "0x20000000")
        self.assertEqual(regions["entry"]["length_hex"], "0x1fffffff")
        self.assertEqual(regions["ram"]["base_hex"], "0x80000000")
        self.assertEqual(regions["ram"]["length_hex"], "0x10000")
        self.assertEqual(regions["itim"]["base_hex"], "0x1800000")
        self.assertEqual(regions["itim"]["length_hex"], "0x2000")


if __name__ == '__main__':
    unittest.main()
