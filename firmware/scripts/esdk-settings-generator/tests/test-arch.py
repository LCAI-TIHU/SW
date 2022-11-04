#!/usr/bin/env python3
# Copyright (c) 2020 SiFive Inc.
# SPDX-License-Identifier: Apache-2.0

import unittest

from generate_settings import get_greatest_common_arch

class TestArch(unittest.TestCase):
    def test_greatest_common_arch(self):
        self.assertEqual(get_greatest_common_arch(["rv32imac", "rv32imac"]), "rv32imac")
        self.assertEqual(get_greatest_common_arch(["rv32imac", "rv32imafc"]), "rv32imac")
        self.assertEqual(get_greatest_common_arch(["rv64imac", "rv64imafdc"]), "rv64imac")
        self.assertEqual(get_greatest_common_arch(["rv64ima", "rv64ifdc"]), "rv64i")

if __name__ == '__main__':
    unittest.main()
