#!/usr/bin/env python3
# Copyright (c) 2020 SiFive Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functions for converting Devicetrees to the template parameterization"""

import re
import sys

def get_ram_memories(tree):
    """Given a Devicetree, get the list of ram memories to describe in the
       linker script"""

    # RAMs (TIM, LIM, ILS, DLS, main memory) that may have ECC protection
    # define ram list filters
    itim_count = 0
    dtim_count = 0
    sram_count = 0
    ils_count = 0
    dls_count = 0
    memory_count = 0

    memories = dict()
    # Get a list of RAM nodes with node as the its key
    for node in list(tree.all_nodes()):
        if any([re.compile("itim").search(node.name), \
                re.compile("dtim").search(node.name), \
                re.compile("cache-controller").search(node.name), \
                re.compile("sys-sram").search(node.name), \
                re.compile("ils").search(node.name),  \
                re.compile("dls").search(node.name), \
                node.name == 'memory']):

            name = node.name.replace('-', '_')
            if name.isalpha():
                name += '_'
                if re.compile("itim").search(node.name):
                    name += str(itim_count)
                    itim_count += 1
                elif re.compile("dtim").search(node.name):
                    name += str(dtim_count)
                    dtim_count += 1
                elif re.compile("sys-sram").search(node.name):
                    name += str(sram_count)
                    sram_count += 1
                elif re.compile("ils").search(node.name):
                    name += str(ils_count)
                    ils_count += 1
                elif re.compile("dls").search(node.name):
                    name += str(dls_count)
                    dls_count += 1
                else:
                    name += str(memory_count)
                    memory_count += 1

            if name == "cache_controller" and len(node.get_reg()) == 2:
                name = "lim_0" # For now, only single LIM region per core design.

            region = dict()
            region.update(name=name, node=node, path=node.get_path(), \
                          region=0, offset=0)
            memories.update({name : region})

    if len(memories) == 0:
        return memories

    compute_address_ranges(memories)
    memories = consolidate_address_ranges(memories)
    format_hex(memories)

    return memories


def get_memories(tree):
    """Given a Devicetree, get the list of memories to describe in the
       linker script"""
    regions = get_chosen_regions(tree)
    compute_address_ranges(regions)
    memories = invert_regions_to_memories(regions)
    compute_attributes(memories)
    format_hex(memories)

    return memories


def get_load_map(memories, scratchpad):
    """Given the list of memories in the linker script, get the lma/vma
       pairs for each of the regions in the linker script"""
    ram = dict()
    rom = dict()
    itim = dict()
    lim = dict()

    if "testram" in memories:
        rom["vma"] = "testram"
        ram["lma"] = "testram"
        ram["vma"] = "testram"
        itim["lma"] = "testram"
        itim["vma"] = "testram"
        lim["lma"] = "testram"

        if "itim" in memories:
            itim["vma"] = "itim"
        if "itim" in memories["testram"]["contents"]:
            itim["vma"] = "testram"

        if "lim" in memories:
            lim["vma"] = "lim"
        else:
            lim["vma"] = "testram"
    else:
        if scratchpad:
            hex_load = "ram"
        else:
            hex_load = "rom"

        rom["vma"] = hex_load
        ram["lma"] = hex_load
        ram["vma"] = "ram"
        itim["lma"] = hex_load
        itim["vma"] = hex_load
        lim["lma"] = hex_load

        if "itim" in memories:
            itim["vma"] = "itim"
        if "itim" in memories["ram"]["contents"]:
            itim["vma"] = "ram"

        if "lim" in memories:
            lim["vma"] = "lim"
        else:
            lim["vma"] = "ram"

    return ram, rom, itim, lim

def get_chosen_region(dts, chosen_property_name):
    """Extract the requested address region from the chosen property"""
    chosen_property = dts.chosen(chosen_property_name)

    if chosen_property:
        chosen_node = dts.get_by_reference(chosen_property[0])
        chosen_region = chosen_property[1]
        chosen_offset = chosen_property[2]

        return {
            "node": chosen_node,
            "region": chosen_region,
            "offset": chosen_offset,
        }
    return None

def get_lim_region(dts):
    """Extract LIM region"""
    lim = dts.match("sifive,ccache0")

    if lim and len(lim[0].get_reg()) == 2:
        return {
            "node": lim[0],
            "region": 0,
            "offset": 0,
        }
    return None

def get_chosen_regions(tree):
    """Given the tree, get the regions requested by chosen properties.
       Exits with an error if required properties are missing or the
       parameters are invalid"""
    regions = {
        "entry": get_chosen_region(tree, "metal,entry"),
        "ram": get_chosen_region(tree, "metal,ram"),
        "itim": get_chosen_region(tree, "metal,itim"),
        "lim": get_lim_region(tree),
    }

    if regions["entry"] is None:
        print("ERROR: metal,entry is not defined by the Devicetree")
        sys.exit(1)
    if regions["ram"] is None:
        print("ERROR: metal,ram is not defined by the Devicetree")
        sys.exit(1)
    return regions


def consolidate_address_ranges(regions):
    """Given the requested regions, consolidate the region address ranges
       if they are contiguous"""
    sorted_list = list(regions.values())
    sorted_list.sort(key=lambda m: m["base"])
    print("RAM memories:", file=sys.stderr)
    for memory in sorted_list:
        print("\t%4s: 0x%08x-0x%08x" %
              (memory["name"], memory["base"], memory["length"]), file=sys.stderr)

    removal_list = []
    base = dict()
    base.update(name="", node=None, path="", region=0, offset=0)
    for region in sorted_list:
        if base["node"] is not None:
            if region["base"] == (base["base"] + base["length"]):
                # Next region is a continual from last, merge with current
                base["length"] += region["length"]
                removal_list.append(region)
            else:
                # Not continual, update for next region check
                base = region
        else:
            base = region

    # remove the collapse region from the list
    if len(removal_list) != 0:
        for region in removal_list:
            sorted_list.remove(region)

    # rebuild a new memories regions
    memories = dict()
    for region in sorted_list:
        memories.update({region["name"] : region})

    for _, memory in memories.items():
        if memory["length"] > 0x10000:
            # At most 64KB of each memory is zeroed, limit for RTL sim run
            memory["length"] = 0x10000

    return memories

def compute_address_range(region):
    """Extract the address range from the reg of the Node"""

    sideband = region["node"].get_reg().get_by_name("sideband")
    if sideband:
        block_size = region["node"].get_field("cache-block-size")
        sets = region["node"].get_field("cache-sets")
        # Way0 is always enabled. Substract corresponding memory area.
        region["length"] = region["node"].get_field("cache-size") - (sets * block_size)
        region["base"] = sideband[0]
    else:
        reg = region["node"].get_reg()
        base = reg[region["region"]][0] + region["offset"]
        length = reg[region["region"]][1] - region["offset"]
        region["base"] = base
        region["length"] = length

def compute_address_ranges(regions):
    """Given the requested regions, compute the effective address ranges
       to use for each"""
    for _, region in regions.items():
        if region is not None:
            compute_address_range(region)

    # partition regions by node
    region_values = [r for r in regions.values() if r is not None]
    nodes = list({region["node"] for region in region_values})
    for node in nodes:
        partition = [
            region for region in region_values if region["node"] == node]

        # sort regions by offset
        partition.sort(key=lambda x: x["offset"])

        # shorten regions so that they don't overlap
        if len(partition) >= 2 and partition[0]["base"] != partition[1]["base"]:
            partition[0]["length"] = partition[1]["base"] - \
                partition[0]["base"]
        if len(partition) >= 3 and partition[1]["base"] != partition[2]["base"]:
            partition[1]["length"] = partition[2]["base"] - \
                partition[1]["base"]

    return regions


def regions_overlap(region_a, region_b):
    """Test if regions are identical"""
    if region_a is None or region_b is None:
        return False
    return region_a["base"] == region_b["base"] and region_a["length"] == region_b["length"]


def invert_regions_to_memories(regions):
    """Given the requested regions with computed effective address ranges,
       invert the data structure to get the list of memories for the
       linker script"""
    memories = dict()

    if regions_overlap(regions["ram"], regions["entry"]):
        memories["testram"] = {
            "name": "testram",
            "base": regions["ram"]["base"],
            "length": regions["ram"]["length"],
            "contents": ["ram", "entry"],
            "path": regions["ram"]["node"].get_path()
        }
        if regions_overlap(regions["itim"], regions["entry"]):
            memories["testram"]["contents"].append("itim")
        elif regions["itim"] is not None:
            memories["itim"] = {
                "name": "itim",
                "base": regions["itim"]["base"],
                "length": regions["itim"]["length"],
                "contents": ["itim"],
                "path": regions["itim"]["node"].get_path()
            }
    else:
        memories["rom"] = {
            "name": "rom",
            "base": regions["entry"]["base"],
            "length": regions["entry"]["length"],
            "contents": ["entry"],
            "path": regions["entry"]["node"].get_path()
        }
        memories["ram"] = {
            "name": "ram",
            "base": regions["ram"]["base"],
            "length": regions["ram"]["length"],
            "contents": ["ram"],
            "path": regions["ram"]["node"].get_path()
        }
        if regions_overlap(regions["entry"], regions["itim"]):
            memories["rom"]["contents"].append("itim")
        elif regions_overlap(regions["ram"], regions["itim"]):
            memories["ram"]["contents"].append("itim")
        elif regions["itim"] is None:
            memories["ram"]["contents"].append("itim")
        else:
            memories["itim"] = {
                "name": "itim",
                "base": regions["itim"]["base"],
                "length": regions["itim"]["length"],
                "contents": ["itim"],
                "path": regions["itim"]["node"].get_path()
            }

    if regions["lim"] is not None:
        memories["lim"] = {
            "name": "lim",
            "base": regions["lim"]["base"],
            "length": regions["lim"]["length"],
            "contents": ["lim"],
            "path": regions["lim"]["node"].get_path()
        }

    return memories


def attributes_from_contents(contents):
    """Get the attributes from the contents of the memory"""
    attributes = ""
    if "entry" in contents:
        attributes += "rxi"
    if "ram" in contents:
        attributes += "rwa"
    if "itim" in contents:
        attributes += "rwxai"
    if "lim" in contents:
        attributes += "rwxai"

    # Remove duplicates
    attributes = ''.join(sorted(list(set(attributes))))

    antiattributes = ""
    for char in "rwxai":
        if char not in attributes:
            antiattributes += char

    if antiattributes != "":
        attributes += "!" + antiattributes

    return attributes


def compute_attributes(memories):
    """Given the list of memories and their contents, compute the linker
       script attributes"""
    for _, memory in memories.items():
        memory["attributes"] = attributes_from_contents(memory["contents"])


def format_hex(memories):
    """Provide hex-formatted base and length for parameterizing template"""
    for _, memory in memories.items():
        memory["base_hex"] = "0x%x" % memory["base"]
        memory["length_hex"] = "0x%x" % memory["length"]
