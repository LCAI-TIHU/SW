# Mapping Devicetree to Linker Script Contents

## Step 0: Devicetree

### Required `/chosen` node properties:

 - `metal,entry`
 - `metal,ram`

### Optional `/chosen` node properties:

 - `metal,itim`

## Step 1: Extract Devicetree Content

The following are the valid possible results of parsing the Devicetree

### All defined, none overlapping

 - `metal,entry`
 - `metal,ram`
 - `metal,itim`
 - All point at different memories in the design

```
{
    "entry": {
        "node": <Node a>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
    "ram": {
        "node": <Node b>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
    "itim": {
        "node": <Node c>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
}
```

### All defined, all overlapping

 - `metal,entry`
 - `metal,ram`
 - `metal,itim`
 - All point at the same memory in the design

```
{
    "entry": {
        "node": <Node a>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
    "ram": {
        "node": <Node a>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
    "itim": {
        "node": <Node a>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
}
```

### All defined, RAM and ITIM overlapping

 - `metal,entry`
 - `metal,ram`
 - `metal,itim`
 - RAM and ITIM point at the same memory in the design

```
{
    "entry": {
        "node": <Node a>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
    "ram": {
        "node": <Node b>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
    "itim": {
        "node": <Node b>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
}
```

### All defined, RAM and Entry overlapping

 - `metal,entry`
 - `metal,ram`
 - `metal,itim`
 - RAM and Entry point at the same memory in the design

```
{
    "entry": {
        "node": <Node a>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
    "ram": {
        "node": <Node a>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
    "itim": {
        "node": <Node b>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
}
```

### All defined, ITIM and Entry overlapping

 - `metal,entry`
 - `metal,ram`
 - `metal,itim`
 - RAM and Entry point at the same memory in the design

```
{
    "entry": {
        "node": <Node a>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
    "ram": {
        "node": <Node b>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
    "itim": {
        "node": <Node a>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
}
```

### RAM and Entry Defined, none overlapping

 - `metal,entry`
 - `metal,ram`
 - Both point at different memories in the design

```
{
    "entry": {
        "node": <Node a>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
    "ram": {
        "node": <Node b>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
}
```

### RAM and Entry Defined, both overlapping

 - `metal,entry`
 - `metal,ram`
 - Both point at the same memory in the design

```
{
    "entry": {
        "node": <Node a>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
    "ram": {
        "node": <Node b>, # type: pydevicetree.Node
        "region": 0, # type: int
        "offset", 0, # type: int
    },
}
```

### Different offsets in the same node

The different regions (entry, ram, itim) may point at the same node but to different offsets.
In this case, the regions should not be considered overlapping.

### Invalid states

 - `metal,entry` not provided
 - `metal,ram` not provided
 - Requested region or offset is not available in the device

## Step 2 Convert from `(node, region, offset)` to address range

### All defined, none overlapping

```
{
    "entry": {
        "base": 0x20000000, # type: int
        "length": 0x1000000, # type: int
    },
    "ram": {
        "base": 0x80000000, # type: int
        "length": 0x10000, # type: int
    },
    "itim": {
        "base": 0x8000000, # type: int
        "length": 0x8000, # type: int
    },
}
```

### All defined, all overlapping

```
{
    "entry": {
        "base": 0x60000000, # type: int
        "length": 0x100000, # type: int
    },
    "ram": {
        "base": 0x60000000, # type: int
        "length": 0x100000, # type: int
    },
    "itim": {
        "base": 0x60000000, # type: int
        "length": 0x100000, # type: int
    },
}
```

### All defined, RAM and ITIM overlapping

```
{
    "entry": {
        "base": 0x20000000, # type: int
        "length": 0x10000, # type: int
    },
    "ram": {
        "base": 0x80000000, # type: int
        "length": 0x10000, # type: int
    },
    "itim": {
        "base": 0x80000000, # type: int
        "length": 0x10000, # type: int
    },
}
```

### All defined, RAM and Entry overlapping

```
{
    "entry": {
        "base": 0x60000000, # type: int
        "length": 0x100000, # type: int
    },
    "ram": {
        "base": 0x60000000, # type: int
        "length": 0x100000, # type: int
    },
    "itim": {
        "base": 0x8000000, # type: int
        "length": 0x8000, # type: int
    },
}
```

### All defined, ITIM and Entry overlapping

```
{
    "entry": {
        "base": 0x60000000, # type: int
        "length": 0x100000, # type: int
    },
    "ram": {
        "base": 0x80000000, # type: int
        "length": 0x1000000, # type: int
    },
    "itim": {
        "base": 0x60000000, # type: int
        "length": 0x100000, # type: int
    },
}
```

### RAM and Entry Defined, none overlapping

```
{
    "entry": {
        "base": 0x20000000, # type: int
        "length": 0x10000, # type: int
    },
    "ram": {
        "base": 0x80000000, # type: int
        "length": 0x10000, # type: int
    },
}
```

### RAM and Entry Defined, both overlapping

```
{
    "entry": {
        "base": 0x60000000, # type: int
        "length": 0x100000, # type: int
    },
    "ram": {
        "base": 0x60000000, # type: int
        "length": 0x100000, # type: int
    },
}
```

## Step 3: Invert Structure to get memories and their contents

### RAM, ROM, ITIM

```
memories = {
    "rom" {
        "base": 0x20000000, # type: int
        "length": 0x100000, # type: int
        "contents": ["entry"] # type: List[str]
    },
    "ram" {
        "base": 0x80000000, # type: int
        "length": 0x1000000, # type: int
        "contents": ["ram"] # type: List[str]
    },
    "itim" {
        "base": 0x8000000 # type: int
        "length": 0x1000 # type: int
        "contents": ["itim"] # type: List[str]
    },
}
```

### RAM, ROM

```
memories = {
    "rom" {
        "base": 0x20000000, # type: int
        "length": 0x1000000, # type: int
        "contents": ["entry"] # type: List[str]
    },
    "ram" {
        "base": 0x80000000, # type: int
        "length": 0x1000000, # type: int
        "contents": ["ram", "itim"] # type: List[str]
    },
}
```

or

```
memories = {
    "rom" {
        "base": 0x20000000, # type: int
        "length": 0x1000000, # type: int
        "contents": ["entry", "itim"] # type: List[str]
    },
    "ram" {
        "base": 0x80000000, # type: int
        "length": 0x1000000, # type: int
        "contents": ["ram"] # type: List[str]
    },
}
```

### TESTRAM, ITIM

```
memories = {
    "testram" {
        "base": 0x20000000, # type: int
        "length": 0x1000000, # type: int
        "contents": ["entry", "ram"] # type: List[str]
    },
    "itim" {
        "base": 0x8000000 # type: int
        "length": 0x1000 # type: int
        "contents": ["itim"] # type: List[str]
    },
}
```

### TESTRAM

```
memories = {
    "testram" {
        "base": 0x20000000, # type: int
        "length": 0x1000000, # type: int
        "contents": ["entry", "ram", "itim"] # type: List[str]
    },
}
```

## Step 4: Add attributes based on contents

- "entry" gets "rxi"
- "ram" gets "rwa"
- "itim gets "rwxi"

### RAM, ROM, ITIM

```
memories = {
    "rom" {
        "base": 0x20000000, # type: int
        "length": 0x100000,0 # type: int
        "attributes": "rxi!wa", # type: str
        "contents": ["entry"] # type: List[str]
    },
    "ram" {
        "base": 0x80000000, # type: int
        "length": 0x1000000, # type: int
        "attributes": "rwa!xi", # type: str
        "contents": ["ram"] # type: List[str]
    },
    "itim" {
        "base": 0x8000000 # type: int
        "length": 0x1000 # type: int
        "attributes": "rwxai", # type: str
        "contents": ["itim"] # type: List[str]
    },
}
```

### RAM, ROM

```
memories = {
    "rom" {
        "base": 0x20000000, # type: int
        "length": 0x1000000, # type: int
        "attributes": "rxi!wa", # type: str
        "contents": ["entry"] # type: List[str]
    },
    "ram" {
        "base": 0x80000000, # type: int
        "length": 0x1000000, # type: int
        "attributes": "rwxa!i", # type: str
        "contents": ["ram", "itim"] # type: List[str]
    },
}
```

or

```
memories = {
    "rom" {
        "base": 0x20000000, # type: int
        "length": 0x1000000, # type: int
        "attributes": "rxi!wa", # type: str
        "contents": ["entry", "itim"] # type: List[str]
    },
    "ram" {
        "base": 0x80000000, # type: int
        "length": 0x1000000, # type: int
        "attributes": "rwxa!i", # type: str
        "contents": ["ram"] # type: List[str]
    },
}
```

### TESTRAM, ITIM

```
memories = {
    "testram" {
        "base": 0x20000000, # type: int
        "length": 0x1000000, # type: int
        "attributes": "rwxai", # type: str
        "contents": ["entry", "ram"] # type: List[str]
    },
    "itim" {
        "base": 0x8000000 # type: int
        "length": 0x1000 # type: int
        "attributes": "rwxai", # type: str
        "contents": ["itim"] # type: List[str]
    },
}
```

### TESTRAM

```
memories = {
    "testram" {
        "base": 0x20000000, # type: int
        "length": 0x1000000, # type: int
        "attributes": "rwxai", # type: str
        "contents": ["entry", "ram", "itim"] # type: List[str]
    },
}
```

## Step 5: Turn contents into lma/vma pairs

### RAM, ROM, ITIM

```
rom = {
    "lma": "rom",
}
ram = {
    "lma": "rom",
    "vma": "ram",
}
itim = {
    "lma": "rom",
    "vma": "itim",
}
```

### RAM, ROM

```
rom = {
    "lma": "rom",
}
ram = {
    "lma": "rom",
    "vma": "ram",
}
itim = {
    "lma": "rom",
    "vma": "ram",
}
```

or

```
rom = {
    "lma": "rom",
}
ram = {
    "lma": "rom",
    "vma": "ram",
}
itim = {
    "lma": "rom",
    "vma": "rom",
}
```

### TESTRAM, ITIM

```
rom = {
    "lma": "testram",
}
ram = {
    "lma": "testram",
    "vma": "testram",
}
itim = {
    "lma": "testram",
    "vma": "ram",
}
```

### TESTRAM

```
rom = {
    "lma": "testram",
}
ram = {
    "lma": "testram",
    "vma": "testram",
}
itim = {
    "lma": "testram",
    "vma": "testram",
}
```
