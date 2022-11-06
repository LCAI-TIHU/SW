TIHU FIRMWARE
# Introduction
TIHU runs in barematel mode, and is driven by firmware to complete computing tasks submitted by RUNTIME. TIHU is comprised of RISC-V cpu, nvdla, NoC bus, PCIe module, DDR, SRAM, bootROM, DMA and peripherals.

The workflow of TIHU is:
    * Initialize CPU clock, serial, timer and interrupt;  
    * Wait for computing tasks submitted by RUNTIME;  
    * RUNTIME submits task-list and address-list to TIHU, and that will raise PCIe interrupt to CPU;  
    * Firmware assigns computing tasks to CPU or DLA based on device type;  
    * Host interrupt will be raised when all tasks are completed;  

# Compile and Download
Riscv-gnu-toolchain should be downloaded to compile firmware.  
` apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    bzip2 \
    rsync \
    wget `
