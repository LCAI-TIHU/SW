CC ?= gcc

all: reg_rw irq_rw riscv_irq_rw host_irq_rw dma_to_device dma_from_device performance dma_download_firmware

dma_to_device: dma_to_device.o
	$(CC) -lrt -o $@ $< -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -D_LARGE_FILE_SOURCE

dma_from_device: dma_from_device.o
	$(CC) -lrt -o $@ $< -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -D_LARGE_FILE_SOURCE

performance: performance.o
	$(CC) -o $@ $< -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -D_LARGE_FILE_SOURCE

dma_download_firmware: dma_download_firmware.o
	$(CC) -lrt -o $@ $< -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -D_LARGE_FILE_SOURCE -lpthread

reg_rw: reg_rw.o
	$(CC) -o $@ $<

irq_rw: irq_rw.o
	$(CC) -o $@ $<

irq_rw: riscv_irq_rw.o
	$(CC) -o $@ $<

irq_rw: host_irq_rw.o
	$(CC) -o $@ $<

%.o: %.c
	$(CC) -c -std=c99 -o $@ $< -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -D_LARGE_FILE_SOURCE

clean:
	rm -rf reg_rw irq_rw riscv_irq_rw host_irq_rw *.o *.bin dma_to_device dma_from_device performance dma_download_firmware

