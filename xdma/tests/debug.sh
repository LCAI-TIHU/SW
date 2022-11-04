#! /bin/sh
insmod ../xdma/xdma.ko
python3 download_firmware.py -i firmware.bin -a 0x80000000 -e  0x80000000
echo "python3 download_firmware.py -i main.bin -a 0x80000000 -e  0x80000000"
