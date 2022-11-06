#! /bin/sh

sudo su <<EOF
pwd

echo "Insmod xdma.ko."
insmod ../xdma/xdma.ko

echo "Download firmware."
## copy firmware 
cp ../../firmware/src/debug/firmware.bin ./

## download to ram
# python3 download_firmware.py -i firmware.bin -a 0x40640000 -e  0x40640000
# echo "python3 download_firmware.py -i firmware.bin -a 0x40640000 -e  0x40640000"

##download to ddr
python3 download_firmware.py -i firmware.bin -a 0x80000000 -e  0x80000000
echo "python3 download_firmware.py -i firmware.bin -a 0x80000000 -e  0x80000000"

EOF
