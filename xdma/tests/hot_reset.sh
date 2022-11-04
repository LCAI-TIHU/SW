#!/bin/sh

rmmod drv_kmem
rmmod pci_hwlogic_driver
rmmod inspur_common
rmmod inspur_aem
rmmod postsi_val_dev
#rmmod xdma
echo "hot reseting.........................."
dev=`lspci | grep Xilinx | cut -d ' ' -f 1`
dev="0000:"$dev
if [ ! -e "/sys/bus/pci/devices/$dev" ]; then
	echo "Error: device $dev not found"
	exit 1
fi

port=$(basename $(dirname $(readlink "/sys/bus/pci/devices/$dev")))
# echo /sys/bus/pci/devices/$dev
# echo $(readlink "/sys/bus/pci/devices/$dev")
# echo $(dirname $(readlink "/sys/bus/pci/devices/$dev"))
# printf "port = %s\n" $port

if [ ! -e "/sys/bus/pci/devices/$port" ]; then
	echo "Error: device $port not found"
	exit 1
fi

echo "Removing device"
echo 1 > /sys/bus/pci/devices/$dev/remove
sleep 3

reg1=$(setpci -s $port BRIDGE_CONTROL)
#old_reg=$((16#${reg1}))
old_reg=0x10
printf "Bridge control old:0x%x\n" $old_reg

# new_reg=$(($old_reg|0x40))
new_reg=0x50
printf "Bridge control new:0x%x\n" $new_reg

setpci -s $port BRIDGE_CONTROL=$new_reg
sleep 3

setpci -s $port BRIDGE_CONTROL=$old_reg
sleep 3


dev=`lspci | grep Xilinx`
echo "$dev"

echo "Rescanning device"
echo 1 > /sys/bus/pci/devices/$port/rescan
sleep 3

dev=`lspci | grep Xilinx | cut -b 1-7 | xargs lspci -vvv -s | grep Region`
echo "$dev"

#./load_driver.sh
