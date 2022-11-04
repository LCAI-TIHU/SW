#!/nfs/tools/freeware/python/Python-3.7.5/bin/python3
##################################################################
# File name:    download_firmware.py
# Usage:
#   python3 download_firmware.py --help
# Description:
# this script works together with testexe to run testcases.
# this script also gives a chance to check testcase result here
# to check testcase result, please define a checker for testcase
# in testcase list file, and implement the checker in this script
# Author:   zhanghui02@inspur.com
# Date: 2020.10.10
##################################################################

import os
import re
import sys
import time
import shutil
import filecmp
import argparse
import datetime as dt

#global variable
g_log_folder=''

def append_log(verbose, *log_vars):

    global g_log_folder
    final_str=""
    for item in log_vars:
        if type(item) is not str:
            final_str=final_str+str(item)
        else:
            final_str=final_str+item
    #delay 1ms to ensure file is flushed
    if(verbose):
        print(final_str)
    time.sleep(0.1)

'''
description:
    call testexe to run testcase, return True if testexe succeeds
'''
def execute_tc(tc_cmd, verbose):

    global g_log_folder
    ret = True

    if verbose:
        #show log from testexe on screen, and save it to file
        full_cmd = tc_cmd+" 2>&1 | exit ${PIPESTATUS[0]}"
    else:
        #save log from testexe to file
        full_cmd = tc_cmd+" 2>&1"
    #print(full_cmd)
    if (0 != os.system(full_cmd)):
        ret = False

    time.sleep(0.1)
    return ret

'''
description:
    reset and prepare the SoC for a testcase
'''
def reset_prepare_soc(image, start_address, entry_point, verbose):

    #step 1, sign firmware image
    output_image_name = "firmware_host_signed_level_0.bin"
    address = start_address
    entry_point = entry_point
    if (address < "0x80000000"):
        size = ("%s") % (0x40600000 + 4 * 1024 * 1024)
    else:
        size = ("%s") % (0x80000000 + 2 * 1024 * 1024 * 1024)
    cmd = "python3 sign_wys.py -i %s -o %s -a %s -e %s -b %s > /dev/null 2>&1 || exit 1" %(image, output_image_name, address, entry_point, size)
    if ( 0 != os.system(cmd) ):
        append_log(True, "sign image failed")
        return False
    else:
        append_log(True, "sign image done")

    print("\033[34;1mPCIe hot_reset will take few seconds, please wait patiently...\033[0m")
    #step 2, reset PCIE and SoC
    time.sleep(0.1)
    cmd = "./hot_reset.sh > /dev/null 2>&1 || exit 1"
    if ( 0 != os.system(cmd) ):
        append_log(True, "download_firmware reset pcie (hot_reset.sh) failed")
        return False
    else:
        append_log(True, "download_firmware pcie reset done")

    append_log(True, "download_firmware start downloading firmware, please wait 30 seconds at most")
    time.sleep(0.1)
    tc_cmd = "./dma_download_firmware %s " %(output_image_name)
    if ( True != execute_tc(tc_cmd, verbose) ):
        append_log(True, "download_firmware firmware downloading failed")
        return False
    else:
        append_log(True, "download_firmware firmware downloading done")

    return True;

def run_tc_list(image, start_address, entry_point, verbose):

    append_log(True,"download_firmware starts at ", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    #reset the SoC and load FW firstly
    ret = reset_prepare_soc(image, start_address, entry_point, verbose)
    if (ret == False):
        append_log(True, "download_firmware reset_prepare_soc failed\n")
        #test_going_on = False #if soc reset failed will be continue next trip

    append_log(True,"download_firmware ends at ", (dt.datetime.now()).strftime("%Y-%m-%d %H:%M:%S"))


'''
description:
    build the executable binary, and copy it to runtest folder
'''
def make_testexe():

    print("download_firmware elf is buiding")
    cur_pwd = os.getcwd()

    os.chdir(os.path.join(os.path.abspath(os.path.pardir), "xdma"))
    ret = os.system("make clean all")
    if ret != 0:
        print("userspace code build error")
        return False
    else:
        #copy elf to runtest folder
        time.sleep(0.1)

        os.chdir(os.path.abspath(cur_pwd))
        os.system ("cp ../xdma/xdma.ko  ./xdma.ko")
        print("dma_download_firmware is built")

    os.chdir(os.path.join(os.path.abspath(os.path.pardir), "tools"))
    ret = os.system("make clean all")
    if ret != 0:
        print("userspace code build error")
        return False
    else:
        #copy testexe to runtest folder
        time.sleep(0.1)

        os.chdir(os.path.abspath(cur_pwd))
        os.system ("cp ../tools/dma_download_firmware  ./dma_download_firmware")
        print("dma_download_firmware is built")

    return True

'''
description:
    copy something to log folder for archiving. return False if it doesn't exist
'''
def copy_to_log_folder(src_file):

    global g_log_folder

    if not os.path.exists(src_file):
        return False
    if(g_log_folder):
        dirname, fname = os.path.split(src_file)
        target = os.path.join(g_log_folder, fname)
        shutil.copy(src_file, target)

    return True

'''
description:
    create log folder
'''
def prepare_log_folder():

    time_now = dt.datetime.now()
    archive_folder = "log_"+time_now.strftime("%Y%m%d%H%M%S")
    while(os.path.exists(archive_folder) == True):
        time.sleep(1)
        time_now = dt.datetime.now()
        archive_folder = "log_"+time_now.strftime("%Y%m%d%H%M%S")
    os.system("mkdir %s" % archive_folder)

    return (archive_folder)

'''
description:
    process the arguments
'''
def params_process(argv):

    parser = argparse.ArgumentParser(description='run a batch of testcases defined in a list, or a single testcase')
    parser.add_argument('-i', '--image', required=True, help="riscv will execute this firmware image which will be downloaded to soc")
    parser.add_argument('-a', '--address', required=True, help="riscv start address")
    parser.add_argument('-e', '--entrypoint', required=True, help="riscv image entrypoint")
    parser.add_argument('-v', '--verbose', default=False, action="store_true" ,help="if set, log will print at screen")

    args = parser.parse_args()

    print("\033[34;1mFirmware image address and entrypoint should be at(0x40600400~0x40A00000) or (0x80000000~0xFFFFFFFF)\033[0m")
    if (args.address < "0x40600400" or args.entrypoint < "0x40600400"):
        print("\033[31;1mFirmware image address and entrypoint should be large than 0x40600400\033[0m")
        exit(1)
    return [args.image, args.address, args.entrypoint, args.verbose]

def main(argv):

    global g_log_folder
    params = params_process(argv)

    make_ret = make_testexe()
    if make_ret != True:
        return

    g_log_folder = prepare_log_folder()

    if not copy_to_log_folder("xdma.ko"):
        print("download_firmware can not find the kernel module \'xdma.ko\'")
        exit(-1)
    if not copy_to_log_folder("dma_download_firmware"):
        print("download_firmware can not find the userspace executable \'dma_download_firmware\'")
        exit(-1)

    run_tc_list(params[0], params[1], params[2], params[3])
    print("download_firmware DONE")

    return


# entry of this script
if __name__ == "__main__":
    main(sys.argv[1:])
