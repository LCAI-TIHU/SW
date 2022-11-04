#!/usr/bin/python3

import os
import sys
from struct import *
import random
import hashlib
import binascii
import argparse

def crc32(File):
    ff = open(File,"rb")
    data = ff.read()
    ff.close()
    crc = binascii.crc32(data) & 0xffffffff
    return crc

def encode_image(origin_image, addr, entry, length, f_out):
    f_name = f_out
    #print("%s" %(f_name))
    hf = open(f_name, "wb")
    # header begin
    hf.write(pack("<II", 0xFFFFFFFF, 0xA0A0A0A0))

    hf.write(pack("<II", 0xFFFFFFFF, addr))
    
    hf.write(pack("<II", 0xFFFFFFFF, length))

    hf.write(pack("<II", 0xFFFFFFFF, entry))

    hf.write(pack("<IIII", 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF))
    
    hf.write(pack("<I", 0xFFFFFFFF))
    #image hash
    hash_v = crc32(origin_image)
    #print ("0x%08X" %hash_v)
    hf.write(pack("<I",hash_v))
    hf.write(pack("<I", 0xFFFFFFFF))
    # hf.write(pack("<II", 0xFFFFFFFF, 0xFFFFFFFF))
    hf.close()
    
    hash_v = crc32(f_name)
    #print ("0x%08X" %hash_v)
    hf = open(f_name, "ab+")
    hf.seek(64 - 4)
    hf.write(pack("<I",hash_v))
    # header end, size is 0x40

    #pack header and image
    imgf = open(origin_image, "rb")
    content = imgf.read()
    imgf.close()
    hf.write(content)
    hf.close()

    # os.system("rm -rf .changed_image.bin")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure arguments')
    parser.add_argument('-i','--image',dest="image",default='',required=True,help="input file")
    parser.add_argument('-o','--output',dest="output",default='',required=True,help="output file")
    parser.add_argument('-a','--address',dest="address",default='',required=True,help="start address in hex")
    parser.add_argument('-e','--entry',dest="entry",default='',required=True,help="entry address in hex")
    parser.add_argument('-b','--boundary',dest="boundary",default='',required=True,help="boundary limit in hex")
    args = parser.parse_args()

    image = args.image
    f_out = args.output
    addr = int(args.address, 0)
    entry = int(args.entry, 0)
    boundary_limit = int(args.boundary, 0)

    image_size = os.path.getsize(image)

    #check size
    #print("image size is %d" %image_size)
    if (image_size + addr > boundary_limit):
        print("ERROR: Please check: %s size + start_address(%d + 0x%08X = 0x%08X) exceed boundary limitation 0x%08X\n" %(image, image_size, addr, (image_size + addr), boundary_limit))
        sys.exit(-1)

    #check entry
    if (entry < addr) or (entry > image_size + addr):
        print("ERROR: Entry 0x%08X is out of range (0x%08X ~ 0x%08X)" %(entry, image_size, image_size + addr))
        sys.exit(-1)

    #encode image
    encode_image(image, addr, entry, image_size, f_out)
