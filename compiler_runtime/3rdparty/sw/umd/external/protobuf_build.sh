#!/bin/bash
cd protobuf-2.6
./configure
autoreconf -vfi
make -j
#make install
cd ..
