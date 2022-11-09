rm -r build
# build nvdla umd
pushd 3rdparty/sw/umd/external && source protobuf_build.sh
cd .. && make clean && source auto.sh
popd
# build tvm
mkdir build
cd build
cp ../config.cmake .
cmake ..
make -j
cd ..
