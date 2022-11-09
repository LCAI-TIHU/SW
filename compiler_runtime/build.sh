rm -r build
# build nvdla umd
pushd 3rdparty/sw/umd/ && source auto.sh
cd external && source protobuf_build.sh
popd
# build tvm
mkdir build
cd build
cp ../config.cmake .
cmake ..
make -j
cd ..
