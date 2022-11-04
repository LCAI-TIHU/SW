rm -r build
mkdir build
cd build
cp ../config.cmake .
cmake ..
make -j
cd ..
