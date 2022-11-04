export TOP=$(pwd)
make clean
make -j8 compiler
make -j8 runtime
