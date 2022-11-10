### How to run
1.Enter docker
2.build protobuf:
  cd ./3rdparty/sw/umd/external
  source protobuf_build.sh
  cd ./../../../..
3.build umd and tvm:
  source build.sh
4.change the dir in env.sh
5.source env.sh:
  source env.sh
6.Run a test:
  cd AIPU_demo
  python3 from_tensorflow_quantize_lenet.py

