# import tensorflow as tf

import os
import cv2
import numpy as np
import json

# tvm, relay
import tvm
from tvm import te
from tvm import relay
import onnx

# try:
#     tf_compat_v1 = tf.compat.v1
# except ImportError:
#     tf_compat_v1 = tf

# Tensorflow utility functions
# import tvm.relay.testing.tf as tf_testing
from tvm.contrib import graph_executor


import pdb
print(f'Python: {os.getpid()}')
# pdb.set_trace()

image_path = "./LeNet_TVM/t10k-images-idx3-ubyte"
label_path = "./LeNet_TVM/t10k-labels-idx1-ubyte"
model_path = "./LeNet_TVM/frozen_lenet_3.onnx"
#model_path = "./LeNet_TVM/frozen_lenet_3_modify.onnx"
image007_path = "./LeNet_TVM/00000_7.jpg"
#image007_path = "./LeNet_TVM/crop-230.jpg"
target = "llvm"
dev = tvm.device(target, 0)
layout="NCHW"

# Returns a numpy buffer of shape (num_images, 1, 28, 28)
def load_mnist_data(filepath):
    with open(filepath, "rb") as f:
        raw_buf = np.fromstring(f.read(), dtype=np.uint8)
    # Make sure the magic number is what we expect
    assert raw_buf[0:4].view(">i4")[0] == 2051
    num_images = raw_buf[4:8].view(">i4")[0]
    image_c = 1
    image_h = raw_buf[8:12].view(">i4")[0]
    image_w = raw_buf[12:16].view(">i4")[0]
    # Need to scale all values to the range of [-1.0, 1.0]
    return np.ascontiguousarray((raw_buf[16:] / 128.0 - 1.0).astype(np.float32).reshape(num_images, image_c, image_h, image_w))

# Returns a numpy buffer of shape (num_images)
def load_mnist_labels(filepath):
    with open(filepath, "rb") as f:
        raw_buf = np.fromstring(f.read(), dtype=np.uint8)
    # Make sure the magic number is what we expect
    assert raw_buf[0:4].view(">i4")[0] == 2049
    num_labels = raw_buf[4:8].view(">i4")[0]
    return np.ascontiguousarray(raw_buf[8:].astype(np.int32).reshape(num_labels))

test_data = load_mnist_data(image_path)
label_data = load_mnist_labels(label_path)
calibration_samples = 100

def calibrate_dataset():
    calib_data = []
    for data in test_data[:calibration_samples]:
        #print(data.shape)
        data = np.expand_dims(data, axis = 0)
        #print(data.shape)
        calib_data.append({'input:0': data})

    return calib_data

def quantize(mod, params, data_aware, target):
    if data_aware:
        #with relay.quantize.qconfig(calibrate_mode="kl_divergence", weight_scale="channel_max", skip_conv_layers=[], skip_dense_layer=False, do_simulation=False, layout=layout): #, dtype_input="uint8", debug_enabled_ops=["nn.conv2d"], calibrate_chunk_by=16
        #pdb.set_trace()
        weight_scale='channel_max'
        weight_scale='max'
        with relay.quantize.qconfig(target=target, calibrate_mode="percentile", weight_scale=weight_scale, skip_conv_layers=[], skip_dense_layer=False, do_simulation=True, layout=layout): #, dtype_input="uint8", debug_enabled_ops=["nn.conv2d"], calibrate_chunk_by=16
            #pdb.set_trace()
            mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset())
    else:
        with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
            mod = relay.quantize.quantize(mod, params)
    return mod


######################################################################
# Import model ==> Import the graph to Relay ==> Relay Build
# ------------
# Creates tensorflow graph definition from protobuf file.
# Import tensorflow graph definition to relay frontend.
# Compile the graph to llvm target with given input specification.
#
# Results:
#   sym: relay expr for given tensorflow protobuf.
#   params: params converted from tensorflow params (tensor protobuf).
#   lib: target library which can be deployed on target with TVM runtime.


###############################################################################
# The calibration dataset should be an iterable object. We define the
# calibration dataset as a generator object in Python. In this tutorial, we
# only use a few samples for calibration.


def create_graph(model_path):
    print(f'create_graph...')
    onnx_model = onnx.load(model_path)
    shape = {"input:0": (1, 1, 28, 28)}
    print(f'from_onnx...')
    relay_mod, params = relay.frontend.from_onnx(onnx_model, shape=shape)
    relay_mod = relay.transform.DynamicToStatic()(relay_mod)

    return relay_mod, params

def run_test(lib, dtype):
    image = cv2.imread(image007_path, 0)
    #print("1.",image.shape)
    image = np.expand_dims(image, axis=-1)
    #print("2.",image.shape)
    image = image / 128.0 - 1.0
    global target
    if target == "aipu":
        image = np.round(image / 0.00784248)
        image = np.clip(image, -127, 127)
    image = np.expand_dims(image, axis=0)
    #print("3.",image.shape)

    m = graph_executor.GraphModule(lib["default"](dev))
    m.set_input("input:0", tvm.nd.array(image.astype(dtype))) # set inputs
    m.run() # execute

    if target == "aipu":
        tvm_output = m.get_output(0, tvm.nd.empty((1,1,1,10))).asnumpy() # get outputs
    else:
        tvm_output = m.get_output(0).asnumpy()
    print("Runtime Done!!!")
    print("tvm_output: ", tvm_output)

    return tvm_output


if __name__ == '__main__':
#    run_test("a","b")
    mod, params = create_graph(model_path)
    print("-------------original model--------------")
    print(mod["main"].astext(show_meta_data=False))

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)

    tvm_output = run_test(lib, dtype='float32')

    target = "llvm"
    mod_quantized = quantize(mod, params, data_aware=True, target=target)
    print("-------------mod_quantized model--------------")
    print(mod_quantized["main"].astext(show_meta_data=False))

    # with tvm.transform.PassContext(opt_level=3):
    #     lib_quantized = relay.build(mod_quantized, target, params=params)

    # tvm_output = run_test(lib_quantized, dtype='float32')

    target = "aipu"
    mod_quantized = quantize(mod, params, data_aware=True, target=target)
    print("-------------mod_quantized model--------------")
    print(mod_quantized["main"].astext(show_meta_data=False))

    # target = "llvm"  # aipu部分算子不支持，nn.batch_flatten等
    # with tvm.transform.PassContext(opt_level=3):
    #     lib_quantized = relay.build(mod_quantized, target, params=params)

    # tvm_output = run_test(lib_quantized, dtype='float32')

