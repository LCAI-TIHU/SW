# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile Tensorflow Models
=========================
This article is an introductory tutorial to deploy tensorflow models with TVM.

For us to begin with, tensorflow python module is required to be installed.

Please refer to https://www.tensorflow.org/install
"""

# Tensorflow imports
import tensorflow as tf

import os
import cv2
import numpy as np
import json

# tvm, relay
import tvm
from tvm import te
from tvm import relay

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing
from tvm.contrib import graph_executor

from utils.inception_preprocessing import preprocess_image # mobilenet-v2

import pdb
print(f'Python: {os.getpid()}')

image_path = "/path/image_dir"
label_path = "/path/image_dir"
model_path = "/path/model_file"

layout = "NHWC"
label_offset = 1
output_number = 1001
batch_size = 1
total = 10000
calibration_samples = 1000
weight_scale="max"
AIPU_per_channel=tvm.get_global_func("AIPU_config_quantization_PER_FILTER")

with open(label_path) as f:
    lines_table = f.readlines()


# ###############################################################################
# # The calibration dataset should be an iterable object. We define the
# # calibration dataset as a generator object in Python. In this tutorial, we
# # only use a few samples for calibration.


def calibrate_dataset():
    
    calib_data = []
    for i in range(calibration_samples):
        data, _ = gen_data(i)
        data = np.expand_dims(data, axis = 0)
        calib_data.append({'input': data})

    return calib_data

def quantize(mod, params, target, weight_scale, data_aware, do_simulation):
    if data_aware:
        with relay.quantize.qconfig(target=target, calibrate_mode="kl_divergence", weight_scale=weight_scale, skip_conv_layers=[], do_simulation=do_simulation): #, dtype_input="uint8", debug_enabled_ops=["nn.conv2d"], calibrate_chunk_by=16
            mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset())
    else:
        with relay.quantize.qconfig(target=target, calibrate_mode="global_scale", global_scale=8.0):
            mod = relay.quantize.quantize(mod, params)
    return mod


def gen_data(i):
    line_table = lines_table[i].split()
    image_path = os.path.join(image_path, line_table[0])
    image_file = tf_compat_v1.gfile.FastGFile(image_path, 'rb')
    image_raw_data = image_file.read()
    image_file.close()
        
    image = tf.image.decode_jpeg(image_raw_data, channels = 3)
    image = preprocess_image(image, 224, 224, is_training=False)

    image = tf.keras.preprocessing.image.img_to_array(image)
    label = int(line_table[1]) + label_offset

    return image, label


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

def create_graph(model_path):
    print(f'create_graph...')
    with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name="")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        # Add shapes to the graph.
        with tf_compat_v1.Session() as sess:
            graph_def = tf_testing.AddShapesToGraphDef(sess, "MobilenetV2/Predictions/Softmax")

    data,_ = gen_data(0)
    data = np.expand_dims(data, axis = 0)
    shape_dict = {"input": data.shape}

    print(f'from_tensorflow...')
    mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)

    return mod, params


def run_test(lib, origin_shape=None):
    image = cv2.imread("/path/image_file")
    image = cv2.resize(image,(224, 224))
    image = image / 128.0 - 1.0
    image = np.expand_dims(image, axis=0)

    m = graph_executor.GraphModule(lib["default"](dev))
    global target
    if target == "aipu":
        image = np.round(image / 0.00787323)
        image = np.clip(image, -127, 127)
        image = np.expand_dims(image, axis=0)
        m.set_input("input", tvm.nd.array(image.astype("int8"))) # set inputs
        m.run() # execute
        tvm_output = m.get_output(0, tvm.nd.empty(origin_shape)).asnumpy() # get outputs
    else:
        image = np.expand_dims(image, axis=0)
        m.set_input("input", tvm.nd.array(image.astype("float32"))) # set inputs
        m.run() # execute
        tvm_output = m.get_output(0).asnumpy() # get outputs
    print("pred = ", np.argmax(tvm_output))

    return tvm_output

def run_inference(lib, origin_shape=None):
    top1_cnt = 0
    m = graph_executor.GraphModule(lib["default"](dev)) 
    global target
    for i in range(total):
        line_table = lines_table[i].split()
        image_path = os.path.join(image_path, line_table[0])
        image_file = tf_compat_v1.gfile.FastGFile(image_path, 'rb')
        image_raw_data = image_file.read()
        image_file.close()

        image = tf.image.decode_jpeg(image_raw_data, channels = 3)
        image = preprocess_image(image, 224, 224, is_training=False)

        image = tf.keras.preprocessing.image.img_to_array(image)
        label = int(line_table[1]) + label_offset

        if target == "aipu":
            image = np.round(image / 0.00787323)
            image = np.clip(image, -127, 127)
            image = np.expand_dims(image, axis=0)
            m.set_input("input", tvm.nd.array(image.astype("int8"))) # set inputs
            m.run() # execute
            tvm_output = m.get_output(0, tvm.nd.empty(origin_shape)).asnumpy() # get outputs
        else:
            image = np.expand_dims(image, axis=0)
            m.set_input("input", tvm.nd.array(image.astype("float32"))) # set inputs
            m.run() # execute
            tvm_output = m.get_output(0).asnumpy() # get outputs
        pred = np.argmax(tvm_output)
        print("result: ", i, pred, label, [pred == label])
        if pred == label:
            top1_cnt = top1_cnt + 1

    print("accuracy = %f" % (top1_cnt / total))
    return tvm_output

if __name__ == '__main__':

    # Ceate Relay graph
    mod, params = create_graph(model_path)

    # FP32 Inferrence
    target = "llvm"
    dev = tvm.device(target, 0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    out_float = run_test(lib)
    origin_shape = np.array(out_float.shape)

    # TVM Quantize
    mod_quantized = quantize(mod, params, target, weight_scale, data_aware=True, do_simulation=False)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod_quantized, target, params=params)
    out_tvm_int8 = run_test(lib, origin_shape)

    # AIPU Quantize    
    target = "aipu"
    if (weight_scale == "channel_max"):
        AIPU_per_channel(True)
    else: 
        AIPU_per_channel(False)
    mod_quantized = quantize(mod, params, target, weight_scale, data_aware=True, do_simulation=True)
    dev = tvm.device(target, 0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod_quantized, target, params=params)
    out_aipu_int8 = run_test(lib, origin_shape)
    run_inference(lib)

    # Finish
    print("MobileNet inferrence done !!!")

