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
import onnx

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
# pdb.set_trace()

image_label_path = "/workspace/imagenet/val.txt"
image_base = "/workspace/imagenet/val"
model_path = "/workspace/onnxmodel/resnet18_eager_mode_quantized.onnx"
label_offset = 0
output_number = 1000
batch_size = 1
total = 50000

with open(image_label_path) as f:
    lines_table = f.readlines()


###############################################################################
# The calibration dataset should be an iterable object. We define the
# calibration dataset as a generator object in Python. In this tutorial, we
# only use a few samples for calibration.



def gen_data(i):
    line_table = lines_table[i].split()
    image_path = os.path.join(image_base, line_table[0])
    image_file = tf_compat_v1.gfile.FastGFile(image_path, 'rb')
    image_raw_data = image_file.read()
    image_file.close()
        
    image = tf.image.decode_jpeg(image_raw_data, channels = 3)
    image = preprocess_image(image, 224, 224, is_training=False)

    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.transpose(image,(2,0,1))
    label = int(line_table[1]) + label_offset

    return image, label


######################################################################
# Import model ==> Import the graph to Relay ==> Relay Build

def create_graph(model_path):
    print(f'create_graph...')
    onnx_model = onnx.load(model_path)
    shape = {"x.1": (1, 3, 224, 224)}
    print(f'from_onnx...')
    relay_mod, params = relay.frontend.from_onnx(onnx_model, shape=shape, freeze_params=True)
    # relay_mod = relay.transform.DynamicToStatic()(relay_mod)

    return relay_mod, params


def run_test(lib, origin_shape=None):
    image = cv2.imread("/workspace/testimage/crop-230.jpg")
    image = cv2.resize(image,(224, 224))
    image = image / 128.0 - 1.0
    image = np.transpose(image,(2,0,1))
    image = np.expand_dims(image, axis=0)
    print(image.shape)

    m = graph_executor.GraphModule(lib["default"](dev))
    global target
    image = np.expand_dims(image, axis=0)
    m.set_input("x.1", tvm.nd.array(image.astype("float32"))) # set inputs
    m.run() # execute
    tvm_output = m.get_output(0).asnumpy() # get outputs
    print("pred = ", np.argmax(tvm_output))

    return tvm_output


if __name__ == '__main__':
    mod, params = create_graph(model_path)
    print("-------------original model--------------")
    print(mod["main"].astext(show_meta_data=False))

    target = "llvm"
    dev = tvm.device(target, 0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    out_float = run_test(lib)
    origin_shape = np.array(out_float.shape)
    print('*'*100, origin_shape)
    # run_inference(lib)
