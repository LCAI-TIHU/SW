# Tensorflow imports
import tensorflow as tf

import os
import cv2
import numpy as np
import multiprocessing as mp
import json
import time

# tvm, relay
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import tvm.relay.testing.tf as tf_testing

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions

# from utils.vgg_preprocessing import preprocess_image
from tensorflow.keras.applications.vgg16 import preprocess_input

import pdb
print(f'Python: {os.getpid()}')
# pdb.set_trace()
print("---- start ----")


# Creates tensorflow graph definition from protobuf file.
image_label_path = "/workspace/tfmodel/vgg16/val.txt"
image_base = "/workspace/tfmodel/vgg16/val"
model_path = '/workspace/tfmodel/vgg16/vgg16.pb'
layout = "NHWC"
label_offset = 0
output_number = 1000
batch_size = 1
total = 10000
calibration_samples = 10

with open(image_label_path) as f:
    lines_table = f.readlines()

###############################################################################
# The calibration dataset should be an iterable object. We define the
# calibration dataset as a generator object in Python. In this tutorial, we
# only use a few samples for calibration.

def calibrate_dataset():
    
    calib_data = []
    for i in range(calibration_samples):
        data, _ = gen_data(i)
        calib_data.append({'input': data})

    return calib_data

def quantize(mod, params, target, data_aware, do_simulation=True):
    if data_aware:
        with relay.quantize.qconfig(target=target, calibrate_mode="kl_divergence", weight_scale="max", skip_conv_layers=[], skip_dense_layer=False,do_simulation=do_simulation): #, dtype_input="uint8", debug_enabled_ops=["nn.conv2d"], calibrate_chunk_by=16
            mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset())
    else:
        with relay.quantize.qconfig(target=target, calibrate_mode="global_scale", global_scale=8.0):
            mod = relay.quantize.quantize(mod, params)
    return mod


def gen_data(i):
    line_table = lines_table[i].split()
    image_path = os.path.join(image_base, line_table[0])
    label = int(line_table[1]) + label_offset

    # image_file = tf_compat_v1.gfile.FastGFile(image_path, 'rb')
    # image_raw_data = image_file.read()
    # image_file.close()
    # image = tf.image.decode_jpeg(image_raw_data, channels = 3)
    # image = preprocess_image(image, 224, 224, is_training=False)
    # image = tf.keras.preprocessing.image.img_to_array(image)

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(img)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

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
            graph_def = tf_testing.AddShapesToGraphDef(sess, "vgg16/predictions/Softmax") # vgg16/predictions/Softmax  vgg16/flatten/Reshape

    data,_ = gen_data(0)
    shape_dict = {"input": data.shape}
    print(shape_dict)

    print(f'from_tensorflow...')
    mod = tvm.IRModule()
    mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)

    return mod, params

def run_test(lib, origin_shape=None):
    img = tf.keras.preprocessing.image.load_img("/workspace/testimage/crop-230.jpg", target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(img)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    global target
    if target == "aipu":
        image = np.round(image / 1.11223)
        image = np.clip(image, -127, 127)
        m = graph_executor.GraphModule(lib["default"](dev))
        m.set_input("input", tvm.nd.array(image.astype("int8"))) # set inputs
        m.run() # execute
        tvm_output = m.get_output(0, tvm.nd.empty(origin_shape)).asnumpy() # get outputs
    else:
        m = graph_executor.GraphModule(lib["default"](dev))
        m.set_input("input", tvm.nd.array(image.astype("float32"))) # set inputs
        m.run() # execute
        tvm_output = m.get_output(0).asnumpy() # get outputs

    print("pred = ", np.argmax(tvm_output))

    return tvm_output

def run_inference(lib, origin_shape=None):
    print('make run inference!')
    top1_cnt = 0
    m = graph_executor.GraphModule(lib["default"](dev)) 
    global target
    for i in range(0, total, batch_size):
        line_table = lines_table[i].split()
        image_path = os.path.join(image_base, line_table[0])

        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(img)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        label = int(line_table[1]) + label_offset
        if target == "aipu":
            image = np.round(image / 1.11223)
            image = np.clip(image, -127, 127)
            m = graph_executor.GraphModule(lib["default"](dev))
            m.set_input("input", tvm.nd.array(image.astype("int8"))) # set inputs
            m.run() # execute
            tvm_output = m.get_output(0, tvm.nd.empty(origin_shape)).asnumpy() # get outputs
        else:
            m = graph_executor.GraphModule(lib["default"](dev))
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
    mod, params = create_graph(model_path)
    print("-------------original model--------------")
    print(mod["main"].astext(show_meta_data=False))

    target = "llvm"
    dev = tvm.device(target, 0)
    with tvm.transform.PassContext(opt_level=3):
        print("I am here before relay build")
        lib = relay.build(mod, target, params=params)
        print("After build ...........................................")
    out_float = run_test(lib)
    origin_shape = np.array(out_float.shape)
    print('*'*100, origin_shape)
    # run_inference(lib)

    # mod_quantized = quantize(mod, params, data_aware=True, do_simulation=False)
    # print("-------------mod_quantized model--------------")
    # print(mod_quantized["main"].astext(show_meta_data=False))
    # with tvm.transform.PassContext(opt_level=3):
    #     print("I am here before relay build")
    #     lib = relay.build(mod_quantized, target, params=params)
    #     print("After build ...........................................")
    # out_tvm_int8 = run_test(lib)
    # run_inference(lib)

    target = "aipu"
    dev = tvm.device(target, 0)
    mod_quantized = quantize(mod, params, target, data_aware=True, do_simulation=True)
    print("-------------mod_quantized model--------------")
    print(mod_quantized["main"].astext(show_meta_data=False))

    with tvm.transform.PassContext(opt_level=3):
        print("I am here before relay build")
        lib = relay.build(mod_quantized, target, params=params)
        print("After build ...........................................")
    out_aipu_int8 = run_test(lib, origin_shape)

    # print('AIPU error rate: ', np.sum(np.abs(out_float - out_aipu_int8)) / np.sum(np.abs(out_float)))
    # print('tvm error rate: ', np.sum(np.abs(out_float - out_tvm_int8)) / np.sum(np.abs(out_float)))

