
import tensorflow as tf

import os
import cv2
import numpy as np
import json

import tvm
from tvm import relay
from tvm.contrib import graph_executor

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

image_path = "/path/image_dir"
label_path = "/path/image_dir"
model_path = "/path/model_file"
layout = "NHWC"
calibration_samples = 1000 
weight_scale = "channel_max"
AIPU_per_channel=tvm.get_global_func("AIPU_config_quantization_PER_FILTER")


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
    return np.ascontiguousarray((raw_buf[16:] / 128.0 - 1.0).astype(np.float32).reshape(num_images, image_h, image_w, image_c))

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

def calibrate_dataset():
    
    calib_data = []
    for data in test_data[:calibration_samples]:
        data = np.expand_dims(data, axis = 0)
        calib_data.append({'input': data})

    return calib_data

def quantize(mod, params, target, weight_scale, data_aware, do_simulation=True):
    if data_aware:
        with relay.quantize.qconfig(target=target, calibrate_mode="kl_divergence", weight_scale=weight_scale, skip_conv_layers=[], skip_dense_layer=False, do_simulation=True): #, dtype_input="uint8", debug_enabled_ops=["nn.conv2d"], calibrate_chunk_by=16
            mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset())
    else:
        with relay.quantize.qconfig(target=target, calibrate_mode="global_scale", global_scale=8.0):
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
            graph_def = tf_testing.AddShapesToGraphDef(sess, out_node="Predictions/Softmax") 

    print(f'from_tensorflow...')
    shape_dict = {"input": (1, 28, 28, 1)}
    mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)

    return mod, params

  
def run_test(lib, origin_shape=None):
    image = cv2.imread("/path/image_file", 0)
    image = np.expand_dims(image, axis=-1)
    image = image / 128.0 - 1.0
    global target

    m = graph_executor.GraphModule(lib["default"](dev))
    if target == "aipu":#aipu需要预量化
        image = np.round(image / 0.00784248)
        image = np.clip(image, -127, 127)
        image = np.expand_dims(image, axis=0)
        m.set_input("input", tvm.nd.array(image.astype("int8"))) # set inputs
        m.run() # execute
        tvm_output = m.get_output(0, tvm.nd.empty(origin_shape)).asnumpy() # get outputs
    else:
        image = np.expand_dims(image, axis=0)
        m.set_input("input", tvm.nd.array(image.astype("float32"))) # set inputs
        m.run() # execute
        tvm_output = m.get_output(0).asnumpy() 
    print("Runtime Done!!!")
    print("tvm_output: ", tvm_output)

    return tvm_output


###############################################################################
# Run Inference
# -------------
# We create a Relay VM to build and execute the model.

def run_inference(lib):
    ######################################################################
    # Execute the portable graph on TVM
    # ---------------------------------
    # Now we can try deploying the compiled model on target.

    batch_num = 0
    top1_cnt = 0
    error_list = []
    total = len(test_data)
    batch_size = 1
    m = graph_executor.GraphModule(lib["default"](dev))
    global target
    for start_idx in range(0, total, batch_size):
        batch_num += 1
        if batch_num % 10 == 0:
            print(f'Validating batch {batch_num} / {int(total / batch_size + 0.5)}')

        end_idx = min(start_idx + batch_size, total)
        effective_batch_size = end_idx - start_idx
       
        batch = []
        for i in range(start_idx, start_idx+effective_batch_size, 1):
            batch.append((test_data[i], label_data[i]))
     
        labels_batch = np.array([x[1] for x in batch])
        
        if target == "aipu":
            images_batch = np.array([np.clip(np.round(x[0] / 0.00784248), -127, 127) for x in batch]) 
            m.set_input("input", tvm.nd.array(images_batch.astype("int8"))) # set inputs
            m.run() # execute
            tvm_output = m.get_output(0, tvm.nd.empty(origin_shape)).asnumpy() # get outputs
        else:
            images_batch = np.array([x[0] for x in batch]) 
            m.set_input("input", tvm.nd.array(images_batch.astype("float32"))) # set inputs
            m.run() # execute
            tvm_output = m.get_output(0).asnumpy() 

        preds = np.argmax(tvm_output.reshape(batch_size, 10)[0:effective_batch_size], axis=1)
        top1_cnt += np.count_nonzero(np.equal(preds, labels_batch))
        print(start_idx, preds, labels_batch, np.equal(preds, labels_batch))
        if preds != labels_batch:
            error_list.append(start_idx)

    print(f"Total : {total}, accuracy = {top1_cnt / total}")
    print(error_list)

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
    print("target: ",target)
    mod_quantized = quantize(mod, params,  target, weight_scale, data_aware=True)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod_quantized, target, params=params)
    out_tvm_int8 = run_test(lib, origin_shape)

    # AIPU Quantize
    target = "aipu"

    if (weight_scale == "channel_max"):
        AIPU_per_channel(True)
    else: 
        AIPU_per_channel(False)

    mod_quantized = quantize(mod, params, target, weight_scale, data_aware=True)
    dev = tvm.device(target, 0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod_quantized, target, params=params)
    aipu_output = run_test(lib, origin_shape)
    run_inference(lib)
   
    # Finish
    print("Lenet inferrence done !!!")
