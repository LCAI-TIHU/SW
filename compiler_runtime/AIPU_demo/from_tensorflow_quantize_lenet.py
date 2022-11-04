
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


image_path = "/workspace/mnist/t10k-images-idx3-ubyte"
label_path = "/workspace/mnist/t10k-labels-idx1-ubyte"
model_path = "/workspace/tfmodel/frozen_lenet_3.pb"
layout = "NHWC"
calibration_samples = 1000

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

def quantize(mod, params, target, data_aware):
    if data_aware:
        with relay.quantize.qconfig(target=target, calibrate_mode="kl_divergence", weight_scale="max", skip_conv_layers=[], skip_dense_layer=False, do_simulation=True): #, dtype_input="uint8", debug_enabled_ops=["nn.conv2d"], calibrate_chunk_by=16
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
            graph_def = tf_testing.AddShapesToGraphDef(sess, "Predictions/Softmax")

    print(f'from_tensorflow...')
    shape_dict = {"input": (1, 28, 28, 1)}
    mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)

    return mod, params

    
def run_test(lib, origin_shape=None):
    image = cv2.imread("/workspace/testimage/00000_7.jpg", 0)
    image = np.expand_dims(image, axis=-1)
    image = image / 128.0 - 1.0
    global target

    m = graph_executor.GraphModule(lib["default"](dev))
    if target == "aipu":
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
    # pool = mp.Pool(processes=mp.cpu_count())
    error_list = []
    total = len(test_data)
    # total = 100
    batch_size = 1
    m = graph_executor.GraphModule(lib["default"](dev))
    global target
    for start_idx in range(0, total, batch_size):
        batch_num += 1
        if batch_num % 10 == 0:
            print(f'Validating batch {batch_num} / {int(total / batch_size + 0.5)}')

        end_idx = min(start_idx + batch_size, total)
        effective_batch_size = end_idx - start_idx
        # batch = pool.map(gen_data, range(start_idx, start_idx+effective_batch_size, 1))  // multiprocessing vscode debug时有bug

        batch = []
        for i in range(start_idx, start_idx+effective_batch_size, 1):
            batch.append((test_data[i], label_data[i]))
     
        # images_batch = np.array([x[0] for x in batch]) 
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

        # preds = np.array([tvm_output])
        preds = np.argmax(tvm_output.reshape(batch_size, 10)[0:effective_batch_size], axis=1)
        top1_cnt += np.count_nonzero(np.equal(preds, labels_batch))
        print(start_idx, preds, labels_batch, np.equal(preds, labels_batch))
        if preds != labels_batch:
            error_list.append(start_idx)

    # # pool.close()
    print(f"Total : {total}, accuracy = {top1_cnt / total}")
    print(error_list)

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
    # mod_quantized_path = 'mod_quantized_lenet.json'
    # if not os.path.exists(mod_quantized_path):
    #     mod_quantized = quantize(mod, params, data_aware=True)
    #     print("-------------mod_quantized model--------------")
    #     print(mod_quantized["main"].astext(show_meta_data=False))
    #     json_str = tvm.ir.save_json(mod_quantized)
    #     json_data = json.loads(json_str)
    #     with open(mod_quantized_path, 'w') as f:
    #         json.dump(json_data, f)
    # else:
    #     with open(mod_quantized_path, 'r') as f:
    #         data = json.load(f)
    #     json_data = json.dumps(data)
    #     mod_quantized = tvm.ir.load_json(json_data)
    #     print("-------------mod_quantized model--------------")
    #     print(mod_quantized["main"].astext(show_meta_data=False))
    target = "aipu"
    mod_quantized = quantize(mod, params, target, data_aware=True)
    print("-------------mod_quantized model--------------")
    print(mod_quantized["main"].astext(show_meta_data=False))

    target = "aipu"
    dev = tvm.device(target, 0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod_quantized, target, params=params)

    aipu_output = run_test(lib, origin_shape)
    # run_inference(lib, origin_shape)

    # print("aipu error rate: ", np.sum(np.abs(out_float - aipu_output )) / np.sum(np.abs(out_float)))
    print("Lenet inferrence done !!!")
