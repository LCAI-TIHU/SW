
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

import os


data_path = "/workspace/imagenet/val/"
label_path = "/workspace/imagenet/val.txt"
model_path = "/workspace/tfmodel/efficientnet_B5.pb"
layout = "NHWC"
calibration_samples = 10

# Returns a numpy buffer of shape (num_images, 1, 28, 28)
def load_data(data_path):
     
   images = os.listdir(data_path)
   data_img=[]
   length=10
   #length = len(images)
   for i in range(length):
    img = cv2.imread(data_path + images[i])
    img = cv2.resize(img,(456, 456))
    img = img[:,:,::-1] #convert('RGB')
    img = np.array(img, dtype='float32') / 255.0
    data_img.append(img)   
    
   print("------image-shape",len(data_img))

   return data_img

# Returns a numpy buffer of shape (num_images)
def load_label(lablel_path):
    labelMat = []
    fr = open(lablel_path)
    for line in fr.readlines():
        lineArr = line.strip().split()
        labelMat.append(int(lineArr[-1]))
    return labelMat

test_data = load_data(data_path)
label_data = load_label(label_path)

def calibrate_dataset():
    
    calib_data = []
    for data in test_data[:calibration_samples]:
        data = np.expand_dims(data, axis = 0)
        calib_data.append({'images': data})

    return calib_data

def quantize(mod, params,target, data_aware):
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
            graph_def = tf_testing.AddShapesToGraphDef(sess, "Softmax")

    print(f'from_tensorflow...')
    shape_dict = {"images": (1, 456, 456, 3)}
    mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)

    return mod, params

    
def run_test(lib, origin_shape=None):
    
    img = cv2.imread("/workspace/imagenet/val/ILSVRC2012_val_00000001.JPEG")
    img = cv2.resize(img,(456, 456))
    img = img[:,:,::-1] #convert('RGB')

    img_resize = np.array(img, dtype='float32') / 255.0
    image = np.expand_dims(img_resize, 0)

    global target

    m = graph_executor.GraphModule(lib["default"](dev))
    if target == "aipu":
        image = np.round(image / 0.00784248)
        image = np.clip(image, -127, 127)
        
        m.set_input("images", tvm.nd.array(image.astype("int8"))) # set inputs
        m.run() # execute
        tvm_output = m.get_output(0, tvm.nd.empty(origin_shape)).asnumpy() # get outputs
    else:
        image = np.expand_dims(image, axis=0)
        m.set_input("images", tvm.nd.array(image.astype("float32"))) # set inputs
        m.run() # execute
        tvm_output = m.get_output(0).asnumpy() 
    pre = np.argmax(tvm_output) 
    print("Runtime Done!!!")
    print("tvm_output: ", pre)
    
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
            m.set_input(" images", tvm.nd.array(images_batch.astype("int8"))) # set inputs
            m.run() # execute
            tvm_output = m.get_output(0, tvm.nd.empty(origin_shape)).asnumpy() # get outputs
        else:
            images_batch = np.array([x[0] for x in batch]) 
            m.set_input(" images", tvm.nd.array(images_batch.astype("float32"))) # set inputs
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

    # target = "llvm"
    # dev = tvm.device(target, 0)
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build(mod, target, params=params)

    # out_float = run_test(lib)
    # origin_shape = np.array(out_float.shape)
    # print('*'*100, origin_shape)


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
    target = "llvm"
    mod_quantized = quantize(mod, params, target, data_aware=True)
    print("-------------mod_quantized model--------------")
    print(mod_quantized["main"].astext(show_meta_data=False))
    # #### --------------aipu测试 -------------------#######
    # target = "aipu"
    # dev = tvm.device(target, 0)
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build(mod_quantized, target, params=params)

    # aipu_output = run_test(lib, origin_shape)
    # run_inference(lib, origin_shape)
    # print('*'*100,aipu_output)
    #print("aipu error rate: ", np.sum(np.abs(out_float - aipu_output )) / np.sum(np.abs(out_float)))
    print("Efficientnet inferrence done !!!")
