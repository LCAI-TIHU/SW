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
import multiprocessing as mp
import json
from tqdm import tqdm

# tvm, relay
import tvm
from tvm import relay

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing
from tvm.contrib import graph_executor
from tvm.relay.quantize import gen_calibdata

from pycocotools.coco import COCO
from utils.yolo_preprocessing import preprocess_image, postprocess_boxes, nms, draw_bbox, \
        coco80_to_coco91_class, save_one_json, metrics

import pdb
print(f'Python: {os.getpid()}')
# pdb.set_trace()

annFile = "/workspace/coco/annotations/instances_val2017.json"
image_base = "/workspace/coco/images/val2017"
model_path = "/workspace/tfmodel/yolov3_coco_x.pb"
layout = "NHWC"
batch_size = 1
calibration_samples = 10

with open(annFile, 'r') as load_f:
    annotations = json.load(load_f)

###############################################################################
# The calibration dataset should be an iterable object. We define the
# calibration dataset as a generator object in Python. In this tutorial, we
# only use a few samples for calibration.


def calibrate_dataset():
    calib_data = []
    for i in range(calibration_samples):
        data = gen_data(i)
        data = np.expand_dims(data, axis = 0)
        calib_data.append({'input': data})

    return calib_data

def quantize(mod, params, target, data_aware, do_simulation=True):
    if data_aware:
        with relay.quantize.qconfig(target=target, calibrate_mode="percentile", weight_scale="max", skip_conv_layers=[], do_simulation=do_simulation): #, dtype_input="uint8", debug_enabled_ops=["nn.conv2d"], calibrate_chunk_by=16
            mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset())
    else:
        with relay.quantize.qconfig(target=target, calibrate_mode="global_scale", global_scale=8.0):
            mod = relay.quantize.quantize(mod, params)
    return mod


def gen_data(i):
    file_name = annotations['images'][i]['file_name']
    image_path = os.path.join(image_base, file_name)

    image = cv2.imread(image_path)
    image = preprocess_image(image, [416, 416])

    return image

coco = COCO(annFile)
def gen_data_val(i):
    file_name = annotations['images'][i]['file_name']
    image_path = os.path.join(image_base, file_name)

    original_image = cv2.imread(image_path)
    origin_shape = original_image.shape[:2]

    imgIds = annotations['images'][i]['id']
    image = preprocess_image(original_image, [416, 416])

    return original_image, image, imgIds, origin_shape


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
        # with tf_compat_v1.Session() as sess:
        #   graph_def = tf_testing.AddShapesToGraphDef(sess, "pred_sbbox/concat_6")

    data = gen_data(0)
    data = np.expand_dims(data, axis = 0)
    shape_dict = {"input": data.shape}

    print(f'from_tensorflow...')
    mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict, outputs=["pred_sbbox/concat_6", "pred_mbbox/concat_6", "pred_lbbox/concat_6"])

    return mod, params


def run_test(lib):
    original_image = cv2.imread('/workspace/testimage/yolov3_test.jpg')
    # original_image = cv2.resize(original_image, (224,224))
    image = original_image / 255.0
    image = image[np.newaxis, ...]

    m = graph_executor.GraphModule(lib["default"](dev))
    global target
    if target == "aipu":
        image = np.round(image / 0.00787402)
        image = np.clip(image, -127, 127)
        m.set_input("input", tvm.nd.array(image.astype("int8"))) # set inputs
        m.run() # execute
        tvm_output_sbbox = m.get_output(0, tvm.nd.empty((1,52,52,255))).asnumpy()
        tvm_output_mbbox = m.get_output(1, tvm.nd.empty((1,26,26,255))).asnumpy()
        tvm_output_lbbox = m.get_output(2, tvm.nd.empty((1,13,13,255))).asnumpy()
    else:
        m.set_input("input", tvm.nd.array(image.astype("float32"))) # set inputs
        m.run() # execute
        tvm_output_sbbox = m.get_output(0).asnumpy()
        tvm_output_mbbox = m.get_output(1).asnumpy()
        tvm_output_lbbox = m.get_output(2).asnumpy()

    tvm_output_sbbox_0 = tvm_output_sbbox[:,:,:,:12]
    tvm_output_sbbox_0 = np.reshape(tvm_output_sbbox_0, (1,52,52,3,-1))
    tvm_output_sbbox_1 = tvm_output_sbbox[:,:,:,12:15]
    tvm_output_sbbox_1 = np.reshape(tvm_output_sbbox_1, (1,52,52,3,-1))
    tvm_output_sbbox_2 = tvm_output_sbbox[:,:,:,15:]
    tvm_output_sbbox_2 = np.reshape(tvm_output_sbbox_2, (1,52,52,3,-1))
    tvm_output_sbbox = np.concatenate([tvm_output_sbbox_0, tvm_output_sbbox_1, tvm_output_sbbox_2], axis=-1)

    tvm_output_mbbox_0 = tvm_output_mbbox[:,:,:,:12]
    tvm_output_mbbox_0 = np.reshape(tvm_output_mbbox_0, (1,26,26,3,-1))
    tvm_output_mbbox_1 = tvm_output_mbbox[:,:,:,12:15]
    tvm_output_mbbox_1 = np.reshape(tvm_output_mbbox_1, (1,26,26,3,-1))
    tvm_output_mbbox_2 = tvm_output_mbbox[:,:,:,15:]
    tvm_output_mbbox_2 = np.reshape(tvm_output_mbbox_2, (1,26,26,3,-1))
    tvm_output_mbbox = np.concatenate([tvm_output_mbbox_0, tvm_output_mbbox_1, tvm_output_mbbox_2], axis=-1)

    tvm_output_lbbox_0 = tvm_output_lbbox[:,:,:,:12]
    tvm_output_lbbox_0 = np.reshape(tvm_output_lbbox_0, (1,13,13,3,-1))
    tvm_output_lbbox_1 = tvm_output_lbbox[:,:,:,12:15]
    tvm_output_lbbox_1 = np.reshape(tvm_output_lbbox_1, (1,13,13,3,-1))
    tvm_output_lbbox_2 = tvm_output_lbbox[:,:,:,15:]
    tvm_output_lbbox_2 = np.reshape(tvm_output_lbbox_2, (1,13,13,3,-1))
    tvm_output_lbbox = np.concatenate([tvm_output_lbbox_0, tvm_output_lbbox_1, tvm_output_lbbox_2], axis=-1)

    pred_bbox = np.concatenate([np.reshape(tvm_output_sbbox, (-1, 5 + 80)),
                                np.reshape(tvm_output_mbbox, (-1, 5 + 80)),
                                np.reshape(tvm_output_lbbox, (-1, 5 + 80))], axis=0)

    original_image_size = original_image.shape[:2]
    bboxes = postprocess_boxes(pred_bbox, original_image_size, 416, 0.3)
    bboxes = nms(bboxes, 0.45, method='nms')
    draw_image = draw_bbox(original_image, bboxes)
    cv2.imwrite('yolov3_demo_aipu.jpg', draw_image)

    print("Done!")

def run_inference(lib):
    pred_json = 'yolov3_AIPU_predictions.json'  # predictions json
    class_map = coco80_to_coco91_class() 
    jdict = []
    total = len(annotations['images'])
    m = graph_executor.GraphModule(lib["default"](dev))
    global target
    for start_idx in tqdm(range(0,100,1)):
        original_image, image, imgIds, origin_shape = gen_data_val(start_idx)
        image = image[np.newaxis, ...]

        if target == "aipu":
            image = np.round(image / 0.00787402)
            image = np.clip(image, -127, 127)
            m.set_input("input", tvm.nd.array(image.astype("int8"))) # set inputs
            m.run() # execute
            tvm_output_sbbox = m.get_output(0, tvm.nd.empty((1,52,52,255))).asnumpy()
            tvm_output_mbbox = m.get_output(1, tvm.nd.empty((1,26,26,255))).asnumpy()
            tvm_output_lbbox = m.get_output(2, tvm.nd.empty((1,13,13,255))).asnumpy()
        else:
            m.set_input("input", tvm.nd.array(image.astype("float32"))) # set inputs
            m.run() # execute    
            tvm_output_sbbox = m.get_output(0).asnumpy()
            tvm_output_mbbox = m.get_output(1).asnumpy()
            tvm_output_lbbox = m.get_output(2).asnumpy()

        tvm_output_sbbox_0 = tvm_output_sbbox[:,:,:,:12]
        tvm_output_sbbox_0 = np.reshape(tvm_output_sbbox_0, (1,52,52,3,-1))
        tvm_output_sbbox_1 = tvm_output_sbbox[:,:,:,12:15]
        tvm_output_sbbox_1 = np.reshape(tvm_output_sbbox_1, (1,52,52,3,-1))
        tvm_output_sbbox_2 = tvm_output_sbbox[:,:,:,15:]
        tvm_output_sbbox_2 = np.reshape(tvm_output_sbbox_2, (1,52,52,3,-1))
        tvm_output_sbbox = np.concatenate([tvm_output_sbbox_0, tvm_output_sbbox_1, tvm_output_sbbox_2], axis=-1)

        tvm_output_mbbox_0 = tvm_output_mbbox[:,:,:,:12]
        tvm_output_mbbox_0 = np.reshape(tvm_output_mbbox_0, (1,26,26,3,-1))
        tvm_output_mbbox_1 = tvm_output_mbbox[:,:,:,12:15]
        tvm_output_mbbox_1 = np.reshape(tvm_output_mbbox_1, (1,26,26,3,-1))
        tvm_output_mbbox_2 = tvm_output_mbbox[:,:,:,15:]
        tvm_output_mbbox_2 = np.reshape(tvm_output_mbbox_2, (1,26,26,3,-1))
        tvm_output_mbbox = np.concatenate([tvm_output_mbbox_0, tvm_output_mbbox_1, tvm_output_mbbox_2], axis=-1)

        tvm_output_lbbox_0 = tvm_output_lbbox[:,:,:,:12]
        tvm_output_lbbox_0 = np.reshape(tvm_output_lbbox_0, (1,13,13,3,-1))
        tvm_output_lbbox_1 = tvm_output_lbbox[:,:,:,12:15]
        tvm_output_lbbox_1 = np.reshape(tvm_output_lbbox_1, (1,13,13,3,-1))
        tvm_output_lbbox_2 = tvm_output_lbbox[:,:,:,15:]
        tvm_output_lbbox_2 = np.reshape(tvm_output_lbbox_2, (1,13,13,3,-1))
        tvm_output_lbbox = np.concatenate([tvm_output_lbbox_0, tvm_output_lbbox_1, tvm_output_lbbox_2], axis=-1)
        #pred_bbox = np.reshape(tvm_output_sbbox, (-1, 5 + 80))

        pred_bbox = np.concatenate([np.reshape(tvm_output_sbbox, (-1, 5 + 80)),
                                    np.reshape(tvm_output_mbbox, (-1, 5 + 80)),
                                    np.reshape(tvm_output_lbbox, (-1, 5 + 80))], axis=0)
        
        predn = postprocess_boxes(pred_bbox, origin_shape, 416, 0.3)
        predn = nms(predn, 0.45, method='nms')
        print('predn: ', len(predn), origin_shape)
        # if len(predn) == 0:
            # continue
        if start_idx % 1 == 0:
            draw_image = draw_bbox(original_image, predn)
            cv2.imwrite('yolov3_demo_AIPU'+str(start_idx)+'.jpg', draw_image)

        save_one_json(np.array(predn, dtype=np.float), jdict, imgIds, class_map)
    metrics(jdict, annFile, pred_json)

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
    
    # mod_quantized_path = 'mod_quantized_yolov3_{}.json'.format(calibration_samples)
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
    mod_quantized = quantize(mod, params, target, data_aware=True, do_simulation=True)
    print("-------------mod_quantized model--------------")
    print(mod_quantized["main"].astext(show_meta_data=False))

    target = "aipu"
    dev = tvm.device(target, 0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod_quantized, target, params=params)

    run_test(lib)
    # run_inference(lib)

