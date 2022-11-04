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

#
# Inspur.
# This is a new or modified file.
#

"""
Compile Tensorflow Models
=========================
This article is an introductory tutorial to deploy tensorflow models with TVM.

For us to begin with, tensorflow python module is required to be installed.

Please refer to https://www.tensorflow.org/install
"""

'''
import pdb
import os
print(os.getpid())
pdb.set_trace()
'''

# tvm, relay
from numpy.core.multiarray import datetime_as_string
#import tvm
#from tvm import te
#from tvm import relay
# os and numpy
import numpy as np
import os.path
import os
from datasets import download_and_convert_mnist
from lenet_preprocessing import preprocess_image 

# Tensorflow imports
import tensorflow as tf
import tvm
from tvm import relay
tf_compat_v1 = tf
try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing



batch_size = 1

label_offset = 0

target = "llvm"#"llvm"
target_host = "llvm"
#target = tvm.target.Target("llvm", host="llvm")
layout = None
#layout = "NCHW"
dev = tvm.cpu(0)

######################################################################
# Import model
# ------------
# Creates tensorflow graph definition from protobuf file.
model_path='./frozen_lenet_2.pb'
precision='int8' #'int16'
if precision=='int8':
    calibratable = './lenet.json'

with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
    graph_def = tf_compat_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name="")
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    with tf_compat_v1.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, "Predictions/Reshape_1")

######################################################################
# Download and Decode mnist image
# ------------
# .. note::
#
#   tensorflow frontend import doesn't support preprocessing ops like JpegDecode.
#   JpegDecode is bypassed (just return source node).
#   Hence we supply decoded frame to TVM instead.
#

dataset_dir = "MNIST"
_TEST_DATA_FILENAME = 't10k-images-idx3-ubyte.gz'
_TEST_LABELS_FILENAME = 't10k-labels-idx1-ubyte.gz'
data_filename = os.path.join(dataset_dir, _TEST_DATA_FILENAME)
labels_filename = os.path.join(dataset_dir, _TEST_LABELS_FILENAME)
num_images = 10000

if os.path.exists(dataset_dir):
    print('the filepath ',dataset_dir, ' is existed.')
else:
    os.mkdir(dataset_dir)
    print('the filepath ', dataset_dir, 'is created.')

download_and_convert_mnist._download_dataset(dataset_dir)
images = download_and_convert_mnist._extract_images(data_filename, num_images)
labels = download_and_convert_mnist._extract_labels(labels_filename, num_images)

######################################################################
# Import the graph to Relay
# -------------------------
# Import tensorflow graph definition to relay frontend.
#
# Results:
#   sym: relay expr for given tensorflow protobuf.
#   params: params converted from tensorflow params (tensor protobuf).
shape_dict = {"input": (1, 28, 28, 1)}
print(shape_dict)
dtype_dict = {"input": "float32"}
mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)

print("Tensorflow protobuf imported to relay frontend.")
######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
#
# Results:
#   graph: Final graph after compilation.
#   params: final params after compilation.
#   lib: target library which can be deployed on target with TVM runtime.

#print("-------------original model--------------")
#print(mod["main"].astext(show_meta_data=False))
##mod = quantize(mod, params, data_aware=True)
#mod = quantize(mod, params, data_aware=False)
#print("-------------quantized model--------------")
#print(mod.astext(show_meta_data=False))


print("python before relay.build ...")
with tvm.transform.PassContext(opt_level=3):
    #graph, lib, params = relay.build(mod, target, params=params)
    lib = relay.build(mod, target, params=params)

#f = open("mob2_graph.json", 'w')
#print(graph,file=f)
print("python after relay.build ...")
def run_inference_tf(mod):
    ######################################################################
    # Execute the portable graph on TVM
    # ---------------------------------
    # Now we can try deploying the compiled model on target.

    from tvm.contrib import graph_executor

    dtype = "float32"
    output_height = 28
    output_width = 28

    top1_cnt = 0
    total = 10
    #m = tvm.contrib.graph_executor.create(graph, lib, dev)
    #m.set_input(**params)
    m = graph_executor.GraphModule(lib["default"](dev))
    for i in range(total):
        image = preprocess_image(images[i], output_height, output_width, is_training=False)
        #print(type(image))
        #session = tf.Session() # tf1 api
        #data = session.run(image) # tf1 api
        data = image.numpy() # tf2 api
        #print(type(data))
        data = np.expand_dims(data, axis = 0)
        label = labels[i]
        print("python before graph_executor.GraphModule")
        # set inputs
        print("python before set_input")
        m.set_input("input", tvm.nd.array(data.astype(dtype)))
        # execute
        print("python before run")
        m.run()
        # get outputs
        tvm_output = m.get_output(0)
        pred = np.argmax(tvm_output.asnumpy())

        ######################################################################
        # Process the output
        # ------------------
        # Process the model output to human readable text for InceptionV1.

        #print(tvm_output.asnumpy())
        #if i%100 == 0:
        print(i, pred, label, [pred == label])
        if pred == label:
            top1_cnt = top1_cnt + 1

    print("accuracy = %f" % (top1_cnt / total))

    '''
    # Creates node ID --> English string lookup.
    node_lookup = tf_testing.NodeLookup(label_lookup_path=map_proto_path, uid_lookup_path=label_path)

    # Print top 5 predictions from TVM output.
    top_k = predictions.argsort()[-5:][::-1]
    for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        score = predictions[node_id]
        print("%s (score = %.5f)" % (human_string, score))
    '''


run_inference_tf(mod)
######################################################################
# Inference on tensorflow
# -----------------------
# Run the corresponding model on tensorflow


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name="")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)


def run_inference_on_image(image):
    """Runs inference on an image.

    Parameters
    ----------
    image: String
        Image file name.

    Returns
    -------
        Nothing
    """
    dtype = "float32"

    image_label_path = "/workspace/imagenet/val.txt"
    image_label = open(image_label_path)
    image_base = "/workspace/imagenet/"

    output_height = 224
    output_width = 224

    top1_cnt = 0
    total = 100
    for i in range(total):
        line = image_label.readline()
        line_table = line.split()
        image_path = os.path.join(image_base, line_table[0])
        image_file = tf_compat_v1.gfile.FastGFile(image_path, 'rb')
        image_raw_data = image_file.read()
        image_file.close()
        
        image = tf.image.decode_jpeg(image_raw_data, channels = 3)
        image = preprocess_image(image, output_height, output_width, is_training=False)

        data = tf.keras.preprocessing.image.img_to_array(image)
        data = np.expand_dims(data, axis = 0)
        label = int(line_table[1]) + label_offset

        # Creates graph from saved GraphDef.
        create_graph()

        with tf_compat_v1.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name("MobilenetV2/Predictions/Reshape_1:0")
            predictions = sess.run(softmax_tensor, {"input:0": data})

            predictions = np.squeeze(predictions)

            pred = np.argmax(predictions)

            print(i, pred, label, [pred == label])
            if pred == label:
                top1_cnt = top1_cnt + 1

    print("accuracy = %f" % (top1_cnt / total))

    image_label.close()


def run_inference_on_image_2(image):
    """Runs inference on an image.

    Parameters
    ----------
    image: String
        Image file name.

    Returns
    -------
        Nothing
    """
    dtype = "float32"

    image_label_path = "/workspace/imagenet/val.txt"
    image_label = open(image_label_path)
    image_base = "/workspace/imagenet/"

    output_height = 224
    output_width = 224

    top1_cnt = 0
    total = 100
    for i in range(total):
        line = image_label.readline()
        line_table = line.split()
        image_path = os.path.join(image_base, line_table[0])

        """image_file = tf_compat_v1.gfile.FastGFile(image_path, 'rb')
        image_raw_data = image_file.read()
        image_file.close()

        image = tf.image.decode_jpeg(image_raw_data, channels = 3)
        image = preprocess_image(image, output_height, output_width, is_training=False)

        data = tf.keras.preprocessing.image.img_to_array(image)
        data = np.expand_dims(data, axis = 0)
        #print(type(data),' ',data.shape)"""

       
        ##resized_image = Image.open(image_path).resize((224, 224),Image.BILINEAR) # shenfw add
        ##if resized_image.mode != 'RGB':
        ##    resized_image = resized_image.convert("RGB")
        image_data = np.asarray(resized_image).astype("float32")
        image_data = np.expand_dims(image_data, axis=0)
        resized_image = Image.open(image_path).resize((224, 224),Image.BILINEAR) # shenfw add
        if resized_image.mode != 'RGB':
            resized_image = resized_image.convert("RGB")
        ##image_data = np.expand_dims(image_data, axis=0)
        ##print(image_data.shape)
        image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
        image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
        image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
        
        label = int(line_table[1]) + label_offset

        # Creates graph from saved GraphDef.
        create_graph()

        with tf_compat_v1.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name("MobilenetV2/Predictions/Reshape_1:0")
            #predictions = sess.run(softmax_tensor, {"input:0": image_data})
            predictions = sess.run(softmax_tensor, {"input:0": data})

            predictions = np.squeeze(predictions)

            pred = np.argmax(predictions)

            print(i, pred, label, [pred == label])
            if pred == label:
                top1_cnt = top1_cnt + 1

    print("accuracy = %f" % (top1_cnt / total))

    image_label.close()


#run_inference_on_image(img_path)
#run_inference_on_image_2(img_path)
