
#
# Inspur.
# This is a new or modified file.
#

import tvm
from tvm import relay
from PIL import Image
# os and numpy
import numpy as np
import os.path
# Tensorflow imports
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing
import argparse
import time
######################################################################
target = 'xpu -libs=xdnn -split-device-funcs'
target_host = 'llvm'
layout = "NCHW"
#layout = None
#ctx = tvm.context(target, 0)
#print('ctx = ', ctx)
#print('type of ctx = ', type(ctx))
'''
target = 'llvm'
target_host = 'llvm'
layout = None
#layout = "NHWC"
layout = "NCHW"
#ctx = tvm.cpu(0)
ctx = tvm.context(target, 0)

target = 'cuda'
target_host = 'llvm'
layout = None
#layout = "NHWC"
layout = "NCHW"
#ctx = tvm.cpu(0)
ctx = tvm.context(target, 0)
'''

######################################################################
# img_path = '/home/wangfan/.tvm_test_data/data/elephant-299.jpg'
# model_path = '/home/wangfan/.tvm_test_data/tf/resnet50v2/resnet50_v2_inf_graph.frozen.pb'

def preprocess_image_rgb(
    image_path,
    resize_h, resize_w,
    need_scale, image_scale,
    red_bias, green_bias, blue_bias):
    # 1, resize the image
    img = Image.open(image_path).resize((resize_h, resize_w), Image.ANTIALIAS)
    image = np.asarray(img).astype("float32")
    # 2, NHWC format
    img_nhwc = np.expand_dims(image, axis = 0)
    if need_scale:
        img_nhwc[:,:,:,0] = image_scale * img_nhwc[:,:,:,0] + red_bias
        img_nhwc[:,:,:,1] = image_scale * img_nhwc[:,:,:,1] + green_bias
        img_nhwc[:,:,:,2] = image_scale * img_nhwc[:,:,:,2] + blue_bias
    # 3, NCHW format
    img_nchw = img_nhwc.transpose((0,3,1,2))
    return img_nhwc, img_nchw

parser = argparse.ArgumentParser()
parser.add_argument('--pb-path', '-p', type=str, default='',
                    help="The tensorflow pb file path")
parser.add_argument('--image-path', type=str, default='',
                     help="The input data. If not set, will use dnn-root-path's cat image")
args = parser.parse_args()

img_path = '/home/wangfan/.tvm_test_data/data/elephant-299.jpg'
model_path = '/home/wangfan/.tvm_test_data/tf/resnet50v2/resnet50_v2_inf_graph.frozen.pb'
#img_path = args.image_path
#model_path = args.pb_path

img_nhwc, img_tvm = preprocess_image_rgb(
            img_path,
            224, 224,
            True, 2.0 / 255.0,
            -1, -1, -1
            )

######################################################################
# Import model
# ------------
# Creates tensorflow graph definition from protobuf file.

with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    #print(graph_def)
    # Add shapes to the graph.
    with tf.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, 'resnet_v1_50/predictions/Reshape_1')

######################################################################
# Decode image
from PIL import Image
image = Image.open(img_path).resize((224, 224))

x = np.array(img_nhwc) 
#print(x.shape) # shenfw add
#x = np.random.randint(10, size=(1, 224, 224, 3)) # shenfw add
#x = np.array(x.astype(float)) # shenfw add
#print(x.shape) # shenfw add
######################################################################
# Import the graph to Relay
# -------------------------
# Import tensorflow graph definition to relay frontend.
#
# Results:
#   sym: relay expr for given tensorflow protobuf.
#   params: params converted from tensorflow params (tensor protobuf).
shape_dict = {'input__0': x.shape}
#dtype_dict = {'DecodeJpeg/contents': 'uint8'}
mod, params = relay.frontend.from_tensorflow(graph_def,
                                             layout=layout,
                                             shape=shape_dict)
print("Tensorflow protobuf imported to relay frontend.")

# print(mod)

# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
#
# Results:
#   graph: Final graph after compilation.
#   params: final params after compilation.
#   lib: target library which can be deployed on target with TVM runtime.

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod,
                                     target=target,
                                     target_host=target_host,
                                     params=params)
######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.
#print(graph)
from tvm.contrib import graph_runtime
m = graph_runtime.create(graph, lib)
# set inputs
m.set_input('input', tvm.nd.array(x.astype("float32")))
m.set_input(**params)
m.run()
#print('ctx = ', ctx)
print('lib = ', lib)
#print('graph = ', graph)
tvm_output = m.get_output(0, tvm.nd.empty((1, 1000), 'float32')) # 1, 1000

start = time.time()
for i in range(100):
    #m.set_input('input', tvm.nd.array(x.astype("float32")))
    m.run()
    tvm_output = m.get_output(0, tvm.nd.empty((1, 1000), 'float32'))
end = time.time()
#tvm_output = m.get_output(0, tvm.nd.empty((1, 1000), 'float32')) 
print('the runtime is: ')
print((end-start)*1000/100.)
print("=====================  TVM output ===============================")
predictions = tvm_output.asnumpy()
predictions = np.squeeze(predictions)
xtcl_topk = predictions.argsort()[-5:][::-1]
print(xtcl_topk)

######################################################################
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
    # evaluate TF
    tf.reset_default_graph()
    graph_def = graph_pb2.GraphDef()
    with open(model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
    g = tf.import_graph_def(graph_def)
    with tf.Session(graph=g) as sess:
        image_input_tensor = sess.graph.get_tensor_by_name('import/' + 'input:0')
        outputs = [sess.graph.get_tensor_by_name("import/" + 'resnet_v1_50/predictions/Reshape_1:0')]
        predictions = sess.run(outputs, feed_dict={image_input_tensor: img_nhwc})
        predictions = np.squeeze(predictions)
        top_k = predictions.argsort()[-5:][::-1]
        print("=====================  Tensorflow output ===============================")
        print(top_k)
        return top_k
#begin = time.time()
tf_topk = run_inference_on_image(img_path)
#end = time.time()
#print('tf time is: ', end-begin)
assert np.alltrue(xtcl_topk == tf_topk)
