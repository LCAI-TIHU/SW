import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import graph_util

input_checkpoint = "/path/model_file"
output_pb = "/path/frozen_model_file"
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    saver.restore(sess, input_checkpoint)
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
    
    output_name = ['output_name']
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_name)
    with tf.gfile.FastGFile(output_pb, 'wb') as f:
        f.write(constant_graph.SerializeToString())

