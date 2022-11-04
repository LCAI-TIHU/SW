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
from pytorch_transformers.tokenization_bert import BertTokenizer

import os
import random
import numpy as np
import multiprocessing as mp
import json
import collections

# tvm, relay
import tvm
from tvm import te
from tvm import relay
from tvm.relay.quantize import gen_calibdata

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing
from tvm.contrib import graph_executor

from utils.bert_utils import read_squad_examples, input_to_squad_example, squad_examples_to_features, get_answer

# import pdb
# print(f'Python: {os.getpid()}')
# pdb.set_trace()

predict_file = "/workspace/tfmodel/bert/squad_data/dev-v1.1.json"
model_path = "/workspace/tfmodel/bert_base.pb"

layout = "NHWC"

flags = tf_compat_v1.flags
flags.DEFINE_string("vocab_file", "/workspace/tfmodel/bert/uncased_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_integer(
    "max_seq_length", 384, # 128
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")
flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")
flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")


FLAGS = flags.FLAGS

eval_examples = read_squad_examples(predict_file)
tokenizer = BertTokenizer.from_pretrained(FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
# print(features[0].input_ids)
# print(features[0].input_mask)
# print(features[0].segment_ids)
calibration_samples = 20

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

def calibrate_dataset():
    
    calib_data = []
    # for example in random.sample(eval_examples, calibration_samples):
    for example in eval_examples[:calibration_samples]:
        features = squad_examples_to_features(example, tokenizer, FLAGS.max_seq_length, FLAGS.doc_stride, FLAGS.max_query_length)
        for feature in features:
            input_ids = np.array([feature.input_ids])
            input_mask = np.array([feature.input_mask])
            segment_ids = np.array([feature.segment_ids])
            calib_data.append({'input_ids_1': input_ids, 'input_mask_1':input_mask, 'segment_ids_1':segment_ids})

    return calib_data

def quantize(mod, params, target, data_aware, do_simulation=True):
    if data_aware:
        with relay.quantize.qconfig(target=target, calibrate_mode="percentile", weight_scale="max", skip_conv_layers=[], skip_dense_layer=False, do_simulation=do_simulation): #, dtype_input="uint8", debug_enabled_ops=["nn.conv2d"], calibrate_chunk_by=16
            mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset())
    else:
        with relay.quantize.qconfig(target=target, calibrate_mode="global_scale", global_scale=8.0):
            mod = relay.quantize.quantize(mod, params)
    return mod


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
            graph_def = tf_testing.AddShapesToGraphDef(sess, "unstack") # "unstack" ,"bert/encoder/layer_0/attention/self/MatMul"

    shape_dict = {"input_ids_1": (1, 384), "input_mask_1": (1, 384), "segment_ids_1": (1, 384)}
    print(f'from_tensorflow...')
    mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)

    return mod, params

def run_tf(model_path):
    print(f'create_graph...')
    with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

        doc = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."
        q = "Which NFL team represented the AFC at Super Bowl 50?"
        example = input_to_squad_example(doc, q)
        features = squad_examples_to_features(example, tokenizer, FLAGS.max_seq_length, FLAGS.doc_stride, FLAGS.max_query_length)

        # Add shapes to the graph.
        with tf_compat_v1.Session() as sess:
            out_0, out_1 = sess.run(("unstack:0", "unstack:1"), feed_dict={"input_ids_1:0": np.array([features[0].input_ids]).astype("int32"),
                                                    "input_mask_1:0": np.array([features[0].input_mask]).astype("int32"),
                                                    "segment_ids_1:0": np.array([features[0].segment_ids]).astype("int32")})
            unique_id = int(features[0].unique_id)
            result = RawResult(unique_id    = unique_id,
                                start_logits = out_0[0].tolist(),
                                end_logits   = out_1[0].tolist())   

            print(out_0[0].tolist())
            print(out_1[0].tolist())

            answer = get_answer(example, features, [result], FLAGS.n_best_size, FLAGS.max_answer_length, FLAGS.do_lower_case)
            print(answer['answer']) # "Denver Broncos"


def run_test(lib, origin_shape=None):
    doc = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."
    q = "Which NFL team represented the AFC at Super Bowl 50?"
    example = input_to_squad_example(doc, q)
    features = squad_examples_to_features(example, tokenizer, FLAGS.max_seq_length, FLAGS.doc_stride, FLAGS.max_query_length)
    print(features[0].input_ids)
    print(features[0].input_mask)
    print(features[0].segment_ids)

    input_ids = np.array([features[0].input_ids])
    input_mask = np.array([features[0].input_mask])
    segment_ids = np.array([features[0].segment_ids])

    m = graph_executor.GraphModule(lib["default"](dev))
    m.set_input("input_ids_1", tvm.nd.array(input_ids.astype("int32"))) # set inputs
    m.set_input("input_mask_1", tvm.nd.array(input_mask.astype("int32"))) # set inputs
    m.set_input("segment_ids_1", tvm.nd.array(segment_ids.astype("int32"))) # set inputs
    m.run() # execute
    if target == 'aipu':
        tvm_output_0 = m.get_output(0, tvm.nd.empty(origin_shape)).asnumpy() # get outputs
        tvm_output_1 = m.get_output(1, tvm.nd.empty(origin_shape)).asnumpy() # get outputs
    else:
        tvm_output_0 = m.get_output(0).asnumpy() # get outputs
        tvm_output_1 = m.get_output(1).asnumpy() # get outputs
    print(target, tvm_output_0, tvm_output_0.shape)
    print(target, tvm_output_1, tvm_output_1.shape)

    unique_id = int(features[0].unique_id)
    result = RawResult(unique_id    = unique_id,
                        start_logits = tvm_output_0[0].tolist(),
                        end_logits   = tvm_output_1[0].tolist())   

    answer = get_answer(example, features, [result], FLAGS.n_best_size, FLAGS.max_answer_length, FLAGS.do_lower_case)
    print(answer['answer']) # "Denver Broncos"

    return tvm_output_0


def run_inference(lib, origin_shape=None):
    # eval_examples = read_squad_examples(predict_file)
    # tokenizer = BertTokenizer.from_pretrained(FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    m = graph_executor.GraphModule(lib["default"](dev))
    all_result = {}
    # print(len(eval_examples))
    for i, example in enumerate(eval_examples): # [:100]
        features = squad_examples_to_features(example, tokenizer, FLAGS.max_seq_length, FLAGS.doc_stride, FLAGS.max_query_length)
        result = []
        for feature in features:
            input_ids = np.array([feature.input_ids])
            input_mask = np.array([feature.input_mask])
            segment_ids = np.array([feature.segment_ids])

            m.set_input("input_ids_1", tvm.nd.array(input_ids.astype("int32"))) # set inputs
            m.set_input("input_mask_1", tvm.nd.array(input_mask.astype("int32"))) # set inputs
            m.set_input("segment_ids_1", tvm.nd.array(segment_ids.astype("int32"))) # set inputs
            if target == 'aipu':
                tvm_output_0 = m.get_output(0, tvm.nd.empty(origin_shape)).asnumpy() # get outputs
                tvm_output_1 = m.get_output(1, tvm.nd.empty(origin_shape)).asnumpy() # get outputs
            else:
                tvm_output_0 = m.get_output(0).asnumpy() # get outputs
                tvm_output_1 = m.get_output(1).asnumpy() # get outputs

            unique_id = int(feature.unique_id)
            result.append(RawResult(unique_id    = unique_id,
                                start_logits = tvm_output_0[0].tolist(),
                                end_logits   = tvm_output_1[0].tolist()))

        answer = get_answer(example, features, result, FLAGS.n_best_size, FLAGS.max_answer_length, FLAGS.do_lower_case)
        all_result.update({example.qas_id : answer['answer']})
        # print(answer['answer'])
        print(i)

    json_str = json.dumps(all_result, indent=4)
    with open('predictions.json', 'w') as fp:
        fp.write(json_str)

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

    # mod_quantized = quantize(mod, params, data_aware=True, do_simulation=False)
    # print("-------------mod_quantized model--------------")
    # print(mod_quantized["main"].astext(show_meta_data=False))
    # target = "llvm"
    # dev = tvm.device(target, 0)
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build(mod_quantized, target, params=params)

    # out_tvm_int8 = run_test(lib)

    # mod_quantized_path = 'mod_quantized_bert.json'
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

    # mod_quantized = quantize(mod, params, data_aware=True, do_simulation=False)
    # print("-------------mod_quantized model--------------")
    # print(mod_quantized["main"].astext(show_meta_data=False))
    target="aipu"
    mod_quantized = quantize(mod, params, target, data_aware=True, do_simulation=True)
    print("-------------mod_quantized model--------------")
    print(mod_quantized["main"].astext(show_meta_data=False))

    # target = "aipu"
    # dev = tvm.device(target, 0)
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build(mod_quantized, target, params=params)

    # out_aipu_int8 = run_test(lib, origin_shape)
    # print('AIPU error rate: ', np.sum(np.abs(out_float - out_aipu_int8)) / np.sum(np.abs(out_float)))
    # print('tvm error rate: ', np.sum(np.abs(out_float - out_tvm_int8)) / np.sum(np.abs(out_float)))
    print('DONE')
