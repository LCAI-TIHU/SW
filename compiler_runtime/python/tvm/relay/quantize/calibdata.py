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
"""Find scales for quantization on the dataset."""
from __future__ import absolute_import
import logging
import json
import numpy as np

from tvm.relay import expr as _expr, analysis as _analysis


def gen_calibdata(mod, save_path):
    '''
    Parameters
    ----------
    mod: Module
        The simulation graph after calibrate.
    '''
    logging.info("import TVM's calibdata to NVDLA's...")

    const_params = []
    def visit_func(expr):
        """visitor function for traverse"""
        if isinstance(expr, _expr.Call) and expr.op.name in ["relay.op.annotation.simulated_quantize"]:
            if isinstance(expr.args[0], _expr.Call) and ((expr.args[0].op.name == "nn.pad" and isinstance(expr.args[0].args[0], _expr.Call)) \
                or expr.args[0].op.name in ["reshape", "mean", "annotation.stop_fusion"]):
                return
            scale = np.atleast_1d(expr.args[1].data.asnumpy())[0]
            const_params.append({'scale': float('%.8f' % scale)})
        elif isinstance(expr, _expr.Call) and expr.op.name == "mean":
            const_params.append({'scale': const_params[-1]['scale']})

    _analysis.post_order_visit(mod['main'], visit_func)

    calibdata_ = {}
    for i, value in enumerate(const_params):
        calibdata_[i] = value
    json_str = json.dumps(calibdata_, indent=4)
    with open(save_path, 'w') as fp:
        fp.write(json_str)
