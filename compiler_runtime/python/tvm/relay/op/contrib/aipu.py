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
# pylint: disable=invalid-name, unused-argument

#
# Inspur.
# This is a new or modified file.
#

"""aipu Compute Library supported operators."""
import tvm
from tvm import relay
from tvm._ffi import register_func
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.expr import const

from ...dataflow_pattern import is_constant, is_expr, is_op, wildcard, is_var
from ..strategy.generic import is_depthwise_conv2d
from .register import register_pattern_table

from . import _ffi_api
from tvm.runtime import Object

@tvm._ffi.register_object("relay.ext.aipu.PatterTableResult")
class PatterTableResult(Object):
    """Store the result of a build.

    Parameters
    ----------
    filename : Optional[str]
        The filename of built binary file.
    args : List[Tensor]
        The arguments.
    error_no : int
        The error code.
    error_msg : Optional[str]
        The error message if there is any error.
    time_cost : float
        The time cost of build.
    """
    def __init__(self, pattern_names, patterns,checkfuncname):
        self.__init_handle_by_constructor__(
            _ffi_api.PatterTableResult, pattern_names,patterns,checkfuncname)

@register_func("relay.ext.aipu.patterntable")
def aipu_compute_lib_pattern_table():
    """Get the ACL pattern table."""

    def conv_pattern(with_bias=True):
        """Create a convolution pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern. 
        """
        data = wildcard()
        weight = is_constant()
        bias = is_constant()
        pattern = is_op("nn.pad")(wildcard(), wildcard()) 
        pattern1 = is_op("relay.op.annotation.simulated_quantize")(pattern, wildcard(), wildcard(), wildcard())
        conv = is_op("nn.conv2d")(pattern1, weight) | is_op("nn.conv2d")(pattern, weight) | is_op("nn.conv2d")(data, weight)
        if with_bias:
            conv_out = is_op("add")(conv, bias)
        else:
            conv_out = conv
        return conv_out
        '''pattern = is_op("nn.pad")(wildcard(), wildcard()) | wildcard()
        pattern = is_op("nn.conv2d")(pattern, wildcard())
        #pattern = is_op("nn.conv2d")(wildcard(), is_constant())
        pattern = pattern.optional(lambda x: is_op("add")(x, is_constant()))
        #pattern = pattern.optional(is_op("nn.relu"))
        return pattern'''

    def qnn_conv_pattern():
        """Create a quantized convolution pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("nn.pad")(wildcard(), wildcard()) | wildcard()
        pattern = is_op("qnn.conv2d")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        pattern = pattern.optional(is_op("nn.relu"))
        pattern = is_op("qnn.requantize")(
            pattern, wildcard(), wildcard(), is_constant(), is_constant()
        )
        return pattern

    def dense_pattern():
        """Create a dense (fully-connected) pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("nn.dense")(wildcard(), is_constant())
        pattern = is_op("add")(pattern, is_constant())
        return pattern

    def dense_pattern_extended():
        """Create a dense (fully-connected) pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("nn.dense")(wildcard(), is_var())
        pattern = is_op("add")(pattern, is_constant())
        return pattern

    def qnn_dense_pattern():
        """Create a quantized dense (fully-connected) pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("qnn.dense")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        pattern = is_op("qnn.requantize")(
            pattern, wildcard(), wildcard(), is_constant(), is_constant()
        )
        return pattern

    def avg_pool2d_pattern():
        """Creates a pattern that matches either quantized
        avg_pool2d or quantized global_avg_pool2d.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("cast")(wildcard())
        pattern = is_op("nn.avg_pool2d")(pattern) | is_op("nn.global_avg_pool2d")(pattern)
        pattern = is_op("cast")(pattern)
        return pattern

    def l2_pool2d_pattern():
        """Create an l2 pooling pattern from equivalent relay operators.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("power")(wildcard(), is_expr(const(2.0)))
        pattern = is_op("nn.avg_pool2d")(pattern)
        pattern = is_op("sqrt")(pattern)
        return pattern

    

    #return [
        #("aipu.conv2d", conv_pattern(), check_conv),
        #("aipu.qnn_conv2d", qnn_conv_pattern(), check_qnn_conv),
        #("aipu.dense", dense_pattern(), check_dense),
        #("aipu.qnn_dense", qnn_dense_pattern(), check_qnn_dense),
        #("aipu.avg_pool2d", avg_pool2d_pattern(), check_avg_pool2d),
        #("aipu.l2_pool2d", l2_pool2d_pattern(), check_l2_pool2d),
    #]
    #return [("aipu.conv2d", conv_pattern(),check_conv),("aipu.qnn_conv2d", qnn_conv_pattern(), check_qnn_conv)]
    patterTable_result = []
    res = (("aipu.conv2d_add","aipu.qnn_conv2d","aipu.dense_add","aipu.qnn_dense","aipu.avg_pool2d","aipu.l2_pool2d",
            "aipu.dense_add_extended",),
           (conv_pattern(with_bias=True),
           qnn_conv_pattern(),dense_pattern(),qnn_dense_pattern(),avg_pool2d_pattern(),l2_pool2d_pattern(),
            dense_pattern_extended(), ),
           ("relay.ext.aipu.check_conv",
            "relay.ext.aipu.check_qnn_conv",
            "relay.ext.aipu.check_dense",
            "relay.ext.aipu.check_qnn_dense",
            "relay.ext.aipu.check_avg_pool2d",
            "relay.ext.aipu.check_l2_pool2d",
            "relay.ext.aipu.check_dense",),)
    '''res = (("aipu.conv2d",),
           (conv_pattern(),),
           ("relay.ext.aipu.check_conv",),)'''
    patterTable_result.append(PatterTableResult(*res))
    return patterTable_result
    

@register_func("relay.ext.aipu.check_conv")
def check_conv(extract):
    """Check conv pattern is supported by ACL."""
    call = extract
    while call.op.name != "nn.conv2d":
        call = call.args[0]
    return conv2d(call)

@register_func("relay.ext.aipu.check_qnn_conv")
def check_qnn_conv(extract):
    """Check qnn conv pattern is supported by ACL."""
    if extract.attrs.out_dtype != "uint8":
        return False
    call = extract
    while call.op.name != "qnn.conv2d":
        call = call.args[0]
    return qnn_conv2d(call)

@register_func("relay.ext.aipu.check_dense")
def check_dense(extract):
    """Check conv pattern is supported by ACL."""
    call = extract
    while call.op.name != "nn.dense":
        call = call.args[0]
    return dense(call)

@register_func("relay.ext.aipu.check_qnn_dense")
def check_qnn_dense(extract):
    """Check qnn conv pattern is supported by ACL."""
    if extract.attrs.out_dtype != "uint8":
        return False
    call = extract
    while call.op.name != "qnn.dense":
        call = call.args[0]
    return qnn_dense(call)

@register_func("relay.ext.aipu.check_avg_pool2d")
def check_avg_pool2d(extract):
    """Check average pool2d pattern is supported by ACL."""
    if extract.attrs.dtype != "uint8":
        return False
    pool = extract.args[0]
    if pool.args[0].attrs.dtype != "int32":
        return False
    return avg_pool2d(pool, from_quantized_composite=True)

@register_func("relay.ext.aipu.check_l2_pool2d")
def check_l2_pool2d(extract):
    """Check l2 pool2d pattern is supported by ACL."""
    pool = extract.args[0]
    return avg_pool2d(pool)


def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.aipu")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


_register_external_op_helper("reshape")


@tvm.ir.register_op_attr("nn.conv2d", "target.aipu")
def conv2d(expr):
    """Check if the external ACL codegen for conv2d should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.data_layout != "NHWC":
        return False
    if attrs.out_dtype != "float32" and attrs.out_dtype != "":
        return False
    data_typ = args[0].checked_type
    if len(data_typ.shape) != 4 or data_typ.shape[0] != 1 or data_typ.dtype != "float32":
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 4 or kernel_typ.dtype != "float32":
        return False
    is_depthwise = is_depthwise_conv2d(
        data_typ.shape,
        attrs["data_layout"],
        kernel_typ.shape,
        attrs["kernel_layout"],
        attrs["groups"],
    )
    if is_depthwise:
        return depthwise_conv2d(attrs, args)
    # ACL doesn't support grouped convolution
    if attrs.groups != 1 and not is_depthwise:
        return False
    return True


def qnn_conv2d(expr):
    """Check if the external ACL codegen for qnn.conv2d should be used."""
    attrs, args = expr.attrs, expr.args

    if attrs.data_layout != "NHWC":
        return False
    if attrs.out_dtype != "int32" and attrs.out_dtype != "":
        return False
    data_typ = args[0].checked_type
    if len(data_typ.shape) != 4 or data_typ.shape[0] != 1 or data_typ.dtype != "uint8":
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 4 or kernel_typ.dtype != "uint8":
        return False
    is_depthwise = is_depthwise_conv2d(
        data_typ.shape,
        attrs["data_layout"],
        kernel_typ.shape,
        attrs["kernel_layout"],
        attrs["groups"],
    )
    if is_depthwise:
        return depthwise_conv2d(attrs, args)
    # ACL doesn't support grouped convolution
    if attrs.groups != 1 and not is_depthwise:
        return False
    return True


def depthwise_conv2d(attrs, args):
    """Check if the external ACL codegen for depthwise convolution should be used.

    Note
    ----
    Relay does not have a depthwise conv2d operator whilst ACL does. We simply
    separate the checks for depthwise for clarity.
    """
    kernel_typ = args[1].checked_type
    # Only supports 3x3, 5x5 depthwise
    if (
        kernel_typ.shape[0] not in [3, 5]
        or kernel_typ.shape[1] not in [3, 5]
        or kernel_typ.shape[0] != kernel_typ.shape[1]
    ):
        return False
    # Stride must be (1, 1) or (2, 2)
    if (attrs.strides[0], attrs.strides[1]) not in [(1, 1), (2, 2)]:
        return False
    return True


@tvm.ir.register_op_attr("nn.dense", "target.aipu")
def dense(expr):
    """Check if the external ACL codegen for dense should be used."""
    attrs, args = expr.attrs, expr.args
    data_typ = args[0].checked_type
    if data_typ.dtype != "float32":
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 2 or kernel_typ.dtype != "float32":
        return False
    if attrs.out_dtype != "float32" and attrs.out_dtype != "":
        return False
    return True


def qnn_dense(expr):
    """Check if the external ACL codegen for qnn.dense should be used."""
    attrs, args = expr.attrs, expr.args
    data_typ = args[0].checked_type
    if data_typ.dtype != "uint8":
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 2 or kernel_typ.dtype != "uint8":
        return False
    if attrs.out_dtype != "int32":
        return False
    return True


@tvm.ir.register_op_attr("nn.max_pool2d", "target.aipu")
def max_pool2d(expr):
    """Check if the external ACL codegen for maxpool2d should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.layout != "NHWC":
        return False
    typ = args[0].checked_type
    if typ.dtype not in ["float32", "uint8"]:
        return False
    return True


@tvm.ir.register_op_attr("nn.avg_pool2d", "target.aipu")
def avg_pool2d(expr, from_quantized_composite=False):
    """Check if the external ACL codegen for avgpool2d should be used."""
    attrs, args = expr.attrs, expr.args
    typ = args[0].checked_type

    if from_quantized_composite:
        if typ.dtype != "int32":
            return False
    else:
        if typ.dtype not in ["float32"]:
            return False
    if attrs.layout != "NHWC":
        return False

    return True


@tvm.ir.register_op_attr("nn.global_max_pool2d", "target.aipu")
def global_max_pool2d(expr):
    """Check if the external ACL codegen for gloval_maxpool2d should be used."""
    attrs, args = expr.attrs, expr.args
    typ = args[0].checked_type
    if typ.dtype not in ["float32", "uint8"]:
        return False
    if attrs.layout != "NHWC":
        return False
    return True


@tvm.ir.register_op_attr("nn.global_avg_pool2d", "target.aipu")
def global_avg_pool2d(expr):
    """Check if the external ACL codegen for global_avgpool2d should be used."""
    attrs, args = expr.attrs, expr.args
    typ = args[0].checked_type
    if typ.dtype not in ["float32"]:
        return False
    if attrs.layout != "NHWC":
        return False
    return True


@tvm.ir.register_op_attr("maximum", "target.aipu")
def maximum(expr):
    """Check if the external ACL codegen for maximum should be used."""
    args = expr.args
    type_a = args[0].checked_type
    type_b = args[0].checked_type
    return (type_a.dtype == "float32") and (type_b.dtype == "float32")


@tvm.ir.register_op_attr("add", "target.aipu")
def add(expr):
    """Check if the external ACL codegen for add should be used."""
    args = expr.args
    for typ in [args[0].checked_type, args[1].checked_type]:
        if typ.dtype != "float32":
            return False

    return True


@tvm.ir.register_op_attr("qnn.add", "target.aipu")
def qnn_add(expr):
    """Check if the external ACL codegen for add should be used."""
    args = expr.args
    for typ in [args[0].checked_type, args[1].checked_type]:
        if typ.dtype != "uint8":
            return False

    return True


class OpAttrContext(object):
    """ Temporarily changes the attr of an op. """

    def __init__(self, op_name, attr_key, attr_value):
        """Saves the required info for RAII pattern usage.

        Parameters
        ----------
        op_name : str
            The op name.

        attr_key : str
            The attribute name.

        attr_value : object
            The attribute value.
        """
        self.op = relay.op.get(op_name)
        self.attr_key = attr_key
        self.attr_value = attr_value

    def __enter__(self):
        self.older_attr = self.op.get_attr(self.attr_key)
        self.op.reset_attr(self.attr_key)
        self.op.set_attr(self.attr_key, self.attr_value)
        return self

    def __exit__(self, ptype, value, trace):
        self.op.reset_attr(self.attr_key)
        if self.older_attr:
            self.op.set_attr(self.attr_key, self.older_attr)
