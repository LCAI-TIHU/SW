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
# pylint: disable=unused-argument,inconsistent-return-statements

#
# Inspur.
# This is a new or modified file.
#


"""Internal module for registering attribute for annotation."""
import warnings
from tvm import topi
import tvm._ffi
from tvm.relay.op import op as _reg
from .. import expr as _expr
from .. import analysis as _analysis
from .. import op as _op
from . import _quantize
from .quantize import QAnnotateKind, current_qconfig, quantize_context
from .quantize import _forward_op
from ._annotate import QAnnotateExpr, attach_simulated_quantize

if_prequantize_weight = False

def _get_expr_kind(anno):
    """Get the expression and QAnnotateKind from QAnnotateExpr or Expr"""
    if isinstance(anno, QAnnotateExpr):
        return anno.expr, anno.kind
    return anno, None


def register_aipu_annotate_function(op_name, frewrite=None, level=10):
    """register a rewrite function for operator, used by annotation.

    Parameters
    ---------
    op_name: str
        The name of operation

    frewrite : function, optional
        The function to be registered.

    level : int, optional
        The priority level
    """

    def default_rewrite(ref_call, new_args, ctx):
        # recover from QAnnotateExpr
        args = [_get_expr_kind(x)[0] for x in new_args]
        return _forward_op(ref_call, args)

    def _register(func):
        """internal register function"""

        def frewrite_with_guard(ref_call, new_args, ctx):
            if not current_qconfig().guard(ref_call):
                return default_rewrite(ref_call, new_args, ctx)
            return func(ref_call, new_args, ctx)

        return tvm.ir.register_op_attr(op_name, "FQAipuAnnotateRewrite", frewrite_with_guard, level)

    return _register(frewrite) if frewrite is not None else _register


@register_aipu_annotate_function("nn.contrib_conv2d_NCHWc")
def conv2d_nchwc_rewrite(ref_call, new_args, ctx):
    warnings.warn(
        "NCHWc layout Conv2D detected, please use a lower "
        "optimization level before applying the quantization "
        "pass as quantization will have no effect here..."
    )


@register_aipu_annotate_function("nn.conv2d")
def conv2d_rewrite(ref_call, new_args, ctx):
    """Rewrite function for conv2d. Lhs of conv will be quantized to
    input field, and rhs of conv will be quantized to weight field.
    Output would be in activation field"""
    if quantize_context().check_to_skip(ref_call):
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None or lhs_kind == QAnnotateKind.ACTIVATION:
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)

    assert rhs_kind is None
    #
    if if_prequantize_weight:
        axis = -1
        if ref_call.attrs["groups"] == ref_call.attrs["channels"] and ref_call.attrs["groups"] != 1:
            axis = 2
        rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT, True, "round", axis)
    #


    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])

    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)

@register_aipu_annotate_function("nn.pad")
def pad_rewrite(ref_call, new_args, ctx):
    if quantize_context().check_to_skip(ref_call):
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])
    if lhs_kind is None:
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)

        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.INPUT)
    else:
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.INPUT)


@register_aipu_annotate_function("nn.conv1d")
def conv1d_rewrite(ref_call, new_args, ctx):
    """Rewrite function for conv1d. Lhs of conv will be quantized to
    input field, and rhs of conv will be quantized to weight field.
    Output would be in activation field"""
    if quantize_context().check_to_skip(ref_call):
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None or lhs_kind == QAnnotateKind.ACTIVATION:
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)

    assert rhs_kind is None
    rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])

    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)


@register_aipu_annotate_function("nn.dense")
def dense_rewrite(ref_call, new_args, ctx):
    """Rewrite function for dense. Lhs of dense will be quantized to input field, and rhs of
    dense will be quantized to weight field. Output would be in activation field."""

    if current_qconfig().skip_dense_layer:
        return None

    if quantize_context().check_to_skip(ref_call):
        return None

    if list(new_args[1].checked_type.shape) == [768, 2]: #  or list(new_args[1].checked_type.shape) == [2, 768]: # TODO: bert need (first and last dense)
        print("pass1")
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None or lhs_kind == QAnnotateKind.ACTIVATION:
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)

    assert rhs_kind is None
    #
    if if_prequantize_weight:
        rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT, axis=0)
    #

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
    # quantize_context().stop_quantize()
    # expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT)

    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)


@register_aipu_annotate_function("add")
def add_rewrite(ref_call, new_args, ctx):
    """Rewrite function for add."""
    if quantize_context().check_to_skip(ref_call):
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None and rhs_kind is None:
        # trivial case
        return None

    if lhs_kind is None and rhs_kind is not None:
        # quantize lhs to INPUT field if it is normal expression
        assert rhs_kind in [QAnnotateKind.INPUT, QAnnotateKind.ACTIVATION]
        # lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT)
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)

    if lhs_kind is not None and rhs_kind is None:
        if _analysis.check_constant(rhs_expr):
            # - introduced by batch_norm: add(out, const)
            # rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)
            pass
        else:
            # TODO: This is only an ad hoc approach to deal with the bug for the node "bert/encoder/layer_0/intermediate/dense/add" in bert-base model
            if rhs_expr.op == tvm.ir.op.Op.get("multiply"):
                return _forward_op(ref_call, [lhs_expr, rhs_expr])
            ##
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT)
            expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
            return expr
            # rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT)
            # expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
            # return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
            # print("pass2") 
            # return None # TODO: bert need (layer norm input) 

        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT)
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)

    if lhs_kind is not None and rhs_kind is not None:
        if lhs_kind == QAnnotateKind.INPUT and rhs_kind == QAnnotateKind.INPUT:
            expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
            expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT)
            return QAnnotateExpr(expr, QAnnotateKind.INPUT)
        elif lhs_kind == QAnnotateKind.ACTIVATION or rhs_kind == QAnnotateKind.ACTIVATION:
            lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT)
            expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
            expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT)
            return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)

    raise ValueError()


def identity_rewrite(ref_call, new_args, ctx):
    """Simply forward the original operation"""
    if quantize_context().check_to_skip(ref_call):
        return None

    x_expr, x_kind = _get_expr_kind(new_args[0])
    if x_kind is None:
        return None

    ret_expr = _forward_op(ref_call, [x_expr])
    return QAnnotateExpr(ret_expr, x_kind)

register_aipu_annotate_function("nn.batch_flatten", identity_rewrite)
register_aipu_annotate_function("transpose", identity_rewrite)
register_aipu_annotate_function("annotation.stop_fusion", identity_rewrite)

def act_rewrite(ref_call, new_args, ctx):
    """Rewrite function for activation"""
    if quantize_context().check_to_skip(ref_call):
        return None

    x_expr, x_kind = _get_expr_kind(new_args[0])
    if x_kind is None:
        return None

    x_expr = attach_simulated_quantize(x_expr, QAnnotateKind.INPUT)
    ret_expr = _forward_op(ref_call, [x_expr])
    ret_expr = attach_simulated_quantize(ret_expr, QAnnotateKind.INPUT)
    return QAnnotateExpr(ret_expr, x_kind)

register_aipu_annotate_function("clip", act_rewrite)
register_aipu_annotate_function("nn.relu", act_rewrite)
register_aipu_annotate_function("nn.leaky_relu", act_rewrite)


@register_aipu_annotate_function("annotation.cast_hint")
def cast_hint_rewrite(ref_call, new_args, ctx):
    """Rewrite function to force cast"""
    expr, x_kind = _get_expr_kind(new_args[0])

    if quantize_context().check_to_skip(ref_call):
        return expr

    if x_kind is None:
        return new_args[0]
    if x_kind == QAnnotateKind.ACTIVATION:
        expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT)

    expr = _forward_op(ref_call, [expr])
    return QAnnotateExpr(expr, QAnnotateKind.INPUT)


@register_aipu_annotate_function("concatenate")
def concatenate_rewrite(ref_call, new_args, ctx):
    """Rewrite function for concatenate"""
    if quantize_context().check_to_skip(ref_call):
        return None

    input_tuple = new_args[0]
    expr_list = [_get_expr_kind(x)[0] for x in input_tuple]
    kind_list = [_get_expr_kind(x)[1] for x in input_tuple]

    # make sure the inputs of concatenate are all normal
    # expression or annotate expression
    if all([k is None for k in kind_list]):
        return None
    for i, k in enumerate(kind_list):
        if k is not None:
            expr_list[i] = attach_simulated_quantize(expr_list[i], QAnnotateKind.INPUT)
    expr = _forward_op(ref_call, [_expr.Tuple(expr_list)])
    # expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT)
    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)



@register_aipu_annotate_function("nn.batch_matmul")
def batch_matmul_rewrite(ref_call, new_args, ctx):
    """Rewrite function for batch_matmul"""
    if quantize_context().check_to_skip(ref_call):
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None or lhs_kind == QAnnotateKind.ACTIVATION:
        if _analysis.check_constant(lhs_expr):
            lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.WEIGHT)
        else:
            lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)

    if rhs_kind is None or rhs_kind == QAnnotateKind.ACTIVATION:
        if _analysis.check_constant(rhs_expr):
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)
        else:
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
    expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT)
    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
