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
# pylint: disable=invalid-name, unused-argument, too-many-lines, len-as-condition, broad-except
"""Tensorflow2.x to relay converter ops and helper"""

import tvm
from .common import get_relay_op
from .. import expr as _expr
from .. import op as _op

import numpy as np
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes


from tvm.topi.utils import get_const_tuple
from tvm.relay.prelude import Prelude, StaticTensorArrayOps, get_tensor_array_shape, TensorArrayOps
from ..ty import Any, TensorType
from .common import infer_value as _infer_value
from .common import AttrCvt
from .common import infer_shape as _infer_shape
from .common import infer_type as _infer_type
from .common import infer_channels as _infer_channels

def _infer_type_with_prelude(val, prelude):
    body = _infer_type(val, prelude.mod)
    return body.checked_type

def _get_pad_pair(input1d, kernel1d, stride1d):
    if input1d % stride1d == 0:
        pad = max(kernel1d - stride1d, 0)
    else:
        pad = max(kernel1d - (input1d % stride1d), 0)

    pad_before = pad // 2
    pad_after = pad - pad_before

    return [pad_before, pad_after]

def _get_param(params, input_node):
    if isinstance(input_node, _expr.Constant):
        return np.atleast_1d(input_node.data.asnumpy())
    return params[input_node.name_hint].asnumpy()


def _get_num_param(params, input_node):
    return _get_param(params, input_node).item()


def _get_list_param(params, input_node):
    return _get_param(params, input_node).tolist()


def _get_tuple_param(params, input_node):
    return tuple(_get_param(params, input_node))

def _get_more_static_shape(shape0, shape1):
    """Compare two shapes with the same rank,
    and return the one with fewer symbolic dimension.
    """
    assert len(shape0) == len(shape1)
    num_sym_dim0 = 0
    num_sym_dim1 = 0
    for dim0, dim1 in zip(list(shape0), list(shape1)):
        if not isinstance(dim0, int):
            num_sym_dim0 += 1
        if not isinstance(dim1, int):
            num_sym_dim1 += 1

    if num_sym_dim0 < num_sym_dim1:
        return shape0
    return shape1

def _bias_add():
    def _impl(inputs, attr, params, mod):
        # Must expand for proper broadcasting in NCHW.
        if "data_format" in attr and attr["data_format"].decode("utf-8") == "NCHW":
            bias = _op.reshape(inputs[1], newshape=(1, -1, 1, 1))
        else:
            bias = inputs[1]
        return _op.add(inputs[0], bias)

    return _impl

def _squeeze():
    def _impl(inputs, attr, params, mod):
        if len(attr["squeeze_dims"]) == 0:
            attr["squeeze_dims"] = None
        return AttrCvt(op_name="squeeze", transforms={"squeeze_dims": "axis"}, ignores=["T"])(
            inputs, attr
        )

    return _impl

def _where():
    def _impl(inputs, attr, params, mod):
        if len(inputs) == 1:
            return AttrCvt(op_name="argwhere")(inputs, attr)
        return AttrCvt(op_name="where")(inputs, attr)

    return _impl

def _gather():
    def _impl(inputs, attr, params, mod):
        if len(inputs) > 2:
            axis = _get_num_param(params, inputs.pop(2))
        else:
            axis = 0
        if int(attr.get("batch_dims", 0)) != 0:
            raise tvm.error.OpAttributeUnImplemented("Attribute batch_dims is not supported")
        new_input = inputs[0:2]
        return AttrCvt(
            op_name="take",
            extras={"axis": tvm.tir.const(axis, "int32")},
            ignores=["Tindices", "Tparams", "validate_indices", "Taxis", "_class", "batch_dims"],
        )(new_input, attr)

    return _impl

def _softmax():
    def _impl(inputs, attr, params, mod):
        return AttrCvt(op_name="softmax", transforms={"axis": ("axis", 1)})([inputs[0]], attr)

    return _impl

def _conv(opname):
    def _impl(inputs, attr, params, mod):
        attr["data_format"] = attr["data_format"].decode("utf-8")
        flip_layout = False

        if opname == "conv_transpose" and attr["data_format"] == "NHWC":
            # transform to NCHW for TVM backend compatible and set 'flip_layout'
            # to have output flip back to NHWC
            inputs[2] = _op.transpose(inputs[2], axes=(0, 3, 1, 2))
            attr["strides"][1], attr["strides"][2], attr["strides"][3] = (
                attr["strides"][3],
                attr["strides"][1],
                attr["strides"][2],
            )

            attr["data_format"] = "NCHW"

            # Check whether output shapes attribute is set and not None
            if (
                opname == "conv_transpose"
                and len(attr["_output_shapes"]) > 0
                and attr["_output_shapes"][0]
            ):
                tmp_shape = attr["_output_shapes"][0]
                tmp_shape = [tmp_shape[ii] for ii in (0, 3, 1, 2)]
                attr["_output_shapes"][0] = tmp_shape

            flip_layout = True

        inputs_data = inputs[0] if opname != "conv_transpose" else inputs[2]

        # NCHW Layout require weights transpose
        weights_shape = _infer_shape(inputs[1], mod)
        if attr["data_format"] == "NCHW":
            tmp_shape = weights_shape
            if opname in ["conv", "conv_transpose"]:
                tmp_shape = [tmp_shape[ii] for ii in (3, 2, 0, 1)]
                inputs[1] = _op.transpose(inputs[1], axes=(3, 2, 0, 1))
            else:
                tmp_shape = [tmp_shape[ii] for ii in (2, 3, 0, 1)]
                inputs[1] = _op.transpose(inputs[1], axes=(2, 3, 0, 1))
            weights_shape = tmp_shape

        input_shape = _infer_shape(inputs_data, mod)
        if attr["_target_layout"] == "NCHW" and attr["data_format"] == "NHWC":
            input_shape = [input_shape[ii] for ii in (0, 3, 1, 2)]
            inputs_data = _op.transpose(inputs_data, axes=(0, 3, 1, 2))
            if opname in ["conv", "conv_transpose"]:
                weights_shape = [weights_shape[ii] for ii in (3, 2, 0, 1)]
                inputs[1] = _op.transpose(inputs[1], axes=(3, 2, 0, 1))
            else:
                weights_shape = [weights_shape[ii] for ii in (2, 3, 0, 1)]
                inputs[1] = _op.transpose(inputs[1], axes=(2, 3, 0, 1))

            attr["data_format"] = "NCHW"
            attr["strides"] = [attr["strides"][ii] for ii in (0, 3, 1, 2)]
            flip_layout = True

        if attr["data_format"] == "NHWC":
            in_channels = input_shape[3]
            kernel_h, kernel_w, _, depth_mult = weights_shape
            attr["kernel_shape"] = (weights_shape[0], weights_shape[1])
            if opname == "conv":
                attr["channels"] = weights_shape[3]
            elif opname == "conv_transpose":
                attr["channels"] = weights_shape[2]
            else:
                attr["channels"] = input_shape[3] * depth_mult

            if "dilations" in attr:
                attr["dilations"] = (attr["dilations"][1], attr["dilations"][2])
            attr["strides"] = (attr["strides"][1], attr["strides"][2])
        elif attr["data_format"] == "NCHW":
            in_channels = input_shape[1]
            _, depth_mult, kernel_h, kernel_w = weights_shape
            attr["kernel_shape"] = (weights_shape[2], weights_shape[3])
            if opname == "conv":
                attr["channels"] = weights_shape[0]
            elif opname == "conv_transpose":
                attr["channels"] = weights_shape[1]
            else:
                attr["channels"] = input_shape[1] * depth_mult
                if attr["channels"] < 0:
                    attr["channels"] *= -1

            if "dilations" in attr:
                attr["dilations"] = (attr["dilations"][2], attr["dilations"][3])
            attr["strides"] = (attr["strides"][2], attr["strides"][3])
        else:
            msg = 'Value {} in attribute "data_format" of operator Conv is ' "not valid."
            raise tvm.error.OpAttributeInvalid(msg.format(attr["data_format"]))

        if opname == "depthwise":
            attr["groups"] = in_channels

        # Fix padding
        attr["padding"] = attr["padding"].decode("utf-8")

        if attr["padding"] == "VALID":
            attr["padding"] = [0, 0]
        elif attr["padding"] == "SAME":
            stride_h, stride_w = attr["strides"]
            kernel_h, kernel_w = attr["kernel_shape"]

            pdata_shape = input_shape
            # Check whether output shapes attribute is set and not None
            if (
                opname == "conv_transpose"
                and len(attr["_output_shapes"]) > 0
                and attr["_output_shapes"][0]
            ):
                pdata_shape = attr["_output_shapes"][0]

            if attr["data_format"] == "NHWC":
                in_h = pdata_shape[1]
                in_w = pdata_shape[2]
            else:
                in_h = pdata_shape[2]
                in_w = pdata_shape[3]

            dilation_h = attr["dilations"][0]
            dilation_w = attr["dilations"][1]
            dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
            dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
            pad_v = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, dilated_kernel_w, stride_w)

            attr["padding"] = [pad_v[0], pad_h[0], pad_v[1], pad_h[1]]
        else:
            msg = 'Value {} in attribute "padding" of operator Conv is not ' "valid."
            raise tvm.error.OpAttributeInvalid(msg.format(attr["padding"]))

        if "kernel_layout" not in attr:
            if opname in ["conv", "conv_transpose"]:
                attr["kernel_layout"] = "HWIO" if attr["data_format"] == "NHWC" else "OIHW"
            else:
                attr["kernel_layout"] = "HWOI" if attr["data_format"] == "NHWC" else "OIHW"

        # Ignore the new attributes from TF2.0, for now.
        out = AttrCvt(
            op_name=_dimension_picker(
                "conv", surfix="_transpose" if opname == "conv_transpose" else ""
            ),
            ignores=["explicit_paddings"],
            transforms={
                "kernel_shape": "kernel_size",
                "data_format": "data_layout",
                "dilations": ("dilation", (0, 0)),
                "group": ("groups", 1),
            },
            custom_check=_dimension_constraint(),
        )([inputs_data, inputs[1]], attr)

        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 1))

        return out

    return _impl

def _cast():
    def _impl(inputs, attr, params, mod):
        return inputs[0].astype(attr["DstT"].name)

    return _impl

def _dimension_picker(prefix, surfix=""):
    def _impl(attr):
        kernel = attr["kernel_shape"]
        if len(kernel) == 2:
            return prefix + "2d" + surfix
        if len(kernel) == 3:
            return prefix + "3d" + surfix
        raise tvm.error.OpAttributeInvalid(
            "Only 2D or 3D kernels are supported for operator {}".format(prefix + "2d or 3d")
        )

    return _impl


def _dimension_constraint():
    def _dim_check(attrs):
        if len(attrs["kernel_shape"]) in (2, 3):
            return True
        return False

    return _dim_check, "Only 2d or 3d kernel supported."


def _elemwise(name):
    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 2, "{} take 2 inputs, {} given".format(name, len(inputs))
        return get_relay_op(name)(*inputs)

    return _impl

def identity():
    def _impl(inputs, attr, params, mod):
        return inputs[0]

    return _impl


def _matmul():
    def _impl(inputs, attr, params, mod):
        channels = _infer_channels(inputs[1], not attr["transpose_b"])
        if attr["transpose_a"]:
            inputs[0] = _op.transpose(inputs[0], axes=(1, 0))
        if not attr["transpose_b"]:
            inputs[1] = _op.transpose(inputs[1], axes=(1, 0))
        return AttrCvt(
            op_name="dense", extras={"units": channels}, ignores=["transpose_a", "transpose_b", "T"]
        )(inputs, attr)

    return _impl

def _no_op():
    def _impl(inputs, attr, params, mod):
        # ToDo: This should really be an op that returns nothing, which could
        # be represented as an empty tuple. It turns out that TVM
        # infrastructure doesn't like running functions that return None and
        # also don't like running functions that return an empty tuple. So it
        # doesn't work, but it should be made to work and then this could be
        # improved. In the mean time, it is hard to imagine a case where it
        # matters in any real way that a no-op is converted to a constant 0.
        return tvm.relay.const(0)

    return _impl


def _pooling(name):
    def _impl(inputs, attr, params, mod):

        attr["data_format"] = attr["data_format"].decode("utf-8")
        flip_layout = False

        input_shape = _infer_shape(inputs[0], mod)

        if attr["data_format"] == "NHWC":
            attr["kernel_shape"] = (attr["ksize"][1], attr["ksize"][2])
            attr["strides"] = (attr["strides"][1], attr["strides"][2])
        elif attr["data_format"] == "NCHW":
            attr["kernel_shape"] = (attr["ksize"][2], attr["ksize"][3])
            attr["strides"] = (attr["strides"][2], attr["strides"][3])
        else:
            msg = 'Value {} of attribute "data_format" of operator Pooling ' "is not valid."
            raise tvm.error.OpAttributeInvalid(msg.format(attr["data_format"]))

        if attr["_target_layout"] == "NCHW" and attr["data_format"] == "NHWC":
            tmp_shape = _infer_shape(inputs[0], mod)
            input_shape = [tmp_shape[ii] for ii in (0, 3, 1, 2)]
            inputs[0] = _op.transpose(inputs[0], axes=(0, 3, 1, 2))
            attr["data_format"] = "NCHW"
            flip_layout = True

        # Fix padding
        attr["padding"] = attr["padding"].decode("utf-8")

        if attr["padding"] == "VALID":
            attr["padding"] = [0, 0]
        elif attr["padding"] == "EXPLICIT":
            paddings = attr["explicit_paddings"]
            assert len(paddings) == 8
            if flip_layout or attr["data_format"] == "NHWC":
                attr["padding"] = [paddings[2], paddings[4], paddings[3], paddings[5]]
            else:
                attr["padding"] = [paddings[4], paddings[6], paddings[5], paddings[7]]
        elif attr["padding"] == "SAME":
            stride_h, stride_w = attr["strides"]
            kernel_h, kernel_w = attr["kernel_shape"]
            if attr["data_format"] == "NHWC":
                in_h = input_shape[1]
                in_w = input_shape[2]
            else:
                in_h = input_shape[2]
                in_w = input_shape[3]

            pad_v = _get_pad_pair(in_h, kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, kernel_w, stride_w)

            attr["padding"] = [pad_v[0], pad_h[0], pad_v[1], pad_h[1]]
        else:
            msg = 'Value {} in attribute "padding" of operator Pooling is ' "not valid."
            raise tvm.error.OpAttributeInvalid(msg.format(attr["padding"]))

        if name == "avg_pool":
            attr["count_include_pad"] = False

        out = AttrCvt(
            op_name=_dimension_picker(name),
            transforms={"kernel_shape": "pool_size", "data_format": "layout"},
            ignores=["ksize", "explicit_paddings"],
            extras={"ceil_mode": False},
            custom_check=_dimension_constraint(),
        )(inputs, attr)

        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 1))

        return out

    return _impl

def _broadcast(name):
    def _impl(inputs, attr, params, mod):
        return AttrCvt(op_name=name, ignores=["name", "incompatible_shape_error", "Tidx"])(
            inputs, attr
        )

    return _impl


def _reshape():
    def _impl(inputs, attr, params, mod):
        pop_node = inputs.pop(1)

        try:
            shape_arg = _get_tuple_param(params, pop_node)
        except AttributeError:
            # Shape operator is already pruned, hence
            # try to infer shape by precompute prune if possible.
            try:
                params_new = _infer_value(pop_node, params, mod)
                shape_arg = tuple(params_new.asnumpy().astype("int32").flatten())
            except Exception:
                # Deal with symbolic shape case.
                if isinstance(pop_node, _expr.Call) and "shape_of" in str(pop_node.op):
                    # shape_of is the direct ancestor.
                    return _op.reshape_like(inputs[0], pop_node.args[0])
                shape_arg = pop_node

        return AttrCvt(op_name="reshape", extras={"newshape": shape_arg}, ignores=["Tshape"])(
            inputs, attr
        )

    return _impl

def _fused_batch_norm():
    def _impl(inputs, attr, params, mod):
        # Tensorflow: (data, gamma, beta, moving_mean, moving_variance)
        # Relay:       (data, gamma, beta, moving_mean, moving_varience)
        assert len(inputs) == 5
        axis = 3
        need_cast = False

        if "data_format" in attr:
            attr["data_format"] = attr["data_format"].decode("utf-8")
            if attr["data_format"] == "NCHW":
                axis = 1
        if "U" in attr and attr["U"].name != attr["T"].name:
            need_cast = True
            inputs[0] = _op.cast(inputs[0], dtype=attr["U"].name)
        # Check if mean and variance are empty
        # If so, replace them with Mean and Variance Ops
        # For run-time calculation
        moving_mean_shape = [int(n) for n in inputs[3].type_annotation.shape]
        moving_variance_shape = [int(n) for n in inputs[4].type_annotation.shape]
        if moving_mean_shape[0] == 0 and moving_variance_shape[0] == 0:
            inputs[3] = _op.mean(inputs[0], axis=axis, keepdims=False, exclude=True)
            inputs[4] = _op.variance(inputs[0], axis=axis, keepdims=False, exclude=True)
        out = AttrCvt(
            op_name="batch_norm",
            transforms={"scale_after_normalization": "scale", "variance_epsilon": "epsilon"},
            extras={"axis": axis},
            ignores=["data_format", "U", "exponential_avg_factor"],
            disables=["momentum"],
        )(inputs, attr)

        if need_cast:
            out = _expr.TupleGetItem(out.astuple(), 0)
            out = _op.cast(out, dtype=attr["T"].name)
        return out

    return _impl

def _floormod():
    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 2
        return AttrCvt("floor_mod")(inputs, attr)

    return _impl

def _logical(name):
    def _impl(inputs, attr, params, mod):
        return AttrCvt(op_name=name)(inputs, attr)

    return _impl

def _stridedSlice():
    def _impl(inputs, attr, params, mod):
        """Strided Slice.
        Operator description: https://www.tensorflow.org/api_docs/python/tf/strided_slice
        Tensorflow mask validation: https://github.com/tensorflow/tensorflow/blob/master/
        tensorflow/core/util/strided_slice_op.cc#L147-L368
        """
        try:
            begin = _get_list_param(params, inputs[1])
        except (IndexError, KeyError, AttributeError):
            begin = _infer_value(inputs[1], params, mod).asnumpy().tolist()
        end = _get_list_param(params, inputs[2])
        stride = _get_list_param(params, inputs[3])

        begin_mask = int(attr.get("begin_mask", 0))
        end_mask = int(attr.get("end_mask", 0))
        ellipsis_mask = int(attr.get("ellipsis_mask", 0))
        new_axis_mask = int(attr.get("new_axis_mask", 0))
        shrink_axis_mask = int(attr.get("shrink_axis_mask", 0))
        in_type = _infer_type(inputs[0], mod)
        data_shape = get_const_tuple(in_type.checked_type.shape)
        data_dim = len(data_shape)
        stride_dim = len(stride)

        # This is a special routine to handle strided_slice after shape_of.
        # We need this since in some cases we want to do strided_slice on
        # a partial symbolic shape, such as (1, ?), and get a static shape
        # (1,). Directly slice on shape_of will result in fully dynamic shape.
        # TODO(kevinthesun): Can we generalize this process with partial eval?
        if isinstance(inputs[0], _expr.Call) and inputs[0].op == _op.get("shape_of"):
            bg = begin[0]
            ed = end[0]
            st = stride[0]

            if ed <= 0 < st:
                ed += data_shape[0]

            in_shape = _infer_shape(inputs[0].args[0], mod)
            dtype = in_type.checked_type.dtype
            out_data = []
            idx = bg
            while idx < ed:
                if isinstance(in_shape[idx], int):
                    out_data.append(in_shape[idx])
                else:
                    break
                idx += st

            # Only return when in_shape is fully static in the range from begin to end.
            if idx >= ed:
                ret = _expr.const(out_data, dtype)
                if shrink_axis_mask:
                    ret = _op.squeeze(ret)

                return ret

        def _transform_mask(stride_dim, ellipsis_mask):
            """Handle mask inputs to create new begin, end, stride and output shape"""
            m_begin = [0] * data_dim
            m_end = [0] * data_dim
            m_stride = [0] * data_dim
            fshape_indices = []
            # Count new axis after ellipsis_mask, consider while applying ellipsis_mask.
            ellipsis_seen = False
            new_axes_after_ellipsis = 0
            for i in range(stride_dim):
                mask = 1 << i
                if ellipsis_seen and (mask & new_axis_mask) != 0:
                    new_axes_after_ellipsis += 1
                if (mask & ellipsis_mask) != 0:
                    ellipsis_seen = True
            if not ellipsis_seen:
                # Used later for extending the stride attributes in the below loop.
                ellipsis_mask |= 1 << stride_dim
                stride_dim += 1
            final_index = 0
            for index in range(stride_dim):
                mask = 1 << index
                if mask & ellipsis_mask:
                    # Identify the end index for applying ellipsis_mask
                    to_index = min(
                        ((data_dim - (stride_dim - index)) + 1 + new_axes_after_ellipsis), data_dim
                    )
                    for i in range(final_index, to_index):
                        m_begin[final_index] = 0
                        m_end[final_index] = data_shape[final_index]
                        m_stride[final_index] = 1
                        fshape_indices.append(final_index)
                        final_index += 1
                elif mask & new_axis_mask:
                    fshape_indices.append(-1)
                elif not mask & new_axis_mask:
                    if final_index == len(m_begin):
                        break
                    if mask & begin_mask:
                        m_begin[final_index] = -1 if stride[index] < 0 else 0
                    elif begin[index]:
                        m_begin[final_index] = begin[index]
                    if mask & end_mask:
                        m_end[final_index] = (
                            -(data_shape[final_index] + 1)
                            if stride[index] < 0
                            else data_shape[final_index]
                        )
                    elif end[index]:
                        m_end[final_index] = end[index]
                    m_stride[final_index] = stride[index]
                    if mask & shrink_axis_mask:
                        # Tensorflow make axis with shrink_axis_mask as dimension 1
                        m_begin[final_index] = (
                            data_shape[final_index] + begin[index]
                            if begin[index] < 0
                            else begin[index]
                        )
                        m_end[final_index] = begin[index] + 1
                        m_stride[final_index] = 1
                        fshape_indices.append(-2)
                    else:
                        fshape_indices.append(final_index)

                    final_index += 1
            return m_begin, m_end, m_stride, fshape_indices

        fshape_indices = None
        if begin_mask or end_mask or ellipsis_mask or new_axis_mask or shrink_axis_mask:
            begin, end, stride, fshape_indices = _transform_mask(stride_dim, ellipsis_mask)
        out = _op.strided_slice(inputs[0], begin=begin, end=end, strides=stride)
        out_shape = _infer_shape(out, mod=mod)
        if not fshape_indices:
            fshape_indices = range(len(out_shape))

        # Create final output shape.
        final_output = []
        for gather_index in fshape_indices:
            if gather_index == -1:
                final_output.append(1)
            elif gather_index == -2:
                pass
            else:
                final_output.append(out_shape[gather_index])

        if not final_output:
            if not shrink_axis_mask:
                ret = out
            else:
                final_shape = []
                for dim in out_shape:
                    if dim != 1:
                        final_shape.append(dim)
                if len(final_shape) == 0:
                    ret = _op.squeeze(out)
                else:
                    # We need reshape to handle dynamic shape.
                    ret = _op.reshape(out, newshape=tuple(final_shape))
        else:
            ret = _op.reshape(out, newshape=tuple(final_output))
        return ret

    return _impl

def _shape():
    def _impl(inputs, attr, params, mod):
        # input_shape = _infer_shape(inputs[0], mod)
        # not yet tested
        # is_symbolic_shape = False
        # for axis in input_shape:
        #     if not isinstance(axis, (int, tvm.tir.IntImm)):
        #         is_symbolic_shape = True
        #         break

        ret = _op.shape_of(inputs[0], dtype=attr["out_type"].name)

        # is_symbolic_shape is not tested:

        return ret

    return _impl

def _fill():
    def _impl(inputs, attr, params, mod):
        try:
            output_shape = _infer_value(inputs[0], params, mod).asnumpy().tolist()
        except Exception:
            output_shape = inputs[0]

        return _op.full(inputs[1], output_shape, attr["T"].name)

    return _impl

def _pack():
    def _impl(inputs, attr, params, mod):
        axis = int(attr["axis"])
        inputs_reshaped = [_op.expand_dims(i, axis=axis, num_newaxis=1) for i in inputs]
        return _op.concatenate(inputs_reshaped, axis)

    return _impl

def _split(has_size_vector):
    # TF documentation https://www.tensorflow.org/api_docs/python/tf/split
    def _impl(inputs, attr, params, mod):
        try:
            # order and number of inputs are different:
            # if has_size_vector:
            #     https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/split-v
            # else:
            #     https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/split

            # in addition, `axis` and `num_or_size_splits` can be tensors in TensorFlow,
            # we can only support constants
            if has_size_vector:
                input_node_index = 0
                input_axis_index = 2
                size_splits = _get_param(params, inputs[1])
                section_beginnings = np.cumsum(size_splits)[:-1]
                indices_or_sections = tuple(section_beginnings)
            else:
                input_node_index = 1
                input_axis_index = 0
                indices_or_sections = attr["num_split"]
            input_node = inputs[input_node_index]
            axis_input_value = _get_num_param(params, inputs[input_axis_index])
        except (IndexError, KeyError, AttributeError):
            raise TypeError(
                "Unsupported argument for split: `axis` and `num_or_size_splits` "
                "should be constants"
            )
        return _op.split(
            input_node, indices_or_sections=indices_or_sections, axis=int(axis_input_value)
        )

    return _impl


def _concatV2():
    def _impl(inputs, attr, params, mod):
        pop_node = inputs.pop(len(inputs) - 1)
        axis = int(_get_num_param(params, pop_node))
        return AttrCvt(op_name="concatenate", ignores=["T", "N", "Tidx"], extras={"axis": axis})(
            [inputs], attr
        )

    return _impl

def _expand_dims():
    def _impl(inputs, attr, params, mod):
        dim_input = inputs.pop(1)
        axis = _get_num_param(params, dim_input)
        return AttrCvt(
            op_name="expand_dims",
            ignores=["Tdim", "N"],
            extras={"axis": int(axis), "num_newaxis": 1},
        )(inputs, attr)

    return _impl



def _tile():
    def _impl(inputs, attr, params, mod):
        reps_input = inputs.pop()
        if isinstance(reps_input, _expr.Call):
            np_reps = _infer_value(reps_input, params, mod).asnumpy()
            reps = [np_reps.flatten()[i] for i in range(np_reps.flatten().shape[0])]
        else:
            reps = _get_list_param(params, reps_input)
        new_input = [inputs.pop(0)]

        return AttrCvt(op_name="tile", extras={"reps": tuple(reps)}, ignores=["Tmultiples"])(
            new_input, attr
        )

    return _impl


def _transpose():
    def _impl(inputs, attr, params, mod):
        # If perm is not specified, axes is left empty,
        # otherwise its value is get from params
        try:
            axes = _get_list_param(params, inputs[1])
        except (IndexError, KeyError, AttributeError):
            axes = _infer_value(inputs[1], params, mod).asnumpy().tolist()
        return _op.transpose(inputs[0], axes=axes)

    return _impl

def _tensorlist_reserve():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr.get("element_dtype").name
        elem_shape = _infer_value(inputs[0], params, prelude.mod)
        elem_shape = tuple(elem_shape.asnumpy().astype("int32").flatten())
        assert -1 not in elem_shape, "TensorList size and element shape must be static"

        static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, elem_shape)
        static_tensor_array_ops.register()
        tensor_array_constructor = static_tensor_array_ops.get_global_var("tensor_array")
        tensor_array = tensor_array_constructor(inputs[1])
        return tensor_array

    return _impl

def _tensorlist_set_item():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr.get("element_dtype").name
        input_ta = inputs[0]
        input_ta_shape = get_tensor_array_shape(input_ta, dtype_str, prelude)
        input_t_shape = _infer_type_with_prelude(inputs[2], prelude).shape

        static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_ta_shape)
        static_tensor_array_ops.register()
        tensor_func = static_tensor_array_ops.get_ctor("tensor_constructor")
        v = tensor_func(inputs[2])
        # Write tensor with more static shape
        actual_shape = _get_more_static_shape(input_t_shape, input_ta_shape)
        if actual_shape != input_t_shape:
            new_shape = []
            num_any_dim = 0
            for dim in actual_shape:
                if not isinstance(dim, int):
                    num_any_dim += 1
                new_shape.append(dim if isinstance(dim, int) else -1)
            if num_any_dim <= 1:
                v = tensor_func(_op.reshape(inputs[2], new_shape))
        write_func = prelude.get_global_var_static(
            "tensor_array_write", dtype_str, input_ta_shape
        )
        out = write_func(input_ta, inputs[1], v)
        return out

    return _impl

def _tensorlist_get_item():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr["element_dtype"].name
        input_shape = get_tensor_array_shape(inputs[0], dtype_str, prelude)

        static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_shape)
        static_tensor_array_ops.register()
        read_func = static_tensor_array_ops.get_global_var("tensor_array_read")
        out_tensor = read_func(inputs[0], _op.take(inputs[1], tvm.relay.const(0)))
        get_data_func = static_tensor_array_ops.get_global_var("tensor_get_data")
        out = get_data_func(out_tensor)
        return out

    return _impl

def _tensorlist_stack():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr["element_dtype"].name
        input_ta_shape = get_tensor_array_shape(inputs[0], dtype_str, prelude)
        assert input_ta_shape is not None

        static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_ta_shape)
        static_tensor_array_ops.register()
        stack_func = prelude.get_global_var_static("tensor_array_stack", dtype_str, input_ta_shape)
        out_tensor = stack_func(inputs[0])
        out_shape = (Any(),) + input_ta_shape
        static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, out_shape)
        static_tensor_array_ops.register()
        get_data_func = prelude.get_global_var_static("tensor_get_data", dtype_str, out_shape)
        out = get_data_func(out_tensor)

        return out

    return _impl

def _tensorlist_from_tensor():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr["element_dtype"].name
        input_ta_shape = _infer_type_with_prelude(inputs[0], prelude).shape
        assert input_ta_shape is not None

        static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_ta_shape)
        static_tensor_array_ops.register()
        unstack_func = prelude.get_global_var_static("tensor_array_unstack", dtype_str, input_ta_shape)
        out = unstack_func(inputs[0])
        return out

    return _impl

convert_map = {
    # "Add": _elemwise("add"),
    # "AddV2": _elemwise("add"),
    # "BiasAdd": _bias_add(),

    # "Cast": _cast(),
    # "Conv2D": _conv("conv"),
    # "ConcatV2": _concatV2(),

    # "Equal": _broadcast("equal"),
    # "ExpandDims": _expand_dims(),

    # "Fill": _fill(),
    # "FusedBatchNormV3": _fused_batch_norm(),
    # "Floor": AttrCvt("floor"),
    # "FloorMod": _floormod(),
    # "GatherV2": _gather(),
    # "Identity": identity(),
    # "Less": _broadcast("less"),
    # "LogicalAnd": _logical("logical_and"),
    # "LogicalNot": _logical("logical_not"),
    # "LogicalOr": _logical("logical_or"),

    # "MatMul": _matmul(),
    # "MaxPool": _pooling("max_pool"),
    # "Maximum": _elemwise("maximum"),
    # "Minimum": _elemwise("minimum"),
    # "Mul": _elemwise("multiply"),
    # "NoOp": _no_op(),
    # "Pack": _pack(),

    # "RealDiv": _elemwise("divide"),
    # "Relu": AttrCvt("relu"),
    # "Reshape": _reshape(),

    # "Sigmoid": AttrCvt("sigmoid"),
    # "Softmax": _softmax(),
    # "Sub": _elemwise("subtract"),
    # "StridedSlice": _stridedSlice(),
    "Shape": _shape(),
    # "Split": _split(False),
    # "Squeeze": _squeeze(),

    # "Tanh": AttrCvt("tanh"),
    # "Tile": _tile(),
    # "Transpose": _transpose(),
    # "Where": _where(),

    "TensorListFromTensor": _tensorlist_from_tensor(),
    "TensorListGetItem": _tensorlist_get_item(),
    "TensorListReserve": _tensorlist_reserve(),
    "TensorListSetItem": _tensorlist_set_item(),
    "TensorListStack": _tensorlist_stack(),
}

def set_span(sym, node_name):
    span = tvm.relay.Span(tvm.relay.SourceName(node_name), 0, 0, 0, 0)
    if isinstance(sym, _expr.Call):
        sym = _expr.Call(sym.op, sym.args, sym.attrs, sym.type_args, span)
    elif isinstance(sym, _expr.TupleWrapper):
        tuple_value = sym.tuple_value
        if isinstance(tuple_value, _expr.Call):
            tuple_value = _expr.Call(
                tuple_value.op, tuple_value.args, tuple_value.attrs, tuple_value.type_args, span
            )
            sym = _expr.TupleWrapper(tuple_value, sym.size)
    return sym

def _get_const_value(node):
    return node.attr["value"].tensor

def convert_const_node(node, shape):
    """convert tf const node into relay const or var
    """
    tensor_value = _get_const_value(node)
    np_array = tensor_util.MakeNdarray(tensor_value)

    if np_array.dtype == np.dtype(object):
        # assert False # not tested, maybe tf string type?
        if shape and node.name in shape:
            var_shape = shape[node.name]
        else:
            var_shape = tensor_util.TensorShapeProtoToList(tensor_value.tensor_shape)
        param = None
        sym = [_expr.var(node.name, shape=var_shape, dtype="uint8")]
        return sym, param

    if len(np_array.shape) == 0:
        param = None
        sym = [tvm.relay.const(np_array, np_array.dtype)]
    else:
        param = tvm.nd.array(np_array)
        sym = [
            _expr.var(node.name, shape=param.shape, dtype=param.dtype)
        ]

    return sym, param


def get_attr(buf):
    """convert value of a node attribute. node attribute is part of a node in a graph.

        // tensorflow/core/framework/attr_value.proto
        message AttrValue {
            oneof value {
                bytes s = 2;                 // "string"
                int64 i = 3;                 // "int"
                float f = 4;                 // "float"
                bool b = 5;                  // "bool"
                DataType type = 6;           // "type"
                TensorShapeProto shape = 7;  // "shape"
                TensorProto tensor = 8;      // "tensor"
                ListValue list = 1;          // any "list(...)"            }
        }        

    Parameters
    ----------
    buf: attrvalue protobuf.  <class 'tensorflow.core.framework.attr_value_pb2.AttrValue'>

    Returns
    -------
    The value of the attr, as a Python object.

    """
    fields = ["s", "i", "f", "b", "type", "shape", "tensor", "func"]

    x = buf

    ret = []

    if not x.WhichOneof("value"):
        assert False # not yet tested; why would there be empty attribute value in a node def?

    if x.HasField("list"):
        for f in fields:
            if getattr(x.list, f):
                if f == "type":
                    ret += [dtypes.as_dtype(x) for x in list(getattr(x.list, f))]
                else:
                    ret += list(getattr(x.list, f))
    else:
        for f in fields:
            if x.HasField(f):
                if f == "type":
                    ret = dtypes.as_dtype(getattr(x, f))
                else:
                    ret = getattr(x, f)
    return ret
    
def parse_attr(attr_proto):
    """Convert node attributes (a serialized map of key-value pairs) in a node to a dict

    Parameters
    ----------
    attr_proto: <class 'google.protobuf.pyext._message.MessageMapContainer'>
    attributes of a tf node

    protobuf message format:
        // tensorflow/core/framework/node_def.proto
        message NodeDef {
            map<string, AttrValue> attr = 5;
        }

    Returns
    -------
    Dict {string: python object}


    Examples
    --------
    attributes in following node converted to {'_user_specified_name': b'x', 'dtype': tf.float32 }

        node {
        name: "x"
        op: "Placeholder"
        attr {
            key: "_user_specified_name"
            value {
            s: "x"
            }
        }
        attr {
            key: "dtype"
            value {
            type: DT_FLOAT
            }
        }


    """
    attrs = {}
    for key, value in attr_proto.items():
        attrs[key] = get_attr(value)

    return attrs

def convert_place_holder(shape, node, in_type=None):
    """ convert tf place holder into relay var. 
    
    Examples
    --------    
    a tf place holder with name "x" is converted to [Var(x, ty=TensorType([], float32))]

    """

    if shape and node.name in shape:
        input_shape = list(shape[node.name])
        # assert False # not yet tested
    else:
        input_shape = tensor_util.TensorShapeProtoToList(
            node.attr["shape"].shape
        )
        for idx, dim in enumerate(input_shape):
            if dim < 0:
                input_shape[idx] = Any()
    attr = parse_attr(node.attr)
    if in_type is not None:
        sym = [
            _expr.var(
                node.name, type_annotation=in_type
            )
        ]
    else:
        sym = [
            _expr.var(
                node.name, shape=input_shape, dtype=attr["dtype"].name
            )
        ]
    return input_shape, sym


