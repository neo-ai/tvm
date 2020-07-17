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
"""TIDL library supported operators.
"""
from topi.util import get_const_tuple
from tvm import relay
import tvm.ir
from tvm.relay.dataflow_pattern import is_op, is_constant, wildcard, is_tuple_get_item

def _merge_sequential_ops(mod):
    """Fuse sequential ops for op registration.
    """
    # Squeeze has to be followed by reshape.
    def _squeeze_reshape_pattern():
        squeeze_out = is_op('squeeze')(wildcard())
        reshape_out = is_op('reshape')(squeeze_out, wildcard())
        return reshape_out

    #tranpose has to be preceded and followed by reshape
    #TODO: add import of op 'transpose' and uncomment 2 patterns below
    #def _transpose_reshape_pattern():
    #    reshape_out1 = is_op('reshape')(wildcard(), wildcard())
    #    transpose_out = is_op('transpose')(reshape_out1)
    #    reshape_out2 = is_op('reshape')(transpose_out, wildcard())
    #    return reshape_out2

    #tranpose has to be followed by batch_flatten
    #def _transpose_batch_flatten_pattern():
    #    transpose_out = is_op('transpose')(wildcard())
    #    batch_flatten_out = is_op('nn.batch_flatten')(transpose_out)
    #    return batch_flatten_out

    #reshape has to be preceded by avg_pool2d, global_avg_pool2d, dense
    def _reshape_avg_pool_pattern():
        avg_pool_out = is_op('nn.avg_pool2d')(wildcard())
        reshape_out = is_op('reshape')(avg_pool_out, wildcard())
        return reshape_out

    def _reshape_global_avg_pool_pattern():
        global_avg_pool_out = is_op('nn.global_avg_pool2d')(wildcard())
        reshape_out = is_op('reshape')(global_avg_pool_out, wildcard())
        return reshape_out

    def _reshape_dense_pattern():
        dense_out = is_op('nn.dense')(wildcard(), is_constant())
        reshape_out = is_op('reshape')(dense_out, wildcard())
        return reshape_out

    #reshape has to be followed by softmax
    def _reshape_softmax_pattern():
        reshape_out = is_op('reshape')(wildcard(), wildcard())
        softmax_out = is_op('nn.softmax')(reshape_out)
        return softmax_out

    #relu has to be preceded by conv2d
    def _conv2d_relu_pattern():
        conv2d_out = is_op('nn.conv2d')(wildcard(), is_constant())
        relu_out = is_op('nn.relu')(conv2d_out)
        return relu_out

    #relu has to be preceded by conv2d and bias_add
    def _conv2d_bias_relu_pattern():
        conv2d_out = is_op('nn.conv2d')(wildcard(), is_constant())
        bias_out = is_op('nn.bias_add')(conv2d_out, is_constant())
        relu_out = is_op('nn.relu')(bias_out)
        return relu_out

    #relu has to be preceded by conv2d and bias_add
    def _conv2d_add_relu_pattern():
        conv2d_out = is_op('nn.conv2d')(wildcard(), is_constant())
        add_out = is_op('add')(conv2d_out, is_constant())
        relu_out = is_op('nn.relu')(add_out)
        return relu_out

    #bias_add has be preceded by conv2d
    def _conv2d_bias_pattern():
        conv2d_out = is_op('nn.conv2d')(wildcard(), is_constant())
        bias_out = is_op('nn.bias_add')(conv2d_out, is_constant())
        return bias_out

    #bias_add has be preceded by conv2d
    def _conv2d_add_pattern():
        conv2d_out = is_op('nn.conv2d')(wildcard(), is_constant())
        add_out = is_op('add')(conv2d_out, is_constant())
        return add_out

    #pad has be preceded by conv2d
    def _conv2d_pad_pattern():
        conv2d_out = is_op('nn.conv2d')(wildcard(), is_constant())
        pad_out = is_op('nn.pad')(conv2d_out)
        return pad_out

    #relu has to be preceded by batch_norm, add, dense
    def _bn_relu_pattern():
        bn_out = is_op('nn.batch_norm')(wildcard(), wildcard(), wildcard(), wildcard(), wildcard())
        tuple_get_item_node = is_tuple_get_item(bn_out, 0)
        relu_out = is_op('nn.relu')(tuple_get_item_node)
        return relu_out

    def _add_relu_pattern():
        add_out = is_op('add')(wildcard(), wildcard())
        relu_out = is_op('nn.relu')(add_out)
        return relu_out

    def _add_relu_checker(extract):
        add = extract.args[0]
        if any([isinstance(arg, tvm.relay.expr.Constant) for arg in add.args]):
            # Can't add constant unless used like bias_add in a pattern such as "conv2d_add_relu".
            return False
        return True

    def _dense_relu_pattern():
        dense_out = is_op('nn.dense')(wildcard(), is_constant())
        relu_out = is_op('nn.relu')(dense_out)
        return relu_out

    #relu has to be preceded by dense and bias_add
    def _dense_bias_relu_pattern():
        dense_out = is_op('nn.dense')(wildcard(), is_constant())
        bias_out = is_op('nn.bias_add')(dense_out, is_constant())
        relu_out = is_op('nn.relu')(bias_out)
        return relu_out

    #relu has to be preceded by dense and bias_add
    def _dense_add_relu_pattern():
        dense_out = is_op('nn.dense')(wildcard(), is_constant())
        add_out = is_op('add')(dense_out, is_constant())
        relu_out = is_op('nn.relu')(add_out)
        return relu_out

    #bias_add has to be preceded by dense
    def _dense_bias_pattern():
        dense_out = is_op('nn.dense')(wildcard(), is_constant())
        bias_out = is_op('nn.bias_add')(dense_out, is_constant())
        return bias_out

    #bias_add has to be preceded by dense
    def _dense_add_pattern():
        dense_out = is_op('nn.dense')(wildcard(), is_constant())
        add_out = is_op('add')(dense_out, is_constant())
        return add_out

    pattern_table = [
        ('tidl.squeeze_reshape', _squeeze_reshape_pattern()),
        #TODO: add import of op 'transpose' and uncomment 2 items below
        #('tidl.transpose_reshape', _transpose_reshape_pattern()),
        #('tidl.tanspose_batch_flatten', _transpose_batch_flatten_pattern()),
        ('tidl.reshape_avgpool', _reshape_avg_pool_pattern()),
        ('tidl.reshape_globalavgpool', _reshape_global_avg_pool_pattern()),
        ('tidl.reshape_dense', _reshape_dense_pattern()),
        ('tidl.reshape_softmax', _reshape_softmax_pattern()),
        ('tidl.conv2d_relu', _conv2d_relu_pattern()),
        ('tidl.conv2d_bias_relu', _conv2d_bias_relu_pattern()),
        ('tidl.conv2d_add_relu', _conv2d_add_relu_pattern()),
        ('tidl.conv2d_bias', _conv2d_bias_pattern()),
        ('tidl.conv2d_add', _conv2d_add_pattern()),
        ('tidl.conv2d_pad', _conv2d_pad_pattern()),
        ('tidl.bn_relu', _bn_relu_pattern()),
        ('tidl.add_relu', _add_relu_pattern(), _add_relu_checker),
        ('tidl.dense_relu', _dense_relu_pattern()),
        ('tidl.dense_bias_relu', _dense_bias_relu_pattern()),
        ('tidl.dense_add_relu', _dense_add_relu_pattern()),
        ('tidl.dense_bias', _dense_bias_pattern()),
        ('tidl.dense_add', _dense_add_pattern()),
    ]

    return relay.transform.MergeComposite(pattern_table)(mod)

@tvm.ir.register_op_attr("tidl.squeeze_reshape", "target.tidl")
def _tidl_squeeze_reshape_whitelist_fn(attrs, args):
    return True

#TODO: add import of op 'transpose' and uncomment 2 functions below
#@tvm.ir.register_op_attr("tidl.transpose_reshape", "target.tidl")
#def _tidl_transpose_reshape_whitelist_fn(attrs, args):
#    return True

#@tvm.ir.register_op_attr("tidl.tanspose_batch_flatten", "target.tidl")
#def _tidl_transpose_batch_flatten_whitelist_fn(attrs, args):
#    return True

@tvm.ir.register_op_attr("tidl.reshape_avgpool", "target.tidl")
def _tidl_reshape_avgpool_whitelist_fn(attrs, args):
    return _avg_pool_whitelist_fn(attrs, args)

@tvm.ir.register_op_attr("tidl.reshape_globalavgpool", "target.tidl")
def _tidl_reshape_globalavgpool_whitelist_fn(attrs, args):
    return _global_avg_pool_whitelist_fn(attrs, args)

@tvm.ir.register_op_attr("tidl.reshape_dense", "target.tidl")
def _tidl_reshape_dense_whitelist_fn(attrs, args):
    return _dense_whitelist_fn(attrs, args)

@tvm.ir.register_op_attr("tidl.reshape_softmax", "target.tidl")
def _tidl_reshape_softmax_whitelist_fn(attrs, args):
    return True

@tvm.ir.register_op_attr("tidl.conv2d_relu", "target.tidl")
def _conv2d_relu_whitelist_fn(attrs, args):
    conv2d_op = args[0]
    return _conv2d_whitelist_fn(conv2d_op.attrs, conv2d_op.args)

@tvm.ir.register_op_attr("tidl.conv2d_bias_relu", "target.tidl")
def _conv2d_bias_relu_whitelist_fn(attrs, args):
    conv2d_op = args[0].args[0]
    return _conv2d_whitelist_fn(conv2d_op.attrs, conv2d_op.args)

@tvm.ir.register_op_attr("tidl.conv2d_add_relu", "target.tidl")
def _conv2d_add_relu_whitelist_fn(attrs, args):
    conv2d_op = args[0].args[0]
    return _conv2d_whitelist_fn(conv2d_op.attrs, conv2d_op.args)

@tvm.ir.register_op_attr("tidl.bn_relu", "target.tidl")
def _bn_relu_whitelist_fn(attrs, args):
    bn_op = args[0]
    return _batch_norm_whitelist_fn(bn_op.attrs, bn_op.args)

@tvm.ir.register_op_attr("tidl.add_relu", "target.tidl")
def _add_relu_whitelist_fn(attrs, args):
    add_op = args[0]
    return _add_whitelist_fn(add_op.attrs, add_op.args)

@tvm.ir.register_op_attr("tidl.dense_relu", "target.tidl")
def _dense_relu_whitelist_fn(attrs, args):
    dense_op = args[0]
    return _dense_whitelist_fn(dense_op.attrs, dense_op.args)

@tvm.ir.register_op_attr("tidl.dense_bias_relu", "target.tidl")
def _dense_bias_relu_whitelist_fn(attrs, args):
    dense_op = args[0].args[0]
    return _dense_whitelist_fn(dense_op.attrs, dense_op.args)

@tvm.ir.register_op_attr("tidl.dense_add_relu", "target.tidl")
def _dense_add_relu_whitelist_fn(attrs, args):
    dense_op = args[0].args[0]
    return _dense_whitelist_fn(dense_op.attrs, dense_op.args)

@tvm.ir.register_op_attr("tidl.conv2d_bias", "target.tidl")
def _conv2d_bias_whitelist_fn(attrs, args):
    conv2d_op = args[0]
    return _conv2d_whitelist_fn(conv2d_op.attrs, conv2d_op.args)

@tvm.ir.register_op_attr("tidl.conv2d_add", "target.tidl")
def _conv2d_add_whitelist_fn(attrs, args):
    conv2d_op = args[0]
    return _conv2d_whitelist_fn(conv2d_op.attrs, conv2d_op.args)

@tvm.ir.register_op_attr("tidl.dense_bias", "target.tidl")
def _dense_bias_whitelist_fn(attrs, args):
    dense_op = args[0]
    return _dense_whitelist_fn(dense_op.attrs, dense_op.args)

@tvm.ir.register_op_attr("tidl.dense_add", "target.tidl")
def _dense_add_whitelist_fn(attrs, args):
    dense_op = args[0]
    return _dense_whitelist_fn(dense_op.attrs, dense_op.args)

@tvm.ir.register_op_attr("tidl.conv2d_pad", "target.tidl")
def _conv2d_pad_whitelist_fn(attrs, args):
    conv2d_op = args[0]
    pad_supported = (float(attrs.pad_value) == 0.0 and attrs.pad_mode == 'constant')
    conv2d_supported = _conv2d_whitelist_fn(conv2d_op.attrs, conv2d_op.args)
    supported = pad_supported and conv2d_supported
    return supported

@tvm.ir.register_op_attr("add", "target.tidl")
def _add_whitelist_fn(attrs, args):
    if any([isinstance(arg, tvm.relay.expr.Constant) for arg in args]):
        # Can't add constant unless used like bias_add in a pattern such as "conv2d_add_relu".
        return False
    supported = True
    return supported

@tvm.ir.register_op_attr("nn.argmax", "target.tidl")
def _argmax_whitelist_fn(attrs, args):
    keepdims = attrs.keepdims
    exclude = attrs.exclude
    axis = attrs.axis
    data = args[0]
    data_shape = data.checked_type.shape
    supported = (int(data_shape[1]) <= 15 and keepdims == 1 and axis == 1 and exclude == 0)
    return supported

@tvm.ir.register_op_attr("nn.avg_pool2d", "target.tidl")
def _avg_pool_whitelist_fn(attrs, args):
    pool_size = get_const_tuple(attrs.pool_size)
    strides = get_const_tuple(attrs.strides)
    supported = (pool_size[0] <= 9 and pool_size[1] <= 9 and strides[0] <= 3 and strides[1] <= 2)
    return supported

@tvm.ir.register_op_attr("nn.batch_flatten", "target.tidl")
def _batch_flatten_fn(attrs, args):
    data = args[0]
    data_shape = data.checked_type.shape
    if len(data_shape) == 4:
        supported = (int(data_shape[2]) <= 65535 and int(data_shape[3]) <= 65535)
    else:
        supported = True
    return supported

@tvm.ir.register_op_attr("nn.batch_norm", "target.tidl")
def _batch_norm_whitelist_fn(attrs, args):
    data1 = args[1]
    if data1.checked_type.dtype != 'float32':
        supported = False
    elif attrs.axis != 1 and attrs.axis != 3:
        supported = False
    else:
        supported = True
    return supported

@tvm.ir.register_op_attr("nn.bias_add", "target.tidl")
def _bias_add_whitelist_fn(attrs, args):
    # Standalone bias_add is not supported.
    return False

@tvm.ir.register_op_attr("clip", "target.tidl")
def _clip_whitelist_fn(attrs, args):
    a_min = attrs.a_min
    a_max = attrs.a_max
    supported = (a_min == 0 and a_max == 6)
    return supported

@tvm.ir.register_op_attr("concatenate", "target.tidl")
def _concatenate_whitelist_fn(attrs, args):
    supported = (attrs.axis == 1) or (attrs.axis == 3)
    return supported

def get_conv2d_num_channels(kernel_layout, weight_shape):
    """ Get number of input and output channels of conv2d """
    if kernel_layout == 'OIHW':
        (num_in_channels, num_out_channels) = (weight_shape[1], weight_shape[0])
    elif kernel_layout == 'HWIO':
        (num_in_channels, num_out_channels) = (weight_shape[2], weight_shape[3])
    else: # 'HWOI'
        (num_in_channels, num_out_channels) = (weight_shape[3], weight_shape[2])
    return (num_in_channels, num_out_channels)

@tvm.ir.register_op_attr("nn.conv2d", "target.tidl")
def _conv2d_whitelist_fn(attrs, args):
    weight = args[1]
    if weight.checked_type.dtype != 'float32':
        return False
    if attrs.kernel_layout not in ('OIHW', 'HWIO', 'HWOI'):
        return False

    weight_shape = weight.data.shape
    strides = get_const_tuple(attrs.strides)
    dilation = get_const_tuple(attrs.dilation)
    kernel_size = get_const_tuple(attrs.kernel_size)
    groups = attrs.groups

    (dh, dw) = dilation
    (kh, kw) = kernel_size
    (num_in_chs, num_out_chs) = get_conv2d_num_channels(attrs.kernel_layout, weight_shape)
    channel_supported = (num_in_chs <= 2048 and num_out_chs <= 2048)
    stride_supported = (strides[0] <= 2 and strides[1] <= 2)
    dilation_supported = (dh in (1, 2, 4)) and (dw in (1, 2, 4))
    kernel_supported = (((kh-1)*dh+1) <= 9) and (((kw-1)*dw+1) <= 9)
    groups_supported = (groups <= 1024)
    supported = channel_supported and stride_supported and dilation_supported \
                and kernel_supported and groups_supported
    return supported

@tvm.ir.register_op_attr("nn.conv2d_transpose", "target.tidl")
def _conv2d_transpose_whitelist_fn(attrs, args):
    if attrs.kernel_layout not in ('OIHW', 'HWIO', 'HWOI'):
        return False
    weight = args[1]
    weight_shape = weight.data.shape
    strides = get_const_tuple(attrs.strides)
    groups = attrs.groups
    (num_in_chs, num_out_chs) = get_conv2d_num_channels(attrs.kernel_layout, weight_shape)
    supported = (num_in_chs == num_out_chs) and (num_in_chs == groups) and (strides[1] == 2)
    return supported

@tvm.ir.register_op_attr("nn.dense", "target.tidl")
def _dense_whitelist_fn(attrs, args):
    weight = args[1]

    weight_shape = weight.data.shape
    w_in = weight_shape[1]
    w_out = weight_shape[0]
    supported = (w_in <= 65536) and (w_out <= 16384) and (w_in * w_out <= 67108864)
    return supported

@tvm.ir.register_op_attr("nn.dropout", "target.tidl")
def _dropout_whitelist_fn(attrs, args):
    supported = True
    return supported

@tvm.ir.register_op_attr("nn.global_avg_pool2d", "target.tidl")
def _global_avg_pool_whitelist_fn(attrs, args):
    shape = list(map(int, args[0].checked_type.shape))
    layout = attrs.layout
    if layout == "NCHW":
        height = shape[2]
        width = shape[3]
    else:
        height = shape[1]
        width = shape[2]
    supported = height * width <= 4096
    return supported

@tvm.ir.register_op_attr("nn.max_pool2d", "target.tidl")
def _max_pool_whitelist_fn(attrs, args):
    pool_size = get_const_tuple(attrs.pool_size)
    strides = get_const_tuple(attrs.strides)
    supported = (pool_size[0] <= 9) and (pool_size[1] <= 9) and (strides[0] <= 3) \
                and (strides[1] <= 2)
    return supported

@tvm.ir.register_op_attr("multiply", "target.tidl")
def _multiply_whitelist_fn(attrs, args):
    #supported = True
    supported = False
    return supported

@tvm.ir.register_op_attr("nn.nms", "target.tidl")
def _nms_whitelist_fn(attrs, args):
    supported = True
    return supported

@tvm.ir.register_op_attr("nn.pad", "target.tidl")
def _pad_whitelist_fn(attrs, args):
    # Standalone pad is not supported.
    return False

@tvm.ir.register_op_attr("nn.relu", "target.tidl")
def _relu_whitelist_fn(attrs, args):
    # Standalone relu is not supported.
    return False

@tvm.ir.register_op_attr("nn.slice_like", "target.tidl")
def _slice_like_whitelist_fn(attrs, args):
    #supported = (attrs.axis == 1)
    supported = False
    return supported

@tvm.ir.register_op_attr("nn.softmax", "target.tidl")
def _softmax_whitelist_fn(attrs, args):
    supported = (attrs.axis != 2)
    return supported
