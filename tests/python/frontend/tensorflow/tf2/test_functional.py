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
# pylint: disable=import-self, invalid-name, unused-argument, too-many-lines, len-as-condition, broad-except
# pylint: disable=import-outside-toplevel, redefined-builtin
"""TF2 to relay converter test: x+1 as a very basic example"""

import tempfile
import tensorflow as tf
import numpy as np
import pytest
import tvm.testing
from tvm import relay
try:
    from .test_common import compile, compile_vm, compare_tf_tvm, sorted_same, run
    from .tf_helper import run_tf_code
except:
    from test_common import compile, compile_vm, compare_tf_tvm, sorted_same, run
    from tf_helper import run_tf_code


class StridedSlice(tf.Module):
    def get_input(self):
        return np.ones((3,2,3), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'Const', 'StridedSlice', 'Identity']

    def expected_lib_ops(self):
        return ['Const', 'StridedSlice', 'Identity']

    """scalar as input"""

    @tf.function(input_signature=[tf.TensorSpec(shape=(3,2,3), dtype=tf.float32)])
    def func(self, x):
        return tf.strided_slice(x, [1, 0, 0], [2, 1, 3], [1, 1, 1])


class Shape(tf.Module):
    def get_input(self):
        return np.ones((3,2,3), dtype=np.float32)

    def expected_ops(self):
        return ['Const', 'Identity', 'Placeholder', 'AddV2', 'Shape', 'Fill']

    def expected_lib_ops(self):
        return ['AddV2', 'Const', 'Identity', 'Shape', 'Fill']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(3, 2, 3), dtype=tf.float32)])
    def func(self, x):
        a = tf.ones_like(tf.raw_ops.Shape(input=x), dtype=tf.float32)
        return a + x


class PackModel(tf.Module):
    def get_input(self):
        return np.ones((2,3), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'Pack', 'Identity']

    def expected_lib_ops(self):
        return ['Pack', 'Identity']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3), dtype=tf.float32)])
    def func(self, x):
        return tf.raw_ops.Pack(values=[x, x], axis=0)


class SplitModel(tf.Module):
    def get_input(self):
        return np.ones((1,30), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'Const', 'Split', 'Pack', 'Identity']

    def expected_lib_ops(self):
        return ['Split', 'Identity', 'Pack', 'Const']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        a,b,c = tf.split(x, 3, axis=1)
        return tf.raw_ops.Pack(values=[a,b,c], axis=1)


class Maximum(tf.Module):
    def get_input(self):
        return np.ones((1,30), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'Maximum', 'Split', 'Const', 'Identity']

    def expected_lib_ops(self):
        return ['Split', 'Identity', 'Maximum', 'Const']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        a,b = tf.split(x, 2, axis=1)
        return tf.math.maximum(a, b, name=None)


class Less(tf.Module):
    def get_input(self):
        return np.ones((1,30), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'Less', 'Split', 'Const', 'Identity']

    def expected_lib_ops(self):
        return ['Split', 'Identity', 'Less', 'Const']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        a,b = tf.split(x, 2, axis=1)
        return tf.math.less(a, b, name=None)


class Equal(tf.Module):
    def get_input(self):
        return np.ones((1,30), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'Equal', 'Split', 'Const', 'Identity']

    def expected_lib_ops(self):
        return ['Split', 'Identity', 'Equal', 'Const']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        a,b = tf.split(x, 2, axis=1)
        return tf.math.equal(a, b, name=None)



class Cast(tf.Module):
    def get_input(self):
        return np.ones((1,30), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'Cast', 'Identity']

    def expected_lib_ops(self):
        return ['Cast', 'Identity']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        return tf.cast(x, tf.int32)


class ExpandDims(tf.Module):
    def get_input(self):
        return np.ones((1,30), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'ExpandDims', 'Identity', 'Const']

    def expected_lib_ops(self):
        return ['ExpandDims', 'Identity', 'Const']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        return tf.expand_dims(x, axis=2)


class Transpose(tf.Module):
    def get_input(self):
        return np.ones((1,30), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'ExpandDims', 'Identity', 'Const', 'Transpose']

    def expected_lib_ops(self):
        return ['ExpandDims', 'Identity', 'Const', 'Transpose']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        x = tf.expand_dims(x, axis=2)
        return tf.transpose(x, perm=[0,2,1])



class Reshape(tf.Module):
    def get_input(self):
        return np.ones((1,30), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'Reshape', 'Identity', 'Const']

    def expected_lib_ops(self):
        return ['Identity', 'Const', 'Reshape']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        return tf.reshape(x, (1,2,15))


class Tanh(tf.Module):
    def get_input(self):
        return np.ones((1,30), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'Tanh', 'Identity']

    def expected_lib_ops(self):
        return ['Tanh', 'Identity']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        return tf.math.tanh(x)


class Sigmoid(tf.Module):
    def get_input(self):
        return np.ones((1,30), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'Sigmoid', 'Identity']

    def expected_lib_ops(self):
        return ['Sigmoid', 'Identity']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        return tf.math.sigmoid(x)


class Relu(tf.Module):
    def get_input(self):
        return np.ones((1,30), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'Relu', 'Identity']

    def expected_lib_ops(self):
        return ['Relu', 'Identity']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        return tf.nn.relu(x)



class Floor(tf.Module):
    def get_input(self):
        return np.ones((1,30), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'Floor', 'Identity']

    def expected_lib_ops(self):
        return ['Floor', 'Identity']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        return tf.math.floor(x)


class FloorMod(tf.Module):
    def get_input(self):
        return np.ones((1,30), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'FloorMod', 'Identity', 'Split', 'Const']

    def expected_lib_ops(self):
        return ['FloorMod', 'Identity', 'Split', 'Const']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        a, b = tf.split(x, 2, axis=1)
        return tf.math.floormod(a, b)

class ConcatV2(tf.Module):
    def get_input(self):
        return np.ones((1,30), dtype=np.float32)

    def expected_ops(self):
        return ['Placeholder', 'Const', 'Split', 'ConcatV2', 'Identity']

    def expected_lib_ops(self):
        return ['Split', 'Identity', 'ConcatV2', 'Const']

    """scalar as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        a,b,c = tf.split(x, 3, axis=1)
        return tf.raw_ops.ConcatV2(values=[a, b, c], axis=1)

class AddOne(tf.Module):

    def get_input(self):
        return np.array(1.0, dtype='float32')

    def expected_ops(self):
        return ['Placeholder', 'Const', 'AddV2', 'Identity']

    def expected_lib_ops(self):
        return ['Const', 'AddV2', 'Identity']

    """scalar as input"""

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
    def func(self, x):
        return x + 1


class AddOne2D(AddOne):

    def get_input(self):
        return np.ones((2, 2), dtype='float32')

    """2D array as input"""

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
    def func(self, x):
        return x + 1


class AddOne2DConstant(AddOne):

    def get_input(self):
        return np.ones((2, 2), dtype='float32')

    """2D array as input with 2D constant as well; 2D constant stored in params after convert"""

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
    def func(self, x):
        return x + np.ones((2, 2), dtype='float32')


class SubOne(tf.Module):

    def get_input(self):
        return np.array(1.0, dtype='float32')

    def expected_ops(self):
        return ['Placeholder', 'Const', 'Sub', 'Identity']

    def expected_lib_ops(self):
        return ['Const', 'Sub', 'Identity']

    """scalar as input"""

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
    def func(self, x):
        return x - 1


class SubOne2D(SubOne):

    def get_input(self):
        return np.ones((2, 2), dtype='float32')

    """2D array as input"""

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
    def func(self, x):
        return x - 1


class SubOne2DConstant(SubOne):

    def get_input(self):
        return np.ones((2, 2), dtype='float32')

    """2D array as input with 2D constant as well; 2D constant stored in params after convert"""

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
    def func(self, x):
        return x - np.ones((2, 2), dtype='float32')


class MulOne(tf.Module):

    def get_input(self):
        return np.array(1.0, dtype='float32')

    def expected_ops(self):
        return ['Placeholder', 'Const', 'Mul', 'Identity']

    def expected_lib_ops(self):
        return ['Const', 'Mul', 'Identity']

    """scalar as input"""

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
    def func(self, x):
        return x * 1


class MulOne2D(MulOne):

    def get_input(self):
        return np.ones((2, 2), dtype='float32')

    """2D array as input"""

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
    def func(self, x):
        return x * 1


class MulOne2D_constant(MulOne):

    def get_input(self):
        return np.ones((2, 2), dtype='float32')

    """2D array as input with 2D constant as well; 2D constant stored in params after convert"""

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
    def func(self, x):
        return x * np.ones((2, 2), dtype='float32')


class DivOne(tf.Module):

    def get_input(self):
        return np.array(1.0, dtype='float32')

    def expected_ops(self):
        return ['Placeholder', 'Const', 'RealDiv', 'Identity']

    def expected_lib_ops(self):
        return ['Const', 'RealDiv', 'Identity']

    """scalar as input"""

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
    def func(self, x):
        return x / 1


class DivOne2D(DivOne):

    def get_input(self):
        return np.ones((2, 2), dtype='float32')

    """2D array as input"""

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
    def func(self, x):
        return x / 1


class DivOne2DConstant(DivOne):

    def get_input(self):
        return np.ones((2, 2), dtype='float32')

    """2D array as input with 2D constant as well; 2D constant stored in params after convert"""

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
    def func(self, x):
        return x / np.ones((2, 2), dtype='float32')


class TensorList(tf.Module):
    def get_input(self):
        in_tens = np.ones((2, 3), dtype='float32')
        in_tens[1,:] = np.zeros((3,), dtype='float32')
        return in_tens

    def expected_ops(self):
        return ['Placeholder', 'TensorListSetItem', 'TensorListReserve', 'StridedSlice', 'TensorListGetItem', 'Identity', 'Const']

    def expected_lib_ops(self):
        return ['TensorListReserve', 'TensorListSetItem', 'TensorListGetItem', 'Const', 'StridedSlice', 'Identity']

    """2D array as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3), dtype=tf.float32)])
    def func(self, x):
        elem_shape = (3,)
        dtype = tf.float32
        tl = tf.raw_ops.TensorListReserve(element_shape=elem_shape, num_elements=2, element_dtype=dtype)
        tl = tf.raw_ops.TensorListSetItem(input_handle=tl, index=0, item=x[0,:])
        tl = tf.raw_ops.TensorListSetItem(input_handle=tl, index=1, item=x[1,:])
        output = tf.raw_ops.TensorListGetItem(input_handle=tl, index=0, element_shape=elem_shape, element_dtype=dtype)
        return output


class TensorList2D(TensorList):
    def get_input(self):
        in_tens = np.ones((2, 3, 4), dtype='float32')
        in_tens[1,:,:] = np.zeros((3,4), dtype='float32')
        return in_tens

    """2D array as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3, 4), dtype=tf.float32)])
    def func(self, x):
        elem_shape = (3, 4)
        dtype = tf.float32
        tl = tf.raw_ops.TensorListReserve(element_shape=elem_shape, num_elements=2, element_dtype=dtype)
        tl = tf.raw_ops.TensorListSetItem(input_handle=tl, index=0, item=x[0,:,:])
        tl = tf.raw_ops.TensorListSetItem(input_handle=tl, index=1, item=x[1,:,:])
        output = tf.raw_ops.TensorListGetItem(input_handle=tl, index=0, element_shape=elem_shape, element_dtype=dtype)
        return output


class TensorListStack(tf.Module):
    def get_input(self):
        in_tens = np.ones((2, 3), dtype='float32')
        in_tens[1] = np.zeros((3,), dtype='float32')
        return in_tens

    def expected_ops(self):
        return ['Const', 'TensorListFromTensor', 'Identity', 'TensorListStack', 'Placeholder', 'TensorListReserve']

    def expected_lib_ops(self):
        return ['Const', 'TensorListFromTensor', 'TensorListStack', 'Identity', 'TensorListReserve']

    """2D array as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3), dtype=tf.float32)])
    def func(self, x):
        elem_shape = (3,)
        dtype = tf.float32
        tl = tf.raw_ops.TensorListReserve(element_shape=elem_shape, num_elements=2, element_dtype=dtype)
        tl = tf.raw_ops.TensorListFromTensor(tensor=x, element_shape=elem_shape)
        output = tf.raw_ops.TensorListStack(input_handle=tl, element_shape=elem_shape, element_dtype=dtype)
        return output


class TensorListStack2D(TensorListStack):
    def get_input(self):
        in_tens = np.ones((2, 3, 4), dtype='float32')
        in_tens[1,:,:] = np.zeros((3,4), dtype='float32')
        return in_tens

    """2D array as input"""
    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3, 4), dtype=tf.float32)])
    def func(self, x):
        elem_shape = (3, 4)
        dtype = tf.float32
        tl = tf.raw_ops.TensorListReserve(element_shape=elem_shape, num_elements=2, element_dtype=dtype)
        tl = tf.raw_ops.TensorListFromTensor(tensor=x, element_shape=elem_shape)
        output = tf.raw_ops.TensorListStack(input_handle=tl, element_shape=elem_shape, element_dtype=dtype)
        return output


class If(tf.Module):
    def get_input(self):
        return np.ones((2,2), dtype='float32')

    def expected_ops(self):
        return ['Placeholder', 'If', 'Identity', 'Const']

    def expected_lib_ops(self):
        return ['If', 'Identity', 'Const', 'Mul']

    @tf.function(input_signature=[tf.TensorSpec(shape=(2,2), dtype=tf.float32)])
    def func(self, x):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2,2), dtype=tf.float32)])
        def double(x):
            return 2*x

        @tf.function(input_signature=[tf.TensorSpec(shape=(2,2), dtype=tf.float32)])
        def triple(x):
            return 3*x

        cond = True
        output = tf.raw_ops.If(cond=cond, input=[x], Tout=[tf.float32], output_shapes=[(2,2)],
            then_branch=double.get_concrete_function(), else_branch=triple.get_concrete_function())
        return output[0]


class StatelessWhile(tf.Module):
    def get_input(self):
        return np.array([6], dtype='float32')

    def expected_ops(self):
        return ['Identity', 'StatelessWhile', 'Const', 'Placeholder']

    def expected_lib_ops(self):
        return ['StatelessWhile', 'Squeeze', 'Const', 'Less', 'Add', 'AddV2', 'Identity']

    @tf.function(input_signature=[tf.TensorSpec(shape=(1,), dtype=tf.float32)])
    def func(self, x):
        i = tf.constant(3.)
        cond = lambda i: tf.less(i, x)
        body = lambda i: (tf.add(i, 2),)
        r = tf.while_loop(cond, body, [i])
        return r[0]


class StatelessWhile2Var(StatelessWhile):
    def get_input(self):
        return np.array([20], dtype='float32')

    @tf.function(input_signature=[tf.TensorSpec(shape=(1,), dtype=tf.float32)])
    def func(self, x):
        i = tf.constant(3.)
        j = tf.constant(5.)
        cond = lambda i,j: tf.less(i+j, x)
        body = lambda i,j: (tf.add(i, 2), tf.add(j, 3))
        r = tf.while_loop(cond, body, [i, j])
        return r


class MultiOutput(tf.Module):
    def get_input(self):
        return np.ones((2,2), dtype='float32')

    def expected_ops(self):
        # this case expected to fail
        return ['Identity', 'Mul', 'Const', 'Placeholder']

    def expected_lib_ops(self):
        return ['Identity', 'Const', 'Mul']

    @tf.function(input_signature=[tf.TensorSpec(shape=(2,2), dtype=tf.float32)])
    def func(self, x):
        y = 2*x
        return x, y



def _function_graph(TestClass):
    f = TestClass().func
    gdef = f.get_concrete_function().graph.as_graph_def()
    expect_ops = TestClass().expected_ops()
    gdef_ops = list(set([n.op for n in gdef.node]))
    assert sorted_same(expect_ops, gdef_ops), print("Expected ops:: {}".format(gdef_ops))
    input = TestClass().get_input()
    output = run_tf_code(f, input)
    return gdef, input, output


def _model_graph(TestClass):
    model = TestClass()
    with tempfile.TemporaryDirectory() as model_path:
        tf.saved_model.save(model, model_path)
        imported = tf.saved_model.load(model_path)

    f = imported.signatures["serving_default"]
    gdef = f.graph.as_graph_def(add_shapes=True)
    gdef_ops = set([n.op for n in gdef.node])
    expect_ops = set(['Placeholder', 'PartitionedCall', 'StatefulPartitionedCall', 'Identity'])
    assert gdef_ops.issubset(expect_ops)

    lib_ops = set()
    for fn in gdef.library.function:
        for n in fn.node_def:
            lib_ops.add(n.op)
    lib_ops = list(lib_ops)
    expect_lib_ops = TestClass().expected_lib_ops()

    assert sorted_same(expect_lib_ops, lib_ops), print("Expected ops:: {}".format(lib_ops))
    input = model.get_input()
    output = run_tf_code(f, input)

    return gdef, input, output


def run_func_graph(TestClass, use_vm=False):
    def _function_params():
        return _function_graph(TestClass)

    run(_function_params, vm=use_vm)

def run_model_graph(TestClass, output_sig=None):
    def _model_params():
        return _model_graph(TestClass)

    run(_model_params, vm=True, output_sig=output_sig)

def run_all(TestClass):
    run_model_graph(TestClass)
    for use_vm in [True, False]:
        run_func_graph(TestClass, use_vm=use_vm)


def test_add():
    run_all(AddOne)
    run_all(AddOne2D)
    run_all(AddOne2DConstant)

def test_sub():
    run_all(SubOne)
    run_all(SubOne2D)
    run_all(SubOne2DConstant)

def test_mul():
    run_all(MulOne)
    run_all(MulOne2D)
    run_all(MulOne2D_constant)

def test_div():
    run_all(DivOne)
    run_all(DivOne2D)
    run_all(DivOne2DConstant)

def test_strided_slice():
    run_all(StridedSlice)

def test_shape():
    run_all(Shape)

def test_pack():
    run_all(PackModel)

def test_split():
    run_all(SplitModel)

def test_max():
    run_all(Maximum)

def test_less():
    run_all(Less)

def test_equal():
    run_all(Equal)

def test_floor():
    run_all(Floor)
    run_all(FloorMod)

def test_concat_v2():
    run_all(ConcatV2)

def test_cast():
    run_all(Cast)

def test_expand_dims():
    run_all(ExpandDims)

def test_transpose():
    run_all(Transpose)

def test_reshape():
    run_all(Reshape)
    
def test_tanh():
    run_all(Tanh)

def test_sigmoid():
    run_all(Sigmoid)

def test_relu():
    run_all(Relu)

def test_tensorlist():
    run_model_graph(TensorList)
    run_func_graph(TensorList, use_vm=True)

def test_tensorlist_stack():
    run_model_graph(TensorListStack)
    run_func_graph(TensorListStack, use_vm=True)

def test_tensorlist_2d():
    run_model_graph(TensorList2D)
    run_func_graph(TensorList2D, use_vm=True)

def test_tensorlist_stack_2d():
    run_model_graph(TensorListStack2D)
    run_func_graph(TensorListStack2D, use_vm=True)

def test_if():
    run_model_graph(If)
    run_func_graph(If, use_vm=True)

def test_multi_output():
    run_model_graph(MultiOutput)

def test_stateless_while():
    run_model_graph(StatelessWhile)

def test_stateless_while_2var():
    run_model_graph(StatelessWhile2Var)

if __name__ == "__main__":
    test_add()
    test_sub()
    test_mul()
    test_div()
    test_strided_slice()
    test_shape()
    test_pack()
    test_split()
    test_max()
    test_less()
    test_equal()
    test_floor()
    test_concat_v2()
    test_cast()
    test_expand_dims()
    test_transpose()
    test_reshape()
    test_tanh()
    test_sigmoid()
    test_relu()
    test_tensorlist()
    test_tensorlist_stack()
    test_tensorlist_2d()
    test_tensorlist_stack_2d()
    test_if()
    test_multi_output()
    test_stateless_while()
    test_stateless_while_2var()
