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
"""TF2 to relay converter test utilities"""

import tvm
from tvm import relay

from tvm.runtime.vm import VirtualMachine
from tvm.contrib import graph_runtime
from tvm.relay.frontend.tensorflow import from_tensorflow
import tvm.testing

def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        out = o.asnumpy().tolist()
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.append(vmobj_to_list(f))
        out = result
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))
    return out

def compile(gdef, target = "llvm", target_host = "llvm", ctx = tvm.cpu(0),
            opt_level=3, output_sig=None):
    mod, params = from_tensorflow(gdef, outputs=output_sig)
    with tvm.transform.PassContext(opt_level):
        lib = relay.build(mod, target=target, target_host=target_host, params=params)
    return graph_runtime.GraphModule(lib["default"](ctx))

def compile_vm(gdef, target = "llvm", target_host = "llvm",
               ctx = tvm.cpu(0), opt_level=3,
               disabled_pass=None, output_sig=None):
    mod, params = from_tensorflow(gdef, outputs=output_sig)
    with tvm.transform.PassContext(opt_level, disabled_pass=disabled_pass):
        mod = relay.transform.InferType()(mod)
        vm_exec = relay.vm.compile(mod, target, target_host, params=params)

    vm = VirtualMachine(vm_exec, ctx)
    return vm

def run_vm(mod, input_):
    if type(mod) is VirtualMachine:
        b = mod.invoke("main", input_)
        b = vmobj_to_list(b)
    else:
        mod.set_input(0, input_)
        mod.run()
        b = mod.get_output(0).asnumpy()
    return b

def assert_allclose_array(arr1, arr2, atol=1e-3):
    assert len(arr1) == len(arr2)
    for val1, val2 in zip(arr1, arr2):
        tvm.testing.assert_allclose(val1, val2, atol=atol)

def compare_tf_tvm(mod, input_, out):
    """ compare tf and tvm execution for the same input.

    Parameters
    ----------
    func: tf function. can be from saved model or not. different ways to pass input
        from saved model: <class 'tensorflow.python.saved_model.load._WrapperFunction'>
        not from saved model:  <class 'tensorflow.python.eager.def_function.Function'>

    mod: compiled relay module (vm or graph runtime). converted from tf func.

    input_: a single numpy array object

    """

    b = run_vm(mod, input_)

    # print(f'TF: {out}')
    # print(f'TVM: {b}')
    tvm.testing.assert_allclose(out, b, atol=1e-5)
    print("Compare TVM/TF: Passed!")

def sorted_same (a, b):
    a = list(set(a))
    b = list(set(b))
    return sorted(a) == sorted(b)

def run(x, vm=True, output_sig=None):
    gdef, input_, output_ = x()
    if vm:
        m = compile_vm(gdef, output_sig=output_sig)
    else:
        m = compile(gdef, output_sig=output_sig)
    compare_tf_tvm(m, input_, output_)
