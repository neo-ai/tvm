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
"""TIDL backend compiler"""

import os
import sys
import subprocess
import shutil
import ctypes
import _ctypes
import re
import functools
import numpy as np
from topi.util import get_const_tuple
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.expr import Tuple, GlobalVar
from tvm.relay.function import Function
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib import graph_runtime
import tvm.relay.op.contrib.tidl as tidl_annotation
from .tidl_reduce_subgraph_size import reduce_subgraph_size

def traverse_expr(node, node_dict):
    if node in node_dict:
        return
    if isinstance(node, tvm.ir.Op):
        return
    node_dict[node] = len(node_dict)

def find_data_layout(mod):
    all_nodes = {}
    traverse_func = functools.partial(traverse_expr, node_dict=all_nodes)
    relay.analysis.post_order_visit(mod['main'], traverse_func)
    data_layout = None
    for node in all_nodes:
        if isinstance(node, relay.expr.Call) and node.op.name == 'nn.conv2d':
            data_layout = node.attrs.data_layout
            break
    return data_layout

def find_in_nodes(all_nodes, this_node, input_prefix):
    r""" Find the input nodes of a given relay.expr.Call node.

         Only find input nodes that are relay.expr.Call or Tuple.
         If an input node is a relay.expr.TupleGetItem, then check this input
         node's input node.

    Parameters
    ----------
    all_nodes : dictionary
        Dictionary of all nodes of the graph
    this_node : relay.expr.Call or relay.expr.Tuple
        A relay.expr.Call or Tuple node whose input nodes are to be found by this function
    input_prefix : string
        Prefix of input tensor name, e.g. "tidl" when target is "tidl"

    Returns
    -------
    input_nodes : list
        A list of all input node indices of the given node
    """

    input_nodes = []
    if isinstance(this_node, relay.expr.Call):
        in_nodes = this_node.args
    else: # this_node is relay.expr.Tuple)
        in_nodes = this_node.fields

    for node in in_nodes:
        if isinstance(node, relay.expr.Call):
            input_nodes.append(all_nodes[node])
        elif isinstance(node, relay.expr.TupleGetItem):
            input_nodes.append(all_nodes[node.tuple_value])
        elif isinstance(node, relay.expr.Tuple):
            input_nodes = input_nodes + find_in_nodes(all_nodes, node, input_prefix)
        elif isinstance(node, relay.expr.Var):
            if input_prefix in node.name_hint and "_i" in node.name_hint:
                # this is the input to the subgraph
                input_nodes.append(-1)
        #else: ignore all other types of nodes: var, const, etc.

    return input_nodes

def find_out_nodes(all_nodes, this_node):
    r""" Find the output nodes of a given relay.expr.Call node.

    Parameters
    ----------
    all_nodes : dictionary
        Dictionary of all relay.expr.Call nodes of the graph
    this_node : relay.expr.Call
        A relay.expr.Call node whose output nodes are to be found by this function

    Returns
    -------
    output_nodes : list
        A list of all output node indices of the given node
    """

    output_nodes = []
    for node, node_idx in all_nodes.items():
        if isinstance(node, relay.expr.Call):
            if this_node in node.args:
                output_nodes.append(node_idx)
        elif isinstance(node, relay.expr.TupleGetItem):
            if this_node == node.tuple_value:
                output_nodes = output_nodes + find_out_nodes(all_nodes, node)
        elif isinstance(node, relay.expr.Tuple):
            if this_node in node.fields:
                tuple_node_outs = find_out_nodes(all_nodes, node)
                if len(tuple_node_outs) == 0:
                    # this is an output node
                    output_nodes.append(all_nodes[node])
                else:
                    # this is an input node to another node
                    output_nodes = output_nodes + tuple_node_outs

    return output_nodes

def find_in_out_nodes(all_nodes, this_node, input_prefix):
    r""" Find the input and output nodes of a given relay.expr.Call node.

    Parameters
    ----------
    all_nodes : dictionary
        Dictionary of all relay.expr.Call nodes of the graph
    this_node : relay.expr.Call
        A relay.expr.Call node whose input and output nodes are to be found
    input_prefix : string
        Prefix of input tensor name, e.g. "tidl" when target is "tidl"

    Returns
    -------
    in_out_nodes : InOutNodes
        Structure that stores indices of input nodes and output nodes
    """

    in_out_nodes = InOutNodes()    # instantiate structure

    in_out_nodes.this_node = all_nodes[this_node]

    in_nodes = find_in_nodes(all_nodes, this_node, input_prefix) # node indices of input nodes
    if len(in_nodes) == 0:
        in_out_nodes.in_nodes = None  # this is the first node
    else:
        # convert list to numpy arrary in order to pass to C library
        in_nodes_array = np.asarray(in_nodes, dtype=np.int32)
        in_out_nodes.in_nodes = ctypes.c_void_p(in_nodes_array.ctypes.data)

    in_out_nodes.num_in_nodes = len(in_nodes)

    out_nodes = find_out_nodes(all_nodes, this_node) # node indices of output nodes
    if len(out_nodes) == 0:
        in_out_nodes.out_nodes = None # this is the last node
    else:
        # convert list to numpy arrary in order to pass to C library
        out_nodes_array = np.asarray(out_nodes, dtype=np.int32)
        in_out_nodes.out_nodes = ctypes.c_void_p(out_nodes_array.ctypes.data)

    in_out_nodes.num_out_nodes = len(out_nodes)

    return in_out_nodes

def obtain_subgraph_tensor(subgraph_tensors, tensor_name_prefix):
    r""" Obtain input/output tensor for a given subgraph"""

    tensor = []
    for key, value in subgraph_tensors.items():
        if key.find(tensor_name_prefix) != -1:
            tensor.append(value)
    return tensor

def tensor_quant_flatten(input_tensor, data_layout):
    r""" Convert float32 n-d array to int8 or uint8 1-d array

    Parameters
    ----------
    input_tensor: float32 array
    data_layout: "NCHW" or "NHWC"
    """

    # only use 1 batch for calibration
    input_tensor = input_tensor[0, :]
    # change layout to CxHxW to use numpy.flattern to change to 1-d array
    if data_layout == "NHWC" and len(input_tensor.shape) == 3:
        input_tensor = input_tensor.transpose(2, 0, 1)

    max_value = max(abs(np.amin(input_tensor)), np.amax(input_tensor))
    if max_value == 0:
        max_value = 1.0  # arbitrary number if input tensor is all 0's
    if np.amin(input_tensor) >= 0:
        # quantize to Uint8
        sign = 0
        scale = 255.0/max_value
        quant_min, quant_max = 0, 255
    else:
        # quantize to Int8
        sign = 1
        scale = 128.0/max_value
        quant_min, quant_max = -128, 127

    tensor_norm = np.multiply(input_tensor, scale)
    tensor_quant = np.rint(tensor_norm)
    tensor_quant = np.clip(tensor_quant, quant_min, quant_max)
    output = tensor_quant.flatten()   # works only if tensor_quant is in "CxHxW" format

    return output, scale, sign

class VarReplacer(ExprMutator):
    """
    Replaces vars in expr according to var_map.
    """
    def __init__(self, var_map):
        ExprMutator.__init__(self)
        self.var_map = var_map

    def visit_var(self, var):
        if var in self.var_map:
            return self.var_map[var]
        return super().visit_var(var)

def unpack_composites(mod):
    """Unpack all composite functions in the module by replacing composite call nodes with the
    ops inside the composite function."""

    class Unpacker(ExprMutator):
        """Unpacks composite functions."""
        def __init__(self):
            ExprMutator.__init__(self)

        def visit_call(self, call):
            if isinstance(call.op, Function):
                if call.op.attrs and call.op.attrs['Composite'] != "":
                    # unpack the function back into new main function.
                    var_map = {}
                    for arg, param in zip(call.args, call.op.params):
                        var_map[param] = super().visit(arg)
                    return VarReplacer(var_map).visit(call.op.body)
            return super().visit_call(call)

    for func in mod.get_global_vars():
        mod[func.name_hint] = Unpacker().visit(mod[func.name_hint])
    return mod

class CalibrationGraphMutator(ExprMutator):
    """This mutator should be called after partioning to produce a module which
    can be executed purely using TVM and will produce additional outputs for
    subgraph inputs. name_map can be used to find the subgraph input name
    corresponding to the output of the same index.
    """
    def __init__(self, compiler):
        ExprMutator.__init__(self)
        self.num_original_outputs = 1
        self.additional_outputs = []
        self.compiler = compiler
        # Will map index in output to subgraph param name.
        self.name_map = {}

    def add_new_outputs(self, subgraph_name, expr, was_input=True):
        """
        Adds expr as an additional output to be generated by the module.
        If expr is a tuple, multiple outputs will be added.
        """
        if isinstance(expr, Tuple):
            for i, out in enumerate(expr.fields):
                if was_input:
                    name = subgraph_name + "_" + str(i)
                else:
                    name = subgraph_name + "_o" + str(i)
                self.name_map[self.num_original_outputs + len(self.additional_outputs)] = name
                self.additional_outputs.append(out)
        else:
            if was_input:
                name = subgraph_name
            else:
                name = subgraph_name + "_o0"
            self.name_map[self.num_original_outputs + len(self.additional_outputs)] = name
            self.additional_outputs.append(expr)

    def visit_call(self, call):
        if isinstance(call.op, Function) and "Compiler" in call.op.attrs \
           and call.op.attrs["Compiler"] == self.compiler:
            var_map = {}
            for arg, param in zip(call.args, call.op.params):
                subgraph_name = "_".join(param.name_hint.split("_")[:2])
                arg = super().visit(arg)
                var_map[param] = arg
                self.add_new_outputs(param.name_hint, arg, was_input=True)
            new_body = VarReplacer(var_map).visit(call.op.body)
            # Add subgraph outputs as well
            self.add_new_outputs(subgraph_name, new_body, was_input=False)
            return new_body
        return super().visit_call(call)

    def make_calibration_graph(self, expr):
        """Builds calibration graph for expr"""

        if isinstance(expr.body.checked_type, relay.TupleType):
            self.num_original_outputs = len(expr.body.checked_type.fields)
        visit_body = super().visit(expr.body)
        # Get original output(s)
        outputs = []
        if isinstance(visit_body, Tuple):
            for out in visit_body.fields:
                outputs.append(out)
        else:
            outputs.append(visit_body)
        # Create new function with added subgraph inputs + outputs
        return relay.Function(expr.params, relay.Tuple(outputs + self.additional_outputs))

class RemoveMultiplyByOne(ExprMutator):
    """
    Removes multiply by 1.0f. This pass when followed by
    RemoveRedundantTranspose is intended to remove a pattern of
    Transpose([1, 0]) -> Scale(1.0f) -> Transpose([1, 0]) produced by
    PyTorch's addmm operator.
    """
    def visit_call(self, call):
        if call.op.name == "multiply":
            if isinstance(call.args[1], tvm.relay.expr.Constant):
                data = call.args[1].data.asnumpy()
                if data.shape == () and data.item() == 1.0:
                    return call.args[0]
            if isinstance(call.args[0], tvm.relay.expr.Constant):
                data = call.args[0].data.asnumpy()
                if data.shape == () and data.item() == 1.0:
                    return call.args[1]
        return super().visit_call(call)

def generate_subgraph_tensors(tidl_target, mod, params, graph_input, save_output=False):
    """Creates calibration graph from mod and executes on the cpu to generate boundary tensors.
    """

    # From partitioned module, create a "calibration model" which can be
    # executed on CPU and will give additional outputs for boundary tensors.
    mod_tvm = relay.transform.InferType()(mod)
    mod_tvm = relay.transform.Inline()(mod_tvm)
    calib_mutator = CalibrationGraphMutator(tidl_target)
    mod_tvm["main"] = calib_mutator.make_calibration_graph(mod_tvm["main"])

    # Build and execute calibration graph to get outputs
    # Use opt_level=0 to avoid optimizations which modify the module (could change original module)
    with relay.build_config(opt_level=0):
        graph, lib, params = relay.build(mod_tvm, "llvm", params=params)
    mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    mod.set_input(**graph_input)
    mod.set_input(**params)
    mod.run()

    results = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]
    np.savetxt('graph_output.txt', results[0].flatten(), fmt='%10.5f')

    # We now have subgraph inputs
    # {1: 'tidl_1_i0', 2: 'tidl_1_o0', 3: 'tidl_0_i0', 4: 'tidl_0_o0'}
    subgraph_tensors = {}
    for i, res in enumerate(results):
        if i in calib_mutator.name_map:
            subgraph_tensors[calib_mutator.name_map[i]] = res
            if save_output:
                file_name = calib_mutator.name_map[i] + ".txt"
                np.savetxt(file_name, res.flatten(), fmt='%10.5f')

    return subgraph_tensors

class VarRenamer(ExprMutator):
    """
    Renames vars to match the new subgraph name. Used when subgraphs are renamed starting from zero.
    If subgraph was originally "tidl_34", it would have inputs named like "tidl_34_i0".
    IF new_subgraph_name is "tidl_0", pass will that input to "tidl_0_i0".
    """
    def __init__(self, new_subgraph_name):
        ExprMutator.__init__(self)
        self.new_subgraph_name = new_subgraph_name

    def visit_var(self, var):
        # TODO: Make sure input isn't from a composite func.
        # TODO: Doesn't account for tuple inputs (not possible due to
        #       prune_subgraphs_with_multiple_inputs)
        if var.name_hint.startswith("tidl") and "_".join(var.name_hint.split('_')[:2]) \
                                                != self.new_subgraph_name:
            new_var_name = self.new_subgraph_name + "_" + var.name_hint.split('_')[2]
            return relay.Var(new_var_name, var.checked_type)
        return super().visit_var(var)

class SubgraphRemover(ExprMutator):
    """
    Removes subgraphs which are in the list subgraphs_to_remove and returns them back to regular
    TVM compilation in main function.
    """
    def __init__(self, subgraphs_to_remove, mod, new_mod, rename_starting_from_0=True):
        ExprMutator.__init__(self)
        self.subgraphs_to_remove = subgraphs_to_remove
        self.mod = mod
        self.new_mod = new_mod
        self.rename_starting_from_0 = rename_starting_from_0
        self.count = 0

    def visit_call(self, call):
        if isinstance(call.op, GlobalVar):
            name = call.op.name_hint
            if name in self.subgraphs_to_remove:
                # "Inline" the subgraph back into new main function.
                func = self.mod[name]
                var_map = {}
                for arg, param in zip(call.args, func.params):
                    var_map[param] = super().visit(arg)
                new_body = VarReplacer(var_map).visit(func.body)
                return new_body
            if name != "main":
                # Copy the GlobalVar (subgraph function) to the new module and call.
                if self.rename_starting_from_0:
                    new_name = name.split('_')[0] + "_" + str(self.count)
                    self.count += 1
                else:
                    new_name = name
                args = []
                for arg in call.args:
                    args.append(super().visit(arg))
                subgraph_gv = relay.GlobalVar(new_name)
                if self.rename_starting_from_0:
                    subgraph_func = VarRenamer(new_name).visit(self.mod[name])
                    subgraph_func = subgraph_func.with_attr("global_symbol", new_name)
                    self.new_mod[subgraph_gv] = subgraph_func
                else:
                    self.new_mod[subgraph_gv] = self.mod[name]
                return subgraph_gv(*args)
        return super().visit_call(call)

def prune_subgraphs_with_multiple_inputs(mod, compiler="tidl"):
    """Removes subgraphs which have more than one input from mod and returns them to the regular
    TVM compilation path.

    Parameters
    ----------
    mod : tvm.IRModule
        Module containing subgraphs using external codegen "compiler"
    compiler : str
        Only subgraphs from this external codegen compiler will be modified.

    Returns
    -------
    ret : tvm.IRModule
        New module with only single-input subgraphs left.
    """
    subgraph_names_to_remove = []
    # Remove subgraphs with more than 1 input or tuple inputs.
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if not mod[name].attrs or mod[name].attrs["Compiler"] != compiler:
            continue
        if len(mod[name].params) != 1 \
           or isinstance(mod[name].params[0].checked_type, relay.TupleType):
            subgraph_names_to_remove.append(name)
    new_mod = tvm.IRModule()
    new_mod["main"] = SubgraphRemover(subgraph_names_to_remove, mod, new_mod).visit(mod["main"])
    return new_mod

def prune_subgraphs(mod, compiler="tidl", num_subgraphs_to_keep=4, min_mac_threshold=None):
    """Removes subgraphs from mod and returns them to the regular TVM compilation path.
    The subgraphs with the highest number of multiply-accumulates are kept.

    Parameters
    ----------
    mod : tvm.IRModule
        Module containing subgraphs using external codegen "compiler"
    compiler : str
        Only subgraphs from this external codegen compiler will be modified.
    num_subgraphs_to_keep : int
        How many subgraphs to keep.
    min_mac_threshold : int (optional)
        If set, will also prune all subgraphs with # macs < the threshold.

    Returns
    -------
    ret : tvm.IRModule
        New module with only "num_subgraphs_to_keep" subgraphs left.
    """
    subgraph_with_macs = []
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if not mod[name].attrs or mod[name].attrs["Compiler"] != compiler:
            continue
        num_macs = relay.analysis.get_total_mac_number(mod[name])
        subgraph_with_macs.append([name, num_macs])
    subgraph_with_macs = sorted(subgraph_with_macs, key=lambda x: int(x[1]))
    subgraphs_to_prune = subgraph_with_macs[:-num_subgraphs_to_keep]
    if min_mac_threshold:
        # Also remove all subgraphs under the minimum threshold.
        subgraphs_to_prune += [[x[0], x[1]] for x in subgraph_with_macs if x[1] < min_mac_threshold]
    subgraph_names_to_remove = {x[0] for x in subgraphs_to_prune}
    # Create new pruned module
    new_mod = tvm.IRModule()
    new_mod["main"] = SubgraphRemover(subgraph_names_to_remove, mod, new_mod).visit(mod["main"])
    return new_mod


def subgraph_cfg_gen(artifacts_folder, subgraph_id, data_layout,
                     input_scale, input_signed, output_scale, output_signed):
    r""" Generate subgraph configuration file to be used by TIDL runtime

    Parameters
    ----------
    input_scale : vector
        scaling factor to convert floating point TVM tensors to 8-bit TIDL inputs,
        where TIDL_input[i] = TVM_tensor[i] * input_scale[i]
    input_signed : vector
        indicating whether input tensor to TIDL is signed (1) or unsigned (0)
    output_scale : vector
        scaling factor to convert 8-bit TIDL outputs to floating point TVM tensors,
        where TVM_tensor[i] = TIDL_input[i] / output_scale[i]
    output_signed : vector
        indicating whether output tensor of TIDL is signed (1) or unsigned (0)

    Returns
    -------
    """

    def print_list(in_list):
        str0 = str(in_list)
        str1 = str0.replace("[", "")
        str2 = str1.replace("]", "")
        str3 = str2.replace(",", "", len(in_list)-1)
        return str3

    if data_layout == "NCHW":
        layout_is_nchw = 1
    else:
        layout_is_nchw = 0
    out_conv_type = [0 for i in range(len(output_scale))]
    out_is_nchw = [layout_is_nchw for i in range(len(output_scale))]

    sub_graph_cfg = os.path.join(artifacts_folder, "subgraph" + str(subgraph_id) + ".cfg")
    sub_graph_net_file = "./tidl_subgraph" + str(subgraph_id) + "_net.bin"
    sub_graph_params_file = "./tidl_subgraph" + str(subgraph_id) + "_params.bin"
    with open(sub_graph_cfg, 'w') as cfg_file:
        cfg_file.write("netBinFile    = {}\n".format(sub_graph_net_file))
        cfg_file.write("paramsBinFile = {}\n".format(sub_graph_params_file))
        cfg_file.write("inConvType    = 0\n")
        cfg_file.write("inIsSigned    = {}\n".format(input_signed))
        cfg_file.write("inScaleF2Q    = {}\n".format(round(input_scale, 2)))
        cfg_file.write("inIsNCHW      = {}\n".format(layout_is_nchw))
        cfg_file.write("outConvType   = {}\n".format(print_list(out_conv_type)))
        cfg_file.write("outIsSigned   = {}\n".format(print_list(output_signed)))
        cfg_file.write("outScaleF2Q   = {}\n".format(print_list(output_scale)))
        cfg_file.write("outIsNCHW     = {}\n".format(print_list(out_is_nchw)))

def subgraph_calibration(calib_tool, input_quant_vec, input_signed, net_file, params_file):
    """ Run TIDL calibation for the imported subgraph.
    """
    # Prepare for calibration
    temp_folder = './tempDir/'
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)
    os.mkdir(temp_folder)

    # Save quantized input vector to a file for calib tool to read
    # Saving as 'int8' or 'uint8' is the same
    calib_raw_image = temp_folder + 'calib_raw_data.bin'
    if input_signed == 1:
        input_quant_vec.astype('int8').tofile(calib_raw_image)
    else:
        input_quant_vec.astype('uint8').tofile(calib_raw_image)

    output_tmp_file = temp_folder + 'precalib_net.bin'
    shutil.copyfile(net_file, output_tmp_file)

    calib_config_file = temp_folder + 'configFilesList.txt'
    quant_config_file = './tempDir/quant_stats_config.txt'
    with open(calib_config_file, 'w') as config_file:
        config_file.write('1 ' + quant_config_file + '\n')
        config_file.write('0\n')

    with open(quant_config_file, 'w') as quant_file:
        quant_file.write('rawImage    = 1\n')
        quant_file.write('numFrames   = 1\n')
        quant_file.write('preProcType = 0\n')
        quant_file.write('inData      = {}\n'.format(calib_raw_image))
        quant_file.write('outData     = {}\n'.format(temp_folder + 'stats_tool_out.bin'))
        quant_file.write('traceDumpBaseName  = {}\n'.format(temp_folder + 'trace_dump_'))
        quant_file.write('updateNetWithStats = 1\n')
        quant_file.write('outputNetBinFile   = {}\n'.format(net_file))
        quant_file.write('paramsBinFile      = {}\n'.format(params_file))
        quant_file.write('netBinFile         = {}\n'.format(output_tmp_file))

    # Invoke TIDL emulation to calibrate
    try:
        proc = subprocess.Popen([calib_tool, calib_config_file], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = proc.communicate()
        console_out = out.decode('ascii')
        error = err.decode('ascii')
        print(console_out)
    except: # pylint: disable=bare-except
        print("TIDL calibration crashed")
        return False, None

    # Find output dataQs from calibration console output
    if console_out.find('error') == -1 and console_out.find('ERROR') == -1 and error == '':
        output_data_token = "Number of output dataQ:"
        out_buf_ind = console_out.rfind(output_data_token)
        if out_buf_ind == -1:
            print("TIDL calibration failed - can't find number of output buffers.")
            status, out_data_q = False, None
        else:
            last_line = console_out.split(output_data_token, 1)[1]
            num_outputs = int(last_line.split(". Output dataQ:", 1)[0])
            out_quants = last_line.split(". Output dataQ:", 1)[1]
            quants = out_quants.split("End of output dataQ", 1)[0]
            outq_str = re.findall(r"\d+", quants)
            outq = list(map(int, outq_str))
            if num_outputs != len(outq):
                print("TIDL calibration failed - can't find all outputQ's")
                status, out_data_q = False, None
            else:
                status, out_data_q = True, outq
    else:
        print("TIDL calibration failed.")
        print(error)
        status, out_data_q = False, None

    return status, out_data_q

class TIDLconfigParams(ctypes.Structure):
    """ TIDL config parameters defined in ctypes for passing to TIDL C library """
    _fields_ = [('numParamBits', ctypes.c_int),
                ('quantRoundAdd', ctypes.c_int),
                ('inQuantFactor', ctypes.c_int),
                ('inElementType', ctypes.c_int),
                ('inNumChannels', ctypes.c_int),
                ('inHeight', ctypes.c_int),
                ('inWidth', ctypes.c_int)]

class Conv2dParams(ctypes.Structure):
    """ Conv2d parameters defined in ctypes for passing to TIDL C library """
    _fields_ = [('num_in_channels', ctypes.c_int),
                ('num_out_channels', ctypes.c_int),
                ('num_groups', ctypes.c_int),
                ('stride_h', ctypes.c_int), ('stride_w', ctypes.c_int),
                ('dilation_h', ctypes.c_int), ('dilation_w', ctypes.c_int),
                ('pad_t', ctypes.c_int), ('pad_l', ctypes.c_int),
                ('pad_b', ctypes.c_int), ('pad_r', ctypes.c_int),
                ('kernel_h', ctypes.c_int), ('kernel_w', ctypes.c_int),
                ('kernel_layout', ctypes.c_char_p),
                ('weights_array', ctypes.c_void_p),
                ('weights_type', ctypes.c_char_p)]

class BatchNormParams(ctypes.Structure):
    """ BatchNorm parameters defined in ctypes for passing to TIDL C library """
    _fields_ = [('num_params', ctypes.c_int),
                ('params_dtype', ctypes.c_char_p),
                ('gama', ctypes.c_void_p),
                ('beta', ctypes.c_void_p),
                ('mean', ctypes.c_void_p),
                ('var', ctypes.c_void_p),
                ('epsilon', ctypes.c_float),
                ('center_enable', ctypes.c_int),
                ('scale_enable', ctypes.c_int)]

class PoolingParams(ctypes.Structure):
    """ Pooling parameters defined in ctypes for passing to TIDL C library """
    _fields_ = [('kernel_h', ctypes.c_int),
                ('kernel_w', ctypes.c_int),
                ('stride_h', ctypes.c_int),
                ('stride_w', ctypes.c_int),
                ('pad_h', ctypes.c_int),
                ('pad_w', ctypes.c_int)]

class InOutNodes(ctypes.Structure):
    """ Input/output nodes defined in ctypes for passing to TIDL C library """
    _fields_ = [('this_node', ctypes.c_int),
                ('num_in_nodes', ctypes.c_int), ('num_out_nodes', ctypes.c_int),
                ('in_nodes', ctypes.c_void_p), ('out_nodes', ctypes.c_void_p)]

class TIDLImport:
    """TIDL import module.
    Parameters
    ----------
    import_lib : ctypes.CDLL
        TIDL import library
    calib_tool : string
        TIDL calibration tool file
    artifacts_folder : string
        Directory path to hold the artifacts
    tidl_target : string
        TIDL compilation target
    data_layout : string
        Data layout, "NCHW" or "NHWC"
    """
    def __init__(self, import_lib, calib_tool, artifacts_folder,
                 tidl_target="tidl", data_layout="NCHW"):
        self.import_lib = import_lib
        self.calib_tool = calib_tool
        self.artifacts_folder = artifacts_folder
        self.tidl_target = tidl_target
        self.data_layout = data_layout

    def tidl_import_conv2d(self, this_node):
        r""" Import conv2d operator to TIDL
            There is an example how to get the attributes of conv2d in Relay:
            https://github.com/dmlc/tvm/blob/master/python/tvm/relay/op/nn/_nn.py#L144
            https://docs.tvm.ai/api/python/ndarray.html

        Parameters
        ----------
        all_nodes : dictionary
            Dictionary of all relay.expr.Call nodes of the graph
        this_node : relay.expr.Call
            A relay.expr.Call node which is a conv2d operator

        Returns
        -------
        True if import succeeds or False if import fails
        """

        weight = this_node.args[1]
        weight_shape = get_const_tuple(weight.checked_type.shape)
        weight_type = weight.checked_type.dtype
        strides = get_const_tuple(this_node.attrs.strides)
        dilation = get_const_tuple(this_node.attrs.dilation)
        padding = get_const_tuple(this_node.attrs.padding)
        kernel_size = get_const_tuple(this_node.attrs.kernel_size)
        groups = this_node.attrs.groups
        kernel_layout = this_node.attrs.kernel_layout

        conv2d_params = Conv2dParams()
        (conv2d_params.stride_h, conv2d_params.stride_w) = strides
        (conv2d_params.dilation_h, conv2d_params.dilation_w) = dilation
        # top, left, bottom, right padding
        if len(padding) == 1:
            pad_t = pad_l = pad_b = pad_r = padding[0]
        elif len(padding) == 2:
            pad_t = pad_b = padding[0]
            pad_l = pad_r = padding[1]
        else:
            (pad_t, pad_l, pad_b, pad_r) = padding
        (conv2d_params.pad_t, conv2d_params.pad_l, conv2d_params.pad_b, conv2d_params.pad_r) \
          = (pad_t, pad_l, pad_b, pad_r)
        (conv2d_params.kernel_h, conv2d_params.kernel_w) = kernel_size
        conv2d_params.num_groups = groups

        # Obtain weights and convert to numpy array and then pass to C lib
        weights = weight.data
        weights_np = weights.asnumpy()

        if kernel_layout == 'OIHW':
            # No need to reshape - TIDL natively uses 'OIHW'
            conv2d_params.kernel_layout = b'OIHW'
            conv2d_params.num_in_channels = weight_shape[1]
            conv2d_params.num_out_channels = weight_shape[0]
            weights_to_tidl = weights_np
        elif kernel_layout == 'HWIO':
            # Reshape numpy array from 'HWIO' to 'OIHW'
            weights_to_tidl = weights_np.transpose((3, 2, 0, 1))
            conv2d_params.num_in_channels = weight_shape[2]
            conv2d_params.num_out_channels = weight_shape[3]
        elif kernel_layout == 'HWOI':
            # Reshape numpy array from 'HWOI' to 'OIHW'
            weights_to_tidl = weights_np.transpose((2, 3, 0, 1))
            conv2d_params.num_in_channels = weight_shape[3]
            conv2d_params.num_out_channels = weight_shape[2]
        else:
            print('Kernel layout ' + kernel_layout + ' not supported')
            return False

        if weight_type == 'float32':
            conv2d_params.weights_type = b'float32'
        else:
            print('Weight type ' + weight_type + ' not supported')
            return False

        weights_flatten = weights_to_tidl.flatten()
        conv2d_params.weights_array = ctypes.c_void_p(weights_flatten.ctypes.data)

        # Invoke C lib functions to pass parameters to TIDL
        import_lib_conv2d = self.import_lib.tidlImportConv2d
        import_lib_conv2d.argtypes = (ctypes.POINTER(Conv2dParams), ctypes.c_void_p)
        import_lib_conv2d.restype = None
        import_lib_conv2d(conv2d_params, ctypes.POINTER(ctypes.c_int)())
        return True

    def tidl_import_pad(self, node):
        r""" Import pad operator to TIDL
            Get attributes pad_width, convert to array, and passs to C library.
            A typical pad_width looks like: [[0,0],[0,1],[0,1],[0,0]]

        Parameters
        ----------
        node : relay.expr.Call
            A relay.expr.Call node which is a pad operator

        Returns
        -------
        """

        pad_width = []
        for width in node.attrs.pad_width:
            pad_width.append(get_const_tuple(width))
        pad_list = [x for xs in pad_width for x in xs]

        # convert list to numpy array in order to pass to C library
        pad_array = np.asarray(pad_list, dtype=np.int32)

        import_lib_pad = self.import_lib.tidlImportPad
        import_lib_pad.argtypes = (ctypes.c_int, ctypes.c_void_p)
        import_lib_pad.restype = None
        import_lib_pad(len(pad_array), ctypes.c_void_p(pad_array.ctypes.data))
        return True

    def tidl_import_add(self):
        r""" Import add operator to TIDL
            An "add" operator may be adding two nodes or adding one node with constant:
                - %3 = add(%2, %1)
                - %3 = add(%2, %MobilenetV2/Conv/Conv2D_bn_offset)
            This function imports "add" opertor with args being two nodes.
        """

        import_lib_add = self.import_lib.tidlImportAdd
        import_lib_add.argtypes = None
        import_lib_add.restype = None
        import_lib_add()
        return True

    def tidl_import_bias_add(self, node):
        r""" Import bias_add or add operator to TIDL
            An "add" operator may be adding two nodes or adding one node with constant:
                - %3 = add(%2, %1)
                - %3 = add(%2, %MobilenetV2/Conv/Conv2D_bn_offset)
            A "bias_add" operator always add one node with constant.
            This function imports a "bias_add" or "add" with args[1] being constant.

        Parameters
        ----------
        node : relay.expr.Call
            A relay.expr.Call node which is a add operator

        Returns
        -------
        True if import succeeds or False if import fails
        """
        bias = node.args[1]
        if isinstance(bias, tvm.relay.expr.Constant):
            # bias is expr.Constant if bind_params_by_name is called
            bias_params = bias.data
        else:
            # bias is expr.Var if bind_params_by_name is not called
            print('bias_add op must have args[1] as expr.Constant')
            return False

        if bias.checked_type.dtype == 'float32':
            bias_params_dtype = b'float32'
        else:
            print('Unsupported data type of bias_add')
            return False

        bias_params_len = bias.checked_type.shape[0]
        bias_params_np = bias_params.asnumpy()

        import_lib_bias = self.import_lib.tidlImportBiasAdd
        import_lib_bias.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
        import_lib_bias.restype = None
        import_lib_bias(bias_params_len, bias_params_dtype,
                        ctypes.c_void_p(bias_params_np.ctypes.data))
        return True

    def tidl_import_batch_norm(self, node):
        r""" Import batch_norm operator to TIDL
            https://docs.tvm.ai/langref/relay_op.html#tvm.relay.nn.batch_norm
            https://docs.tvm.ai/doxygen/structtvm_1_1relay_1_1BatchNormAttrs.html

        Parameters
        ----------
        node : relay.expr.Call
            A relay.expr.Call node which is a batch_norm operator

        Returns
        -------
        True if import succeeds or False if import fails
        """

        bn_params = BatchNormParams()
        if node.args[1].checked_type.dtype == 'float32':
            bn_params.params_dtype = b'float32'
        else:
            print('Unsupported data type of batch norm')
            return False
        bn_params.num_params = node.args[1].checked_type.shape[0]

        gama = node.args[1].data.asnumpy()
        beta = node.args[2].data.asnumpy()
        mean = node.args[3].data.asnumpy()
        var = node.args[4].data.asnumpy()
        bn_params.gama = gama.ctypes.data
        bn_params.beta = beta.ctypes.data
        bn_params.mean = mean.ctypes.data
        bn_params.var = var.ctypes.data
        bn_params.epsilon = node.attrs.epsilon
        center = node.attrs.center
        scale = node.attrs.scale
        bn_params.center_enable = int(center)
        bn_params.scale_enable = int(scale)

        import_lib_bn = self.import_lib.tidlImportBatchNorm
        import_lib_bn.argtypes = (ctypes.POINTER(BatchNormParams), ctypes.c_void_p)
        import_lib_bn.restype = None
        import_lib_bn(bn_params, ctypes.POINTER(ctypes.c_int)())
        return True

    def tidl_import_pooling(self, node):
        r""" Import pooling operator to TIDL
            https://docs.tvm.ai/langref/relay_op.html#tvm.relay.nn.avg_pool2d
            https://docs.tvm.ai/doxygen/structtvm_1_1relay_1_1AvgPool2DAttrs.html
            https://docs.tvm.ai/langref/relay_op.html#tvm.relay.nn.max_pool2d
            https://docs.tvm.ai/doxygen/structtvm_1_1relay_1_1MaxPool2DAttrs.html

        Parameters
        ----------
        node : relay.expr.Call
            A relay.expr.Call node which is a pooling operator

        Returns
        -------
        """

        pooling_params = PoolingParams()
        if node.op.name == "nn.global_avg_pool2d":
            pooling_params.kernel_h = pooling_params.kernel_w = 0
            pooling_params.pad_h = pooling_params.pad_w = 0
            pooling_params.stride_h = pooling_params.stride_w = 1
        else:
            (pooling_params.kernel_h, pooling_params.kernel_w) = node.attrs.pool_size
            (pooling_params.stride_h, pooling_params.stride_w) = node.attrs.strides
            if len(node.attrs.padding) == 4:
                (pooling_params.pad_h, pooling_params.pad_w) = node.attrs.padding[2:4]
            else:
                (pooling_params.pad_h, pooling_params.pad_w) = node.attrs.padding

        if node.op.name == "nn.avg_pool2d" or node.op.name == "nn.global_avg_pool2d":
            pooling_type = b'avg_pool2d'
        else:
            pooling_type = b'max_pool2d'

        import_lib_pooling = self.import_lib.tidlImportPooling
        import_lib_pooling.argtypes = (ctypes.POINTER(PoolingParams), ctypes.c_char_p)
        import_lib_pooling.restype = None
        import_lib_pooling(pooling_params, pooling_type)
        return True

    def tidl_import_concat(self, all_nodes, node):

        in_nodes = find_in_nodes(all_nodes, node, self.tidl_target) # node indices of input nodes
        import_lib_concat = self.import_lib.tidlImportConcat
        import_lib_concat.argtype = ctypes.c_int
        import_lib_concat.restype = None
        import_lib_concat(len(in_nodes))
        return True

    def tidl_import_dense(self, this_node):

        weights = this_node.args[1]
        (num_outnodes, num_innodes) = weights.data.shape
        weights_array = weights.data.asnumpy()
        import_lib_dense = self.import_lib.tidlImportDense
        import_lib_dense.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        import_lib_dense.restype = None
        import_lib_dense(num_innodes, num_outnodes, ctypes.c_void_p(weights_array.ctypes.data))
        return True

    def tidl_import_max_mean(self, node):
        r""" Import 'max' or 'mean' operator to TIDL pooling layer
        """

        pooling_params = PoolingParams()
        pooling_params.kernel_h = pooling_params.kernel_w = 0
        pooling_params.pad_h = pooling_params.pad_w = 0
        pooling_params.stride_h = pooling_params.stride_w = 1
        if node.op.name == "max":
            pooling_type = b'max_pool2d'
        else:
            pooling_type = b'avg_pool2d'
        import_lib_pooling = self.import_lib.tidlImportPooling
        import_lib_pooling.argtypes = (ctypes.POINTER(PoolingParams), ctypes.c_char_p)
        import_lib_pooling.restype = None
        import_lib_pooling(pooling_params, pooling_type)
        return True

    def tidl_import_init(self, input_scale, input_signed, input_shape):
        r""" Initializing TIDL import

        Parameters
        ----------
        input_scale: double
            Scaling factor to convert floating point input to 8-bit quantized input
        input_signed: int
            Signed (1) or unsigned (0) of input
        input_shape: tuple
            Input shape (N,C,H,W) or (N,H,W,C)
        Returns
        -------
        True if initialization succeeds or False if initialization fails
        """

        if len(input_shape) == 2:
            # input is a vector - expand (N,W) to (N,1,1,W) or (N,1,W,1)
            if self.data_layout == "NCHW":
                in_shape = (input_shape[0], 1, 1, input_shape[1])
            else:
                in_shape = (input_shape[0], 1, input_shape[1], 1)
        elif len(input_shape) == 3:
            # expand (N,H,W) to (N,1,H,W) or (N,H,W,1)
            if self.data_layout == "NCHW":
                in_shape = (input_shape[0], 1, input_shape[1], input_shape[2])
            else:
                in_shape = (input_shape[0], input_shape[1], input_shape[2], 1)
        elif len(input_shape) == 4:
            in_shape = input_shape
        else:
            print("Subgraph input_shape " + str(input_shape) + " is not supported")
            return False

        if self.data_layout == "NCHW":
            layout = b'NCHW'
            (channel, height, width) = in_shape[1:4]
        elif self.data_layout == "NHWC":
            layout = b'NHWC'
            (channel, height, width) = (in_shape[3], in_shape[1], in_shape[2])
        else:
            print('data layout ' + self.data_layout + ' is not supported')
            return False

        in_quant_factor = int(round(input_scale*255))  # 255 is due to TIDL implementation
        config_params = TIDLconfigParams(12, 50, in_quant_factor, input_signed,
                                         channel, height, width)

        # Invoking C library call to initialize TIDL import
        import_lib_init = self.import_lib.tidlImportInit
        import_lib_init.argtypes = (ctypes.POINTER(TIDLconfigParams), ctypes.c_char_p)
        import_lib_init.restype = None
        import_lib_init(config_params, layout)

        return True

    def tidl_import_node(self, all_nodes, this_node):
        r""" Importing a given node (operator) to TIDL
            # https://docs.tvm.ai/langref/relay_op.html#relay-core-tensor-operators

        Parameters
        ----------
        all_nodes : dictionary
            Dictionary of all relay.expr.Call nodes of the graph
        this_node : relay.expr.Call
            A relay.expr.Call node which is to be imported

        Returns
        True if import succeeds or False if import fails
        """

        status = True
        if this_node.op.name == 'nn.conv2d':
            status = self.tidl_import_conv2d(this_node)
        elif this_node.op.name == 'nn.pad':
            status = self.tidl_import_pad(this_node)
        elif this_node.op.name == 'add':
            if isinstance(this_node.args[1], tvm.relay.expr.Constant):
                status = self.tidl_import_bias_add(this_node)
            else:
                status = self.tidl_import_add()
        elif this_node.op.name == 'nn.bias_add':
            status = self.tidl_import_bias_add(this_node)
        elif this_node.op.name == 'clip':
            import_lib_relu = self.import_lib.tidlImportRelu
            import_lib_relu.argtype = (ctypes.c_char_p)
            import_lib_relu.restype = None
            import_lib_relu(b'Relu6')
        elif this_node.op.name == 'nn.relu':
            import_lib_relu = self.import_lib.tidlImportRelu
            import_lib_relu.argtype = (ctypes.c_char_p)
            import_lib_relu.restype = None
            import_lib_relu(b'Relu')
        elif this_node.op.name == 'nn.batch_norm':
            status = self.tidl_import_batch_norm(this_node)
        elif this_node.op.name == 'nn.avg_pool2d':
            status = self.tidl_import_pooling(this_node)
        elif this_node.op.name == 'squeeze':
            import_lib_squeeze = self.import_lib.tidlImportSqueeze
            import_lib_squeeze.argtype = None
            import_lib_squeeze.restype = None
            import_lib_squeeze()
        elif this_node.op.name == 'reshape':
            import_lib_reshape = self.import_lib.tidlImportReshape
            import_lib_reshape.argtype = None
            import_lib_reshape.restype = None
            import_lib_reshape()
        elif this_node.op.name == 'nn.softmax':
            import_lib_softmax = self.import_lib.tidlImportSoftmax
            import_lib_softmax.argtype = None
            import_lib_softmax.restype = None
            import_lib_softmax()
        elif this_node.op.name == 'concatenate':
            status = self.tidl_import_concat(all_nodes, this_node)
        elif this_node.op.name == 'nn.max_pool2d':
            status = self.tidl_import_pooling(this_node)
        elif this_node.op.name == 'nn.dropout':
            import_lib_dropout = self.import_lib.tidlImportDropOut
            import_lib_dropout.argtype = None
            import_lib_dropout.restype = None
            import_lib_dropout()
        elif this_node.op.name == 'nn.global_avg_pool2d':
            status = self.tidl_import_pooling(this_node)
        elif this_node.op.name == 'nn.batch_flatten':
            import_lib_flatten = self.import_lib.tidlImportBatchFlatten
            import_lib_flatten.argtype = None
            import_lib_flatten.restype = None
            import_lib_flatten()
        elif this_node.op.name == 'nn.dense':
            status = self.tidl_import_dense(this_node)
        elif this_node.op.name == 'max':
            self.tidl_import_max_mean(this_node)
        elif this_node.op.name == 'mean':
            self.tidl_import_max_mean(this_node)
        else:
            print("Operator " + this_node.op.name + " is not supported by TIDL!")
            status = False

        if not status:
            return False

        # Common for all nodes:
        # fill tensor names, update consumer counts, link input/output tensors
        in_out_nodes = find_in_out_nodes(all_nodes, this_node, self.tidl_target)

        import_lib_link_nodes = self.import_lib.tidlImportLinkNodes
        import_lib_link_nodes.argtypes = (ctypes.POINTER(InOutNodes), ctypes.c_void_p)
        import_lib_link_nodes.restype = ctypes.c_int
        if import_lib_link_nodes(in_out_nodes, ctypes.POINTER(ctypes.c_int)()) == 0:
            return False

        return True

    def tidl_import_tuple_node(self, all_nodes, node):
        """ Importing a Relay tuple node, e.g. (%232, %279, %283, %274).
            If this node is the last node, import it to TIDL output data layer.
            If this node is not the last node, do nothing.
        """

        max_num_outputs_per_data_layer = 16
        out_nodes = find_out_nodes(all_nodes, node)
        if len(out_nodes) == 0:
            # this is the last node of the graph - import this to out data layer
            in_nodes = find_in_nodes(all_nodes, node, self.tidl_target)
            imported_nodes = 0
            new_node_ind = len(all_nodes) + 1
            status = True
            while imported_nodes < len(in_nodes):
                if len(in_nodes) - imported_nodes < max_num_outputs_per_data_layer:
                    nodes_for_this_data_layer = len(in_nodes) - imported_nodes
                    this_is_the_last_one = True
                else:
                    nodes_for_this_data_layer = max_num_outputs_per_data_layer
                    this_is_the_last_one = False

                import_lib_out_data = self.import_lib.tidlImportOutData
                import_lib_out_data.argtype = ctypes.c_int
                import_lib_out_data.restype = None
                import_lib_out_data(nodes_for_this_data_layer)

                # prepare input/output nodes information for linking
                in_out_nodes = InOutNodes()    # instantiate structure
                in_out_nodes.this_node = new_node_ind
                in_out_nodes.num_in_nodes = nodes_for_this_data_layer
                in_nodes_this_layer = \
                    in_nodes[imported_nodes:imported_nodes+nodes_for_this_data_layer]
                in_nodes_array = np.asarray(in_nodes_this_layer, dtype=np.int32)
                in_out_nodes.in_nodes = ctypes.c_void_p(in_nodes_array.ctypes.data)
                in_out_nodes.out_nodes = None
                in_out_nodes.num_out_nodes = 0

                import_lib_link_nodes = self.import_lib.tidlImportLinkNodes
                import_lib_link_nodes.argtypes = (ctypes.POINTER(InOutNodes), ctypes.c_void_p)
                import_lib_link_nodes.restype = ctypes.c_int
                if import_lib_link_nodes(in_out_nodes, ctypes.POINTER(ctypes.c_int)()) == 0:
                    status = False
                    break

                imported_nodes = imported_nodes + nodes_for_this_data_layer
                new_node_ind = new_node_ind + 1
                if this_is_the_last_one:
                    break
        else:
            # this is not the last node of the graph - ignore it
            status = True

        return status

    def import_relay_ir(self, mod, subgraph_tensors):
        r""" Relay IR import to TIDL

        Parameters
        ----------
        mod : tvm.relay.Module
            Relay IR graph with subgraphs
        subgraph_tensors: dict
            Input/output tensors of subgraphs obtained from TVM graph execution

        Returns
        -------
        1: if TIDL import succeeds
        -1: if TIDL import fails
        0: if there are no subgraphs for TIDL offload
        """

        # Define return values
        import_succeed, import_fail, no_import = 1, -1, 0

        # Traverse Relay IR graph and generate a dictionary of all TIDL subgraphs
        all_nodes_main = {}
        traverse_func = functools.partial(traverse_expr, node_dict=all_nodes_main)
        relay.analysis.post_order_visit(mod['main'], traverse_func)
        tidl_subgraphs = []
        for node in all_nodes_main:
            if isinstance(node, relay.expr.GlobalVar):
                if self.tidl_target in node.name_hint:
                    tidl_subgraphs.append(node.name_hint)

        if len(tidl_subgraphs) == 0:
            return no_import

        # For each TIDL subgraph, import to TIDL and calibrate
        for tidl_subgraph in tidl_subgraphs:
            # Extract subgraph id and input/output tensor names from subgraph name
            subgraph_id = int(tidl_subgraph.replace(self.tidl_target+'_', ''))
            in_tensor_name = tidl_subgraph + '_i'
            out_tensor_name = tidl_subgraph + '_o'

            # Obtain input tensor from TVM graph execution
            input_fp = obtain_subgraph_tensor(subgraph_tensors, in_tensor_name)
            if input_fp is None:
                return import_fail
            if len(input_fp) > 1:
                print("Error - only 1 input tensor is supported for now!")
                return import_fail

            # Quantize input tensor into 8-bit integer (only support 1 input tensor)
            input_quant_vec, input_scale, input_signed = tensor_quant_flatten(input_fp[0],
                                                                              self.data_layout)

            # Initialize TIDL import
            if not self.tidl_import_init(input_scale, input_signed, input_fp[0].shape):
                return import_fail

            # Scan through all relay.expr.Call nodes and import each to TIDL
            all_nodes_tidl = {}
            traverse_func = functools.partial(traverse_expr, node_dict=all_nodes_tidl)
            relay.analysis.post_order_visit(mod[tidl_subgraph], traverse_func)
            for node in all_nodes_tidl:
                if isinstance(node, relay.expr.Call):
                    result = self.tidl_import_node(all_nodes_tidl, node)
                    if not result:
                        return import_fail

            # Import expr.Tuple node after importing all expr.call nodes
            for node in all_nodes_tidl:
                if isinstance(node, relay.expr.Tuple):
                    #node.fields: array of expr.call nodes
                    result = self.tidl_import_tuple_node(all_nodes_tidl, node)
                    if not result:
                        print('Error importing output tuple node')
                        return import_fail

            # Invoke TIDL optimization of the imported graph
            net_file = os.path.join(self.artifacts_folder,
                                    'tidl_subgraph'+str(subgraph_id)+'_net.bin')
            par_file = os.path.join(self.artifacts_folder,
                                    'tidl_subgraph'+str(subgraph_id)+'_params.bin')

            import_lib_optimize = self.import_lib.tidlImportOptimize
            import_lib_optimize.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
            import_lib_optimize.restype = ctypes.c_int
            net_fname = net_file.encode('utf-8')
            par_fname = par_file.encode('utf-8')

            if import_lib_optimize(net_fname, par_fname, subgraph_id) == 0:
                print('TIDL import optimization failed')
                return import_fail

            # Calibrate TIDL for the imported subgraph
            status, out_data_q = subgraph_calibration(self.calib_tool, input_quant_vec,
                                                      input_signed, net_file, par_file)
            if not status:
                return import_fail

            # Calculate scaling factor to convert output tensor to floating point
            # Obtain output tensor from TVM graph execution
            output_fp = obtain_subgraph_tensor(subgraph_tensors, out_tensor_name)

            # TODO: convert following lines into a function
            if output_fp is None:
                return import_fail
            if len(output_fp) != len(out_data_q):
                return import_fail
            output_signed = []
            output_scale = []
            for tensor in output_fp:
                # Find out if this output tensor is signed or unsigned
                output_signed.append(int(np.amin(tensor) < 0))
            for data_q in out_data_q:
                # Change data Q to scale - 255 is TIDL implementation specific
                output_scale.append(round(data_q/255.0, 5))

            # Generate subgraph configuration file
            subgraph_cfg_gen(self.artifacts_folder, subgraph_id, self.data_layout,
                             input_scale, input_signed, output_scale, output_signed)
        return import_succeed


class TIDLCompiler:
    """TIDL compiler module.

    This module tries to compile a given Relay IR graph to deploy on devices with TIDL.
    If compilation for TIDL succeeds, artifacts for heterogeneous compute with TIDL
    will be generated.

    Parameters
    ----------
    platform : string
        The platform to deploy the graph on.
    version : tuple
        The Processor-SDK version for the platform.
    **kwargs : keyword arguments to pass what's needed for Relay IR graph conversion
        num_tidl_subgraphs : int
            Number of subgraphs to run on TIDL
        tidl_tools_path : string
            Folder to TIDL tools
        artifacts_folder : string
            Folder to hold TIDL artifacts
    """

    def __init__(self, platform, version, max_num_layers=225, max_total_memory_mb=448, **kwargs):
        if platform == "AM57" and version >= (6, 3):
            # Set default values for AM57 6.3
            self.tidl_target = "tidl"
            self.num_tidl_subgraphs = 1
            self.artifacts_folder = None
            self.tidl_tools_path = None
            # Read arguments provided through regular args
            self.max_num_layers = max_num_layers
            self.max_total_memory_mb = max_total_memory_mb
            # Read arguments provided through **kwargs
            for key in ('num_tidl_subgraphs', 'artifacts_folder', 'tidl_tools_path'):
                if key in kwargs:
                    setattr(self, key, kwargs[key])
            assert self.artifacts_folder, "artifacts_folder must be specified for TIDL compilation"
        else:
            sys.exit("Unsupported TIDL platform or version!")

    def enable(self, mod_orig, params, graph_input):
        """ Enable TIDL compilation

        This function tries to partition and compile the given Relay IR graph.
        If it succeeds, artifacts for heterogeneous compute with TIDL will be
        generated, and the partitioned graph will be returned. Otherwise, it will
        return None.

        Parameters
        ----------
        mod_orig : tvm.relay.Module
            Original Relay IR graph
        params : dict of str to tvm.NDArray
            The parameter dict to be used by relay
        graph_input: dictionary
            A dictionary where the key is input name and the value is input tensor

        Returns
        -------
        mod : tvm.relay.Module
            Paritioned graph with subgraphs to run with TIDL
        status: int
            Status of TIDL compilation:
                1  - compilation success
                -1 - compilation failure
                0  - no compilation due to missing TIDL tools
        """

        mod = relay.transform.RemoveUnusedFunctions()(mod_orig)
        # Bind params so that weights will appear as constants instead of variables
        mod['main'] = bind_params_by_name(mod['main'], params)
        mod = relay.transform.FoldConstant()(mod)
        mod['main'] = RemoveMultiplyByOne().visit(mod['main'])
        mod = relay.transform.EliminateCommonSubexpr()(mod)

        #============= Find data layout of the original graph =============
        data_layout = find_data_layout(mod)

        #============= Annotation and graph partition ==============
        mod = tidl_annotation._merge_sequential_ops(mod)
        mod = relay.transform.AnnotateTarget(self.tidl_target)(mod)
        mod = relay.transform.MergeCompilerRegions()(mod)
        mod = relay.transform.PartitionGraph()(mod)
        mod = prune_subgraphs_with_multiple_inputs(mod, compiler=self.tidl_target)
        mod = reduce_subgraph_size(mod, max_num_layers=self.max_num_layers,
                                   max_total_memory_mb=self.max_total_memory_mb)
        mod = unpack_composites(mod)
        mod = prune_subgraphs(mod, compiler=self.tidl_target,
                              num_subgraphs_to_keep=self.num_tidl_subgraphs,
                              min_mac_threshold=1)
        with tvm.transform.PassContext(opt_level=3):
            convert_pass = [relay.transform.ConvertLayout({'nn.conv2d': ['NCHW', 'default']})]
            mod = tvm.transform.Sequential(convert_pass)(mod) # only affects non-TIDL subgraphs

        #============= Generate subgraph boundary tensors ==============
        subgraph_tensors = generate_subgraph_tensors(self.tidl_target, mod, params, graph_input)

        #================ Import the graph to TIDL =====================
        if self.tidl_tools_path is not None:
            tidl_calib_tool = os.path.join(self.tidl_tools_path, "eve_test_dl_algo_ref.out")
            tidl_import_lib = os.path.join(self.tidl_tools_path, "tidl_relayImport.so")
            if os.path.exists(tidl_calib_tool) and os.path.exists(tidl_import_lib):
                import_lib = ctypes.CDLL(tidl_import_lib, mode=ctypes.RTLD_GLOBAL)
                tidl_import = TIDLImport(import_lib, tidl_calib_tool, self.artifacts_folder,
                                         self.tidl_target, data_layout)
                import_status = tidl_import.import_relay_ir(mod, subgraph_tensors)
                _ctypes.dlclose(import_lib._handle)
                if import_status == 1:
                    print("TIDL import of Relay IR graph succeeded.")
                    print("TIDL artifacts are stored at " + self.artifacts_folder)
                    mod_final, status = mod, 1        # TIDL Compilation success
                elif import_status == -1:
                    print("TIDL import of Relay IR graph failed.")
                    mod_final, status = mod_orig, -1  # TIDL Compilation failure
                else:
                    print("There are no subgraphs for TIDL offload.")
                    mod_final, status = mod_orig, 0   # No TIDL compilation
            else:
                print("TIDL import lib does not exist. TIDL import skipped.")
                mod_final, status = mod_orig, 0       # No TIDL compilation
        else:
            print("TIDL tools path is not set. TIDL import skipped.")
            mod_final, status = mod_orig, 0           # No TIDL compilation

        return mod_final, status
