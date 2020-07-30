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

import numpy as np
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.function import Function

class SubgraphSizeCounter(ExprVisitor):
    """
    Pass to count size of subgraph, both number of layers and estimated total memory usage.
    Used by SubgraphReducer pass.
    """
    def __init__(self):
        ExprVisitor.__init__(self)
        self.num_layers = 0
        self.total_memory = 0

    def get_total_memory_mb(self):
        return self.total_memory / (1024.0 * 1024.0)

    def visit_call(self, call):
        super().visit_call(call)
        # Don't count twice for composite op
        if not isinstance(call.op, Function):
            self.num_layers += 1
            # Add total size of weights (16 bits per)
            for arg in call.args:
                if isinstance(arg, tvm.relay.expr.Constant):
                    self.total_memory += 2 * np.prod(list(map(int, arg.checked_type.shape)))
            # Add activation size (8 bits per)
            if isinstance(call.checked_type, tvm.relay.TensorType):
                self.total_memory += np.prod(list(map(int, call.checked_type.shape)))

class ExprReplacer(ExprMutator):
    """
    Replaces call nodes in expr according to call_map
    """
    def __init__(self, call_map):
        ExprMutator.__init__(self)
        self.call_map = call_map

    def visit_call(self, call):
        if call in self.call_map:
            return self.call_map[call]
        return super().visit_call(call)

    def visit_tuple_getitem(self, op):
        if op in self.call_map:
            return self.call_map[op]
        return super().visit_tuple_getitem(op)

    def visit_tuple(self, tup):
        if tup in self.call_map:
            return self.call_map[tup]
        return super().visit_tuple(tup)

def find_common_ancestor(expr):
    """
    Find the closest common ancestor to expr0 and expr1.
    Returns distance from both.
    Used by SubgraphReducer pass.
    """
    class CommonAncestor(ExprVisitor):
        """
        Creates a map of nodes -> distance from expr
        """
        def __init__(self, expr, ancestors_from_previous=None):
            """
            Parameters
            ----------
            expr : tvm.relay.Expr
                Output node
            ancestors_from_previous : Dict[tvm.relay.ir.expr, int]
                CommonAncestor.ancestors_with_distance from previous traversal of a different
                output of the same graph. Will be used to terminate traversal early to avoid
                visiting nodes unnecessarily.
            """
            ExprVisitor.__init__(self)
            self.ancestors_with_distance = {expr: 0}
            self.call_outputs = {expr: []}
            self.ancestors_from_previous = ancestors_from_previous
            super().visit(expr)

        def _update(self, expr, expr_inputs):
            for arg in expr_inputs:
                if arg in self.call_outputs and expr not in self.call_outputs[arg]:
                    self.call_outputs[arg].append(expr)
                else:
                    self.call_outputs[arg] = [expr]

            if expr in self.call_outputs and len(self.call_outputs[expr]) > 0:
                self.ancestors_with_distance[expr] = \
                        max([self.ancestors_with_distance[output] \
                            for output in self.call_outputs[expr]]) + 1
            else:
                # Op did not have any outputs that we have already visited.
                self.ancestors_with_distance[expr] = 0

        def _terminate_early(self, node):
            # Second traversal (from fields[1] can stop when it reaches any node already visited
            # by first traversal).
            return self.ancestors_from_previous and node in self.ancestors_from_previous

        def visit_tuple_getitem(self, t):
            self._update(t, [t.tuple_value])
            if not self._terminate_early(t):
                super().visit_tuple_getitem(t)

        def visit_tuple(self, tup):
            self._update(tup, tup.fields)
            if not self._terminate_early(tup):
                super().visit_tuple(tup)

        def visit_call(self, call):
            self._update(call, call.args)
            if not self._terminate_early(call):
                # Don't visit function body
                # We don't care what's inside composite functions, we will just
                # remove the whole func.
                for arg in call.args:
                    super().visit(arg)

    def _find_common(field0, field1):
        common0 = CommonAncestor(field0)
        common1 = CommonAncestor(field1, common0.ancestors_with_distance)
        # Find common
        first_common_ancestor = None
        distance_to_0 = 999999
        distance_to_1 = 999999
        for node in common0.ancestors_with_distance:
            if node in common1.ancestors_with_distance:
                if common0.ancestors_with_distance[node] <= distance_to_0 and \
                   common1.ancestors_with_distance[node] <= distance_to_1:
                    first_common_ancestor = node
                    distance_to_0 = common0.ancestors_with_distance[node]
                    distance_to_1 = common1.ancestors_with_distance[node]
        assert first_common_ancestor is not None
        return first_common_ancestor, distance_to_0, distance_to_1

    first_common_ancestor = expr.fields[0]
    distance_to_field = [0 for i in range(len(expr.fields))]
    for i in range(1, len(expr.fields)):
        first_common_ancestor, dist0, dist1 = _find_common(first_common_ancestor, expr.fields[i])
        distance_to_field[i - 1] = dist0
        distance_to_field[i] = dist1

    return first_common_ancestor, distance_to_field

class SubgraphReducer(ExprMutator):
    """
    Removes a single op from end of subgraphs which exceed max_num_layers or max_total_memory_mb.
    If an op is removed, reduced will be set to True.
    """
    def __init__(self, mod, new_mod, max_num_layers=256, max_total_memory_mb=512, compiler="tidl"):
        ExprMutator.__init__(self)
        self.mod = mod
        self.new_mod = new_mod
        self.max_num_layers = max_num_layers
        self.max_total_memory_mb = max_total_memory_mb
        self.compiler = compiler
        self.reduced = False

    def visit_call(self, call):
        if isinstance(call.op, tvm.relay.expr.GlobalVar):
            name = call.op.name_hint
            if not self.mod[name].attrs or self.mod[name].attrs["Compiler"] != self.compiler:
                return super().visit_call(call)
            # Compute size of subgraph to see if we need to reduce it.
            counter = SubgraphSizeCounter()
            counter.visit(self.mod[name])
            if counter.num_layers > self.max_num_layers or \
               counter.get_total_memory_mb() > self.max_total_memory_mb:
                # Mark that we have reduced the subgraph size.
                self.reduced = True
                # "Inline" the last op only back into new main function.
                original_func = self.mod[name]
                last_op = original_func.body
                last_op_args = []
                if isinstance(last_op, tvm.relay.expr.Tuple):
                    # Subgraph has multiple outputs!
                    ancestor, _ = find_common_ancestor(last_op)
                    # Removing only op furthest from LCA greatly increase time taken for this pass.
                    # Instead, always delete all the way up to the lower common ancestor. This may
                    # cause more ops to be removed than is required, but it is much faster.
                    # TODO(trevmorr): Consider rewriting in C++ to improve speed.
                    # last_op_args = self._remove_op_furthest_from_lca(last_op, ancestor, distances)
                    last_op_args = ancestor.args
                elif isinstance(last_op, tvm.relay.expr.Call):
                    last_op_args = last_op.args
                elif isinstance(last_op, tvm.relay.expr.TupleGetItem):
                    for arg in last_op.tuple_value.args:
                        last_op_args.append(arg)
                else:
                    raise ValueError("Last op is not Call, Tuple, or TupleGetItem")
                # Gather new outputs of the subgraph - from removed op's inputs
                # This map will map Expr to index in new_outputs tuplea
                new_outputs = []
                last_op_input_to_new_output_map = {}
                if len(last_op_args) > 1:
                    for arg in last_op_args:
                        # Skip weights
                        if not isinstance(arg, tvm.relay.expr.Constant):
                            new_outputs.append(arg)
                            last_op_input_to_new_output_map[arg] = len(new_outputs) - 1
                    if len(new_outputs) > 1:
                        new_outputs_expr = relay.Tuple(new_outputs)
                    elif len(new_outputs) == 1:
                        new_outputs_expr = new_outputs[0]
                    else:
                        raise ValueError("No ops left in subgraph after reducing size")
                else:
                    new_outputs = [last_op_args[0]]
                    new_outputs_expr = new_outputs[0]
                subgraph_gv = relay.GlobalVar(name)

                # construct new func without last_op
                new_func = relay.Function(original_func.params, new_outputs_expr)
                new_func = new_func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
                new_func = new_func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
                new_func = new_func.with_attr("Compiler", self.compiler)
                new_func = new_func.with_attr("global_symbol", name)
                self.new_mod[subgraph_gv] = new_func
                args = []
                for arg in call.args:
                    args.append(super().visit(arg))
                new_expr = subgraph_gv(*args)
                if len(new_outputs) > 1:
                    call_map = {arg: relay.TupleGetItem(new_expr, index) \
                                for arg, index in last_op_input_to_new_output_map.items()}
                else:
                    call_map = {new_outputs[0]: new_expr}
                new_expr = ExprReplacer(call_map).visit(last_op)

                return new_expr
            if name != "main":
                # Transfer subgraph to new mod without modifying
                args = []
                for arg in call.args:
                    args.append(super().visit(arg))
                subgraph_gv = relay.GlobalVar(name)
                self.new_mod[subgraph_gv] = self.mod[name]
                return subgraph_gv(*args)
        return super().visit_call(call)

    def _remove_op_furthest_from_lca(self, last_op, ancestor, distances):
        """For subgraph with multiple outputs, pick output with logest path to least common
        ancestor. Returns list of new outputs.
        """
        def get_field(field):
            """Get field as it is, unless it is a TupleGetItem which we will remove."""
            if isinstance(field, tvm.relay.expr.Call):
                # Handle concat
                if isinstance(field.args[0], tvm.relay.expr.Tuple):
                    args = []
                    for f in field.args[0].fields:
                        args.append(f)
                    return args
                return [field]
            if isinstance(field, tvm.relay.expr.TupleGetItem):
                args = []
                for arg in field.tuple_value.args:
                    args.append(arg)
                return args
            if isinstance(field, tvm.relay.expr.Tuple):
                args = []
                for arg in field.fields:
                    args.append(arg)
                return args
            raise ValueError("New output of subgraph must be Call node.")

        def get_args(field):
            """Gather args from field, excluding exclude node"""
            args = []
            if isinstance(field, tvm.relay.expr.Call):
                for arg in field.args:
                    # Handle concat
                    if isinstance(arg, tvm.relay.expr.Tuple):
                        for f in arg.fields:
                            args.append(f)
                    else:
                        args.append(arg)
            elif isinstance(field, tvm.relay.expr.TupleGetItem):
                for arg in field.tuple_value.args:
                    args.append(arg)
            elif isinstance(field, tvm.relay.expr.Tuple):
                for arg in field.fields:
                    args.append(arg)
            else:
                raise ValueError("New output of subgraph must be Call node.")
            return args

        # All nodes come from same parent.
        if all([dist == 0 for dist in distances]):
            return ancestor.args
        # Remove node with longest path
        index_to_remove = np.argmax(distances)
        # field[index_to_remove] is further from LCA, remove it
        # by replacing it with its args.
        last_op_args = []
        for i in range(0, len(last_op.fields)):
            if i == index_to_remove:
                last_op_args += get_args(last_op.fields[i])
            else:
                last_op_args += get_field(last_op.fields[i])

        # Remove duplicates.
        seen = set()
        seen_add = seen.add
        last_op_args = [x for x in last_op_args if not (x in seen or seen_add(x))]
        return last_op_args

def reduce_subgraph_size(mod, max_num_layers=256, max_total_memory_mb=512):
    """
    Reduces size of subgraph to fit limitations.
    """
    # Counter just in case to avoid infinite loop.
    sanity_counter = 10000
    # SubgraphReducer removes one op if the subgraph is above the limits.
    # Repeated call SubgraphReducer until no subgraphs are reduced.
    while sanity_counter > 0:
        new_mod = tvm.IRModule()
        reducer = SubgraphReducer(mod, new_mod, max_num_layers, max_total_memory_mb)
        # TODO(trevmorr): Models with Preclude not supported (multiple functions other than main).
        new_mod['main'] = reducer.visit(mod["main"])
        # If no subgraphs where reduced in size, we are done.
        if not reducer.reduced:
            break
        mod = new_mod
        # Avoid infinite loop.
        sanity_counter -= 1
    return mod
