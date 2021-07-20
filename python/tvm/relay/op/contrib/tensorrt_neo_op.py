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
"""TensorRT supported operators."""
import logging
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    wildcard,
    rewrite,
    is_op,
    is_constant,
)

logger = logging.getLogger("TensorRT")


def rewrite_batch_matmul(mod):
    class BatchMatmulRewriteRule1(DFPatternCallback):
        def __init__(self):
            super(BatchMatmulRewriteRule1, self).__init__()
            self.A = wildcard()
            self.B = wildcard()
            transpose = is_op("transpose")
            batch_matmul = is_op("nn.batch_matmul")
            expr = batch_matmul(self.A, transpose(self.B).has_attr({"axes": [0, 2, 1]}))
            self.pattern = expr

        def callback(self, pre, post, node_map):
            A = node_map[self.A][0]
            B = node_map[self.B][0]
            return relay.op.neo_batch_matmul(A, B, False)

    class BatchMatmulRewriteRule2(DFPatternCallback):
        def __init__(self):
            super(BatchMatmulRewriteRule2, self).__init__()
            self.A = wildcard()
            self.B = is_constant()
            batch_matmul = is_op("nn.batch_matmul")
            expr = batch_matmul(self.A, self.B)
            self.pattern = expr

        def callback(self, pre, post, node_map):
            A = node_map[self.A][0]
            B = node_map[self.B][0]
            return relay.op.neo_batch_matmul(A, relay.transpose(B, axes=[0, 2, 1]), False)

    expr = mod["main"]
    expr = rewrite(BatchMatmulRewriteRule1(), expr)
    expr = rewrite(BatchMatmulRewriteRule2(), expr)
    mod = tvm.IRModule.from_expr(expr)
    return mod
