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
# pylint: disable=invalid-name,unused-argument
"""Strategy and AlterOpLayout functions of MLAS operators"""
import tvm
from tvm import topi
from .strategy import wrap_topi_schedule
from . import op as reg


# Neo_batch_matmul
# Neo_batch_matmul strategy
@tvm.target.override_native_generic_func("neo_batch_matmul_strategy")
def neo_batch_matmul_strategy(attrs, inputs, out_type, target):
    """neo_batch_matmul generic strategy"""
    return None


@neo_batch_matmul_strategy.register(["cuda"])
def neo_batch_matmul_strategy_cuda(attrs, inputs, out_type, target):
    """neo_batch_matmul strategy"""
    strategy = reg.OpStrategy()

    def wrap_compute_neo_batch_matmul(topi_compute):
        """wrap neo_batch_matmul topi compute"""

        def _compute_neo_batch_matmul(attrs, inputs, out_type):
            args = [inputs[0], inputs[1], attrs.transb]
            return [topi_compute(*args)]

        return _compute_neo_batch_matmul

    strategy.add_implementation(
        wrap_compute_neo_batch_matmul(topi.neo_batch_matmul),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="neo_batch_matmul",
        plevel=1,
    )
    return strategy


reg.register_strategy("neo_batch_matmul", neo_batch_matmul_strategy)
reg.register_pattern("neo_batch_matmul", reg.OpPattern.OUT_ELEMWISE_FUSABLE)
