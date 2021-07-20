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
# pylint: disable=invalid-name
"""MLAS relay operators"""
from . import _make


def neo_batch_matmul(A, B, transb=True):
    r"""Computes batch matrix multiplication of `A` and `B` when `A` and `B` are
    data in batch. Supports broadcasting for batch dimension.

    Parameters
    ----------
    A : tvm.te.Tensor
        3-D with shape [batch, M, K]

    B : tvm.te.Tensor
        3-D with shape [batch, N, K] or [batch, K, N]

    transb : bool
        Whether the B matrix is transposed.

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    return _make.neo_batch_matmul(A, B, transb)
