/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file neo_op.cc
 * \brief Implementation of temporary Neo operators
 */

#include <tvm/relay/attrs/neo_op.h>

#include "../op_common.h"

namespace tvm {
namespace relay {

// relay.neo_batch_matmul
TVM_REGISTER_NODE_TYPE(NeoBatchMatmulAttrs);

bool NeoBatchMatmulRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* A = types[0].as<TensorTypeNode>();
  const auto* B = types[1].as<TensorTypeNode>();
  if (A == nullptr || B == nullptr) return false;
  const auto* param = attrs.as<NeoBatchMatmulAttrs>();
  ICHECK_EQ(A->shape.size(), B->shape.size());
  ICHECK_EQ(A->shape.size(), 3);

  Array<tvm::PrimExpr> oshape;
  bool is_dyn = false;
  // batch
  if (A->shape[0].as<tir::AnyNode>() != nullptr || B->shape[0].as<tir::AnyNode>() != nullptr) {
    is_dyn = true;
    oshape.push_back(Any());
  } else {
    oshape.push_back(max(A->shape[0], B->shape[0]));
  }
  // M
  if (A->shape[1].as<tir::AnyNode>() != nullptr) {
    is_dyn = true;
    oshape.push_back(Any());
  } else {
    oshape.push_back(A->shape[1]);
  }
  // N
  if (B->shape[1].as<tir::AnyNode>() != nullptr) {
    is_dyn = true;
    oshape.push_back(Any());
  } else {
    if (param->transb) {
      oshape.push_back(B->shape[1]);
    } else {
      oshape.push_back(B->shape[2]);
    }
  }

  if (!is_dyn) {
    ICHECK(reporter->AssertEQ(A->shape[0], B->shape[0]) || reporter->AssertEQ(A->shape[0], 1) ||
           reporter->AssertEQ(B->shape[0], 1))
        << "MlasMatmulRel: batch dimensions don't match, "
        << " A shape=" << A->shape << ", B shape=" << B->shape;
    if (param->transb) {
      ICHECK(reporter->AssertEQ(A->shape[2], B->shape[2]))
          << "MlasMatmulRel (transb=True): shapes of A and B are inconsistent, "
          << " A shape=" << A->shape << ", B shape=" << B->shape;
    } else {
      ICHECK(reporter->AssertEQ(A->shape[2], B->shape[1]))
          << "MlasMatmulRel (transb=False): shapes of A and B are inconsistent, "
          << " A shape=" << A->shape << ", B shape=" << B->shape;
    }
  }

  reporter->Assign(types[2], TensorType(oshape, A->dtype));
  return true;
}

Expr MakeNeoBatchMatmul(Expr x, Expr y, bool transb) {
  auto attrs = make_object<NeoBatchMatmulAttrs>();
  attrs->transb = transb;
  static const Op& op = Op::Get("neo_batch_matmul");
  return Call(op, {x, y}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.neo_batch_matmul").set_body_typed(MakeNeoBatchMatmul);

RELAY_REGISTER_OP("neo_batch_matmul")
    .describe(R"code(Computes batch matrix multiplication

.. math::

  batch\_matmul(A, B)[i, :, :] = batch_matmul(A[i, :, :], B[i, :, :]^T, transb=True)
  or batch\_matmul(A, B)[i, :, :] = batch_matmul(A[i, :, :], B[i, :, :], transb=False)

- **A**: `(b, m, k)`
- **B**: `(b, n, k)` or `(b, k, n)`
- **out**: `(b, m, n)`

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("A", "3D Tensor", "First input.")
    .add_argument("B", "3D Tensor", "Second input.")
    .set_support_level(10)
    .add_type_rel("NeoBatchMatmul", NeoBatchMatmulRel);
}  // namespace relay
}  // namespace tvm
