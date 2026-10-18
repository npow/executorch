# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from executorch.exir.pass_base import ExportPass


class DecomposeGeluTanh(ExportPass):
    """Preserve tanh GELU when lowering to QNN's non-parameterized GELU op."""

    def call_operator(self, op, args, kwargs, meta):
        if (
            op != torch.ops.aten.gelu.default
            or kwargs.get("approximate", "none") != "tanh"
        ):
            return super().call_operator(op, args, kwargs, meta)

        # (1 + tanh(z)) / 2 = sigmoid(2z); factoring avoids the cubic temporary.
        x = args[0]
        coefficient = math.sqrt(8 / math.pi)
        scaled = super().call_operator(
            torch.ops.aten.mul.Scalar, (x, coefficient * 0.044715), {}, meta
        )
        quadratic = super().call_operator(
            torch.ops.aten.mul.Tensor, (scaled, x), {}, meta
        )
        polynomial = super().call_operator(
            torch.ops.aten.add.Scalar, (quadratic, coefficient), {}, meta
        )
        argument = super().call_operator(
            torch.ops.aten.mul.Tensor, (x, polynomial), {}, meta
        )
        sigmoid = super().call_operator(
            torch.ops.aten.sigmoid.default, (argument,), {}, meta
        )
        return super().call_operator(torch.ops.aten.mul.Tensor, (x, sigmoid), {}, meta)
