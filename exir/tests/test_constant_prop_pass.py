# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import unittest

import torch
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from torch.export.exported_program import ConstantArgument


class TestScalarConstantPropagation(unittest.TestCase):
    def test_scalar_used_by_tensor_operation(self) -> None:
        class Scale(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.register_buffer("scale", torch.tensor(value))

            def forward(self, x):
                return x * self.scale.item()

        for value in (True, 3, 2.5):
            with self.subTest(value=value):
                model = Scale(value)
                inputs = (torch.arange(3, dtype=torch.float32),)
                program = constant_prop_pass(torch.export.export(model, inputs))
                program.validate()
                torch.testing.assert_close(program.module()(*inputs), model(*inputs))
                self.assertNotIn(
                    torch.ops.aten.item.default,
                    [node.target for node in program.graph.nodes],
                )

    def test_scalar_keyword_argument(self) -> None:
        class Add(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("alpha", torch.tensor(2.5))

            def forward(self, x):
                return torch.add(x, x, alpha=self.alpha.item())

        model = Add()
        inputs = (torch.ones(3),)
        program = constant_prop_pass(torch.export.export(model, inputs))
        program.validate()
        torch.testing.assert_close(program.module()(*inputs), model(*inputs))
        add = next(
            node
            for node in program.graph.nodes
            if node.target == torch.ops.aten.add.Tensor
        )
        self.assertEqual(add.kwargs["alpha"], 2.5)

    def test_scalar_output_signature(self) -> None:
        class Scale(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.tensor(2.5))

            def forward(self, x):
                return x * self.scale.item(), self.scale.item()

        model = Scale()
        inputs = (torch.ones(3),)
        program = constant_prop_pass(torch.export.export(model, inputs))
        program.validate()
        torch.testing.assert_close(program.module()(*inputs), model(*inputs))
        self.assertIsInstance(
            program.graph_signature.output_specs[1].arg, ConstantArgument
        )
        self.assertEqual(program.graph_signature.output_specs[1].arg.value, 2.5)
        program = constant_prop_pass(program)
        artifact = io.BytesIO()
        torch.export.save(program, artifact)
        artifact.seek(0)
        restored = torch.export.load(artifact)
        torch.testing.assert_close(restored.module()(*inputs), model(*inputs))

    def test_dynamic_dimension_remains_dynamic(self) -> None:
        class Scale(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.tensor(2.5))

            def forward(self, x):
                return x.reshape(x.shape[0], -1) * self.scale.item()

        model = Scale()
        program = torch.export.export(
            model,
            (torch.ones(3, 2),),
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=2)}},
        )
        program = constant_prop_pass(program)
        program.validate()
        for size in (3, 7):
            inputs = (torch.randn(size, 2),)
            torch.testing.assert_close(program.module()(*inputs), model(*inputs))
        self.assertIn(
            torch.ops.aten.sym_size.int,
            [node.target for node in program.graph.nodes],
        )

    def test_random_operation_remains_live(self) -> None:
        class Random(torch.nn.Module):
            def forward(self, x):
                return x + torch.rand(2)

        program = torch.export.export(Random(), (torch.ones(2),))
        rng_state = torch.random.get_rng_state().clone()
        program = constant_prop_pass(program)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
        module = program.module()
        self.assertFalse(torch.equal(module(torch.ones(2)), module(torch.ones(2))))

    def test_mutated_buffer_scalar_remains_live(self) -> None:
        class Counter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("counter", torch.tensor(1.0))

            def forward(self, x):
                self.counter.add_(1)
                return x * self.counter.item()

        program = torch.export.export(Counter(), (torch.ones(2),)).run_decompositions(
            {}
        )
        program = constant_prop_pass(program)
        program.validate()
        module = program.module()
        torch.testing.assert_close(module(torch.ones(2)), torch.full((2,), 2.0))
        torch.testing.assert_close(module(torch.ones(2)), torch.full((2,), 3.0))


if __name__ == "__main__":
    unittest.main()
