# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU export/edge regressions; only the unused QNN dequantization registry is stubbed."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.graph_signature import OutputKind

visitor = types.ModuleType("executorch.backends.qualcomm.builders.node_visitor")
visitor.dq_ops = set()
with patch.dict(sys.modules, {visitor.__name__: visitor}):
    spec = importlib.util.spec_from_file_location(
        "_qnn_expand_broadcast",
        Path(__file__).resolve().parents[1]
        / "_passes"
        / "expand_broadcast_tensor_shape.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ExpandBroadcastTensorShape = module.ExpandBroadcastTensorShape


VIEW = exir_ops.edge.aten.view_copy.default
UNSQUEEZE = exir_ops.edge.aten.unsqueeze_copy.default


class TestExpandBroadcastTensorShape(unittest.TestCase):
    def test_dynamic_width_executes_and_reuses_rank_promotion(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                return a + b, a - b

        width = torch.export.Dim("width", min=2, max=16)
        exported = torch.export.export(
            Model(),
            (torch.randn(3, 4), torch.randn(4)),
            dynamic_shapes={"a": {1: width}, "b": {0: width}},
        )
        ep = to_edge(exported).exported_program()
        gm = ExpandBroadcastTensorShape()(ep.graph_module).graph_module
        promotions = [node for node in gm.graph.nodes if node.target == UNSQUEEZE]
        self.assertEqual(len(promotions), 1)
        self.assertFalse(any(node.target == VIEW for node in gm.graph.nodes))
        for node in gm.graph.nodes:
            if node.target in (
                exir_ops.edge.aten.add.Tensor,
                exir_ops.edge.aten.sub.Tensor,
            ):
                self.assertIn(promotions[0], node.args)
        gm.graph.lint()
        for size in (2, 4, 7, 16):
            with self.subTest(size=size):
                inputs = (torch.randn(3, size), torch.randn(size))
                torch.testing.assert_close(gm(*inputs), Model()(*inputs))

    def test_two_leading_dimensions_preserve_symbolic_extent(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                return a * b

        width = torch.export.Dim("width", min=2, max=16)
        ep = to_edge(
            torch.export.export(
                Model(),
                (torch.randn(2, 3, 4), torch.randn(4)),
                dynamic_shapes={"a": {2: width}, "b": {0: width}},
            )
        ).exported_program()
        gm = ExpandBroadcastTensorShape()(ep.graph_module).graph_module
        promotions = [node for node in gm.graph.nodes if node.target == UNSQUEEZE]
        self.assertEqual(len(promotions), 2)
        self.assertEqual([node.meta["val"].dim() for node in promotions], [2, 3])
        for size in (2, 9):
            inputs = (torch.randn(2, 3, size), torch.randn(size))
            torch.testing.assert_close(gm(*inputs), (Model()(*inputs),))

    def test_static_same_rank_consumers_share_one_view(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                return a + b, a - b

        inputs = (torch.randn(3, 4), torch.randn(4))
        ep = to_edge(torch.export.export(Model(), inputs)).exported_program()
        gm = ExpandBroadcastTensorShape()(ep.graph_module).graph_module
        views = [node for node in gm.graph.nodes if node.target == VIEW]
        self.assertEqual(len(views), 1)
        self.assertFalse(any(node.target == UNSQUEEZE for node in gm.graph.nodes))
        for node in gm.graph.nodes:
            if node.target in (
                exir_ops.edge.aten.add.Tensor,
                exir_ops.edge.aten.sub.Tensor,
            ):
                self.assertIn(views[0], node.args)
        torch.testing.assert_close(gm(*inputs), Model()(*inputs))

    def test_rank_promotion_only_changes_the_consumer_that_needs_it(self):
        class Model(torch.nn.Module):
            def forward(self, a, c, b):
                return a + b, c + b

        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic):
                width = torch.export.Dim("width", min=2, max=16)
                shapes = {"a": {1: width}, "c": {0: width}, "b": {0: width}}
                inputs = (torch.randn(3, 4), torch.randn(4), torch.randn(4))
                ep = to_edge(
                    torch.export.export(
                        Model(), inputs, dynamic_shapes=shapes if dynamic else None
                    )
                ).exported_program()
                gm = ExpandBroadcastTensorShape()(ep.graph_module).graph_module
                for node in gm.graph.nodes:
                    if node.target == exir_ops.edge.aten.add.Tensor:
                        promoted = any(
                            isinstance(arg, torch.fx.Node)
                            and arg.target in (VIEW, UNSQUEEZE)
                            for arg in node.args
                        )
                        self.assertEqual(promoted, node.meta["val"].dim() == 2)
                runtime = (
                    (torch.randn(3, 7), torch.randn(7), torch.randn(7))
                    if dynamic
                    else inputs
                )
                torch.testing.assert_close(gm(*runtime), Model()(*runtime))

    def test_scalar_input_mutation_retains_rank_zero(self):
        class Model(torch.nn.Module):
            def forward(self, x, counter):
                position = torch.arange(x.shape[-1]) + counter
                counter.add_(x.shape[-1])
                return x + position.to(x.dtype)

        inputs = (torch.randn(1, 4), torch.tensor(0))
        manager = to_edge(torch.export.export(Model(), inputs))
        ep = manager.exported_program()
        gm = ExpandBroadcastTensorShape()(ep.graph_module).graph_module
        mutation_names = {
            output.arg.name
            for output in ep.graph_signature.output_specs
            if output.kind == OutputKind.USER_INPUT_MUTATION
        }
        self.assertTrue(mutation_names)
        for node in gm.graph.nodes:
            if node.name in mutation_names:
                self.assertEqual(node.meta["val"].dim(), 0)
                self.assertFalse(
                    any(
                        isinstance(arg, torch.fx.Node)
                        and arg.target in (VIEW, UNSQUEEZE)
                        for arg in node.args
                    )
                )
        counter = torch.tensor(0)
        expected_output = ep.module()(inputs[0], counter)
        self.assertEqual(counter.dim(), 0)
        self.assertEqual(counter.item(), 4)
        torch.testing.assert_close(expected_output, inputs[0] + torch.arange(4))
        manager.to_executorch()

    def test_dynamic_operand_mutation_does_not_inherit_promoted_rank(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                output = a + b
                b.add_(1)
                return output

        width = torch.export.Dim("width", min=2, max=16)
        ep = to_edge(
            torch.export.export(
                Model(),
                (torch.randn(3, 4), torch.randn(4)),
                dynamic_shapes={"a": {1: width}, "b": {0: width}},
            )
        ).exported_program()
        gm = ExpandBroadcastTensorShape()(ep.graph_module).graph_module
        a, b = torch.randn(3, 7), torch.randn(7)
        original_b = b.clone()
        output = ep.module()(a, b)
        self.assertEqual(b.dim(), 1)
        torch.testing.assert_close(b, original_b + 1)
        torch.testing.assert_close(output, a + original_b)
        gm.graph.lint()

    def test_dynamic_expand_copy_promotes_without_unbound_shape_arguments(self):
        class Model(torch.nn.Module):
            def forward(self, b):
                return b.expand(2, 3, b.shape[0])

        width = torch.export.Dim("width", min=2, max=16)
        ep = to_edge(
            torch.export.export(
                Model(), (torch.randn(4),), dynamic_shapes={"b": {0: width}}
            )
        ).exported_program()
        gm = ExpandBroadcastTensorShape()(ep.graph_module).graph_module
        self.assertEqual(sum(node.target == UNSQUEEZE for node in gm.graph.nodes), 2)
        for size in (2, 7):
            sample = torch.randn(size)
            torch.testing.assert_close(gm(sample), (Model()(sample),))


if __name__ == "__main__":
    unittest.main()
