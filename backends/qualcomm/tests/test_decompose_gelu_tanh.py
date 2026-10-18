# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only tests that import the pass without initializing the QNN SDK."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import executorch
import torch

QUALCOMM_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DecomposeGeluTanh = load_module(
    "_qnn_decompose_gelu_tanh",
    QUALCOMM_ROOT / "_passes" / "decompose_gelu_tanh.py",
).DecomposeGeluTanh


class Gelu(torch.nn.Module):
    def __init__(self, approximate):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate=self.approximate)


class TestDecomposeGeluTanh(unittest.TestCase):
    def test_tanh_matches_pytorch_across_transition_and_saturation(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                x = torch.cat(
                    (
                        torch.linspace(-8, 8, 2001),
                        torch.tensor([-65504, -1000, -100, -0.0, 0, 100, 1000, 65504]),
                    )
                ).to(dtype)
                original = Gelu("tanh")
                graph_module = torch.export.export(original, (x,)).module()
                result = DecomposeGeluTanh()(graph_module).graph_module
                torch.testing.assert_close(
                    result(x),
                    original(x),
                    rtol=2e-3 if dtype == torch.float16 else 1e-5,
                    atol=1e-3 if dtype == torch.float16 else 1e-6,
                )
                targets = [node.target for node in result.graph.nodes]
                self.assertNotIn(torch.ops.aten.gelu.default, targets)
                self.assertIn(torch.ops.aten.sigmoid.default, targets)
                self.assertNotIn(torch.ops.aten.pow.Tensor_Scalar, targets)
                self.assertTrue(result(x).isfinite().all())

    def test_exact_gelu_remains_bit_identical(self):
        x = torch.linspace(-4, 4, 101)
        graph_module = torch.export.export(Gelu("none"), (x,)).module()
        expected = graph_module(x)
        result = DecomposeGeluTanh()(graph_module).graph_module
        torch.testing.assert_close(result(x), expected, rtol=0, atol=0)
        self.assertEqual(
            sum(
                node.target == torch.ops.aten.gelu.default
                for node in result.graph.nodes
            ),
            1,
        )

    def test_omitted_approximation_remains_exact(self):
        sample = torch.randn(2, 3)
        graph_module = torch.export.export(Gelu("none"), (sample,)).module()
        for node in graph_module.graph.nodes:
            if node.target == torch.ops.aten.gelu.default:
                node.kwargs = {}
        graph_module.recompile()
        expected = graph_module(sample)
        result = DecomposeGeluTanh()(graph_module).graph_module
        torch.testing.assert_close(result(sample), expected, rtol=0, atol=0)
        self.assertTrue(
            any(
                node.target == torch.ops.aten.gelu.default
                for node in result.graph.nodes
            )
        )

    def test_mixed_graph_preserves_exact_branch_and_multiple_users(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                tanh = torch.nn.functional.gelu(x, approximate="tanh")
                return tanh + tanh, torch.nn.functional.gelu(x)

        x = torch.randn(2, 3, 5).transpose(0, 2)
        model = Model()
        graph_module = torch.export.export(model, (x,)).module()
        result = DecomposeGeluTanh()(graph_module).graph_module
        torch.testing.assert_close(result(x), model(x), atol=1e-6, rtol=1e-5)
        self.assertEqual(
            sum(
                node.target == torch.ops.aten.gelu.default
                for node in result.graph.nodes
            ),
            1,
        )
        result.graph.lint()
        count = len(list(result.graph.nodes))
        second = DecomposeGeluTanh()(result).graph_module
        self.assertEqual(len(list(second.graph.nodes)), count)
        torch.testing.assert_close(second(x), result(x), rtol=0, atol=0)

    def test_dynamic_batch_and_scalar_metadata(self):
        x = torch.randn(3, 7)
        exported = torch.export.export(
            Gelu("tanh"),
            (x,),
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=8)}},
        )
        result = DecomposeGeluTanh()(exported.module()).graph_module
        sample = torch.randn(5, 7)
        torch.testing.assert_close(
            result(sample), Gelu("tanh")(sample), atol=1e-6, rtol=1e-5
        )
        for node in result.graph.nodes:
            if node.op == "call_function":
                self.assertIn("val", node.meta)


class TestGeluVisitorApproximation(unittest.TestCase):
    def setUp(self):
        package = "_qnn_gelu_builder_tests"
        base = types.ModuleType(package + ".node_visitor")
        base.NodeVisitor = type("NodeVisitor", (), {})
        registry = types.ModuleType(package + ".node_visitor_manager")
        registry.register_node_visitor = lambda visitor: visitor
        constants = types.ModuleType(package + ".qnn_constants")
        constants.OpGelu = types.SimpleNamespace(op_name="Gelu")
        constants.QNN_OP_PACKAGE_NAME_QTI_AISW = "qti.aisw"
        self.adapter = Mock()
        modules = {
            package: types.ModuleType(package),
            package + ".node_visitor": base,
            package + ".node_visitor_manager": registry,
            package + ".qnn_constants": constants,
            "executorch.backends.qualcomm.python.PyQnnManagerAdaptor": self.adapter,
        }
        backends = types.SimpleNamespace(
            qualcomm=types.SimpleNamespace(
                python=types.SimpleNamespace(PyQnnManagerAdaptor=self.adapter)
            )
        )
        with patch.dict(sys.modules, modules), patch.object(
            executorch, "backends", backends, create=True
        ):
            self.visitor_class = load_module(
                package + ".op_gelu", QUALCOMM_ROOT / "builders" / "op_gelu.py"
            ).GeluVisitor

    def test_tanh_without_decomposition_falls_back_before_creating_qnn_op(self):
        node = Mock(kwargs={"approximate": "tanh"})
        with self.assertWarnsRegex(UserWarning, "must be decomposed"):
            self.assertIsNone(self.visitor_class().define_node(node, {}))
        self.adapter.PyQnnOpWrapper.assert_not_called()

    def test_default_and_explicit_exact_still_create_native_gelu(self):
        for kwargs in ({}, {"approximate": "none"}):
            with self.subTest(kwargs=kwargs):
                visitor = self.visitor_class()
                visitor.get_node = Mock()
                visitor.get_tensor = Mock()
                visitor.define_tensor = Mock()
                node = types.SimpleNamespace(args=(Mock(),), kwargs=kwargs, name="gelu")
                result = visitor.define_node(node, {})
                self.assertIs(result, self.adapter.PyQnnOpWrapper.return_value)
                self.adapter.PyQnnOpWrapper.assert_called_with(
                    "gelu", "qti.aisw", "Gelu"
                )


if __name__ == "__main__":
    unittest.main()
