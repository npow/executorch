import argparse
import faulthandler
import json
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch
from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.fx.passes.operator_support import OperatorSupportBase

ROOT = Path(__file__).resolve().parent


# Repeated Arm layout passes otherwise deepcopy an expanding provenance tree.
original_current_meta = torch.fx.traceback.set_current_meta


@contextmanager
def bounded_current_meta(node, pass_name=""):
    node.meta.pop("from_node", None)
    with original_current_meta(node, pass_name):
        yield


torch.fx.traceback.set_current_meta = bounded_current_meta
original_node_source_init = torch.fx.traceback.NodeSource.__init__


def bounded_node_source_init(self, node, *args, **kwargs):
    if node is not None:
        node.meta.pop("from_node", None)
    original_node_source_init(self, node, *args, **kwargs)


torch.fx.traceback.NodeSource.__init__ = bounded_node_source_init


class BackboneBoundaries(OperatorSupportBase):
    def is_node_supported(self, submodules, node):
        return node.name not in {"add_47", "add_75"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=["backbone", "denoise"])
    parser.add_argument("--resume-quantized", action="store_true")
    parser.add_argument("--split-backbone", action="store_true")
    parser.add_argument("--skip-layout-propagation", action="store_true")
    args = parser.parse_args()
    label = args.name + ("_split" if args.split_backbone else "") + ("_plain" if args.skip_layout_propagation else "")
    if args.skip_layout_propagation:
        from executorch.backends.transforms.propagate_view_copy_permute_pass import PropagateViewCopyPermutePass
        from executorch.exir.pass_base import PassResult

        def retain_layout_operations(self, graph_module):
            return PassResult(graph_module, False)

        PropagateViewCopyPermutePass.call = retain_layout_operations
    torch.set_num_threads(8)
    spec = EthosUCompileSpec(
        "ethos-u85-256",
        system_config="Ethos_U85_SRAM_OSPI",
        memory_mode="Dedicated_Sram",
        config_ini="/Users/npow/alif/CMSIS-Executorch/.cmsis/ensemble_vela.ini",
    )
    spec.dump_intermediate_artifacts_to(str(ROOT / f"{label}_vela"))
    samples = torch.load(ROOT / "inputs.pt", weights_only=True)
    inputs = samples[args.name]
    if args.resume_quantized:
        exported = torch.export.load(ROOT / f"{args.name}_quantized.pt2")
    else:
        module = torch.export.load(ROOT / f"{args.name}.pt2").module()
        for node in list(module.graph.nodes):
            if node.op == "call_module" and node.target == "_guards_fn":
                module.graph.erase_node(node)
        module.graph.eliminate_dead_code()
        module.recompile()
        print("loaded", args.name, flush=True)
        quantizer = EthosUQuantizer(spec).set_global(get_symmetric_quantization_config(is_per_channel=True))
        prepared = prepare_pt2e(module, quantizer)
        print("prepared", flush=True)
        with torch.no_grad():
            prepared(*inputs)
        quantized = convert_pt2e(prepared)
        print("calibrated on 1 bundled sample; no quality evaluation", flush=True)
        exported = torch.export.export(quantized, inputs, strict=False)
        torch.export.save(exported, ROOT / f"{args.name}_quantized.pt2")
    print("quantized export saved; lowering", flush=True)
    edge = to_edge_transform_and_lower(
        exported, partitioner=[EthosUPartitioner(spec, additional_checks=[BackboneBoundaries()] if args.split_backbone else None)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    graph = edge.exported_program().graph_module
    ops = Counter(str(node.target) for node in graph.graph.nodes if node.op == "call_function")
    print("partitioned", json.dumps(ops, indent=2), flush=True)
    program = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    path = ROOT / f"{label}.pte"
    path.write_bytes(program.buffer)
    report = {"name": label, "pte_bytes": path.stat().st_size, "ops": dict(ops),
              "planned_buffer_sizes": [list(plan.non_const_buffer_sizes) for plan in program.executorch_program.execution_plan]}
    (ROOT / f"{label}_compile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
