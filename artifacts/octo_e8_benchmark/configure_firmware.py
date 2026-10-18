from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent / "firmware"
project = ROOT / "cmsis-executorch.cproject.yml"
data = yaml.safe_load(project.read_text())
data["project"]["groups"] = [{"group": "App", "files": [
    {"file": "./src/app_main.cpp"},
    {"file": "./src/appkit_psram.cpp"},
    {"file": "./src/arm_embedded_module.cpp"},
    {"file": "./src/arm_embedded_module.hpp"},
    {"file": "./src/history_layout_copy.cpp"},
    {"file": "./src/history_layout_cat.cpp"},
]}]
project.write_text(yaml.safe_dump(data, sort_keys=False))
layer = ROOT / "board/DevKit-E8/Board-U85.clayer.yml"
data = yaml.safe_load(layer.read_text())
data["layer"]["define"] = [x for x in data["layer"]["define"] if not (
    isinstance(x, str) and x in ["APP_DISPLAY", "APP_BUTTONS", "APP_INTERACTIVE"]
)]
for component in ["BSP:OSPI PSRAM XIP", "BSP:OSPI PSRAM XIP:APS PSRAM", "Device:SOC Peripherals:OSPI"]:
    if {"component": component} not in data["layer"]["components"]:
        data["layer"]["components"].append({"component": component})
layer.write_text(yaml.safe_dump(data, sort_keys=False))
for group in data["layer"]["groups"]:
    group["files"] = [f for f in group["files"] if f["file"] not in ["./display.c", "./buttons.c"]]
layer.write_text(yaml.safe_dump(data, sort_keys=False))
ai = ROOT / "ai_layer"
ai.mkdir(exist_ok=True)
(ai / "ai_layer.clayer.yml").write_text(yaml.safe_dump({"layer": {
    "type": "AI", "packs": [{"pack": "PyTorch::ExecuTorch@1.5.1"}],
    "define": [{"ET_LOG_ENABLED": 1}, {"ET_MIN_LOG_LEVEL": "Error"},
               {"RTE_ML_EXECUTORCH_OP_PORTABLE_TO_DIM_ORDER_COPY": 1},
               {"RTE_ML_EXECUTORCH_OP_PORTABLE_CAT": 1}],
    "components": [{"component": c} for c in [
        "Machine Learning:ExecuTorch:Runtime",
        "Machine Learning:ExecuTorch:Kernel Utils",
        "Machine Learning:ExecuTorch:Kernel Registration",
        "Machine Learning:ExecuTorch:Backend EthosU",
        "Machine Learning:ExecuTorch Operators:Quantized dequantize",
        "Machine Learning:ExecuTorch Operators:Quantized quantize",
        "Machine Learning:ExecuTorch Operators:Portable view_copy",
        "Machine Learning:ExecuTorch Operators:Portable alias_copy",
    ]],
}}, sort_keys=False))
