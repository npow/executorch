# Octo-small on the Alif AppKit-E8

Experimental ExecuTorch 1.5.1 / Ethos-U85-256 benchmark. Full Octo policy timing
is **not yet measured**: the USB debug probe disconnected before the fixed
firmware could be tested on the full policy. Language encoding is precomputed
on the host and excluded from policy timing. No robot-task quality evaluation
has been performed. PTQ uses one bundled calibration sample.

The firmware retains the CMSIS `DevKit-E8` target name, but its PSRAM setup is
specifically for the **AppKit-E8 APS512XXN dual-octal memory**. Do not assume this
RAM initialization applies to a DevKit-E8 board.

## PSRAM regression

Variable read latency corrupted odd bytes after larger NPU workloads. The
controlled three-convolution reproducer had 93,945 bad bytes in its 209,616-byte
model readback. With fixed read latency, the same model completed in 1,642 ms
(944 ms NPU), all 360,448 output elements were finite, and the complete model
readback matched its source SHA256. Output RMSE against the host quantized graph
was 0.00000556. See `psram_fix_report.json` and `probe_exact3/`.

`firmware/src/appkit_psram.cpp` sets MR0 bit 5 on both x8 dies, uses eight read
clocks, sets controller WAIT to 7, and disables variable-latency detection.
Write latency remains four clocks. Runtime metadata stays in internal SRAM;
planned tensors and bulk scratch use external PSRAM. The current full-policy
candidate copies original Vela command bytes unchanged into SRAM. The earlier
controlled reproducer used serialized diagnostic commands; its saved source is
`probe_exact3/app_main_fixed_serialized.cpp`.

The previously verified Octo head-only result is 527 ms for 20 diffusion steps
producing four 7D actions. This excludes the vision/transformer backbone.

## Local workflow

These are research scripts for the original macOS workstation. Review the
absolute paths in the Python, shell, J-Link, and build configuration files before
using them elsewhere. Dependencies include CMSIS-Toolbox 2.15, Arm GCC 15.3.1,
SEGGER J-Link, pylink, PyTorch, torchao, ExecuTorch 1.5.1, and Vela 5.1.

Model checkpoints, compiled models, binary fixtures, firmware binaries, original
application backups, and build directories are intentionally excluded from Git.
`prepare_export.py` expects the separate `../octo_ternary/octo-pytorch` source and
`../octo_ternary/octo-pytorch-checkpoint` inputs. Their checkpoint hash is recorded
in `metadata.json`.

1. Run `prepare_export.py`, then `compile_ethosu.py backbone` and
   `compile_ethosu.py denoise` in the matching exporter environment. Run
   `prepare_fixtures.py` to write binary board inputs and the host reference. Fixture
   hashes are recorded in `fixture_audit.json`; `inputs.pt` contains source tensors.
2. Run `configure_firmware.py` and `bash build_firmware.sh` from this directory.
3. With exclusive board access, run `run_fixed_baseline.py`. It acquires
   `/tmp/alif-e8-1219599493.lock`, performs pin-reset recovery, flashes firmware,
   uploads and verifies models, and requests one cold policy execution.

Flashing replaces the HP application and uploads overwrite external RAM. Back up
the existing application and RAM assets, and arrange their restoration before
running hardware commands. `run_board.py` and `run_probe.py` are lower-level tools
and do not acquire the board lock themselves.

To reproduce the synthetic regression, use `compile_probe.py --kind exact3`,
flash the benchmark firmware, and run `run_probe.py --kind exact3 --modes 11`
while holding the same board lock. Current firmware uses unmodified commands;
use the archived diagnostic source only when comparing the original controlled
before/after records.

`status.json` records the last session state. The cross-model ACT timing there
is separate evidence and must not be reported as Octo latency. The original
application had not been restored at the last disconnect; restoration remained
with the ACT task that held the board. `fixed_baseline_manifest.json` records the
locally built candidate and model hashes; binary files are not committed.

Firmware compilation and Python syntax checks passed. Hardware evidence is
limited to the scopes and sample counts documented above.

Alif board files retain their vendor license in `firmware/LICENSE-Alif.txt`.
Arm board scaffolding retains its Apache-2.0 notices; the embedded module and
portable kernels retain their ExecuTorch license notices.

Developed with OpenAI Codex.
