import argparse
import ctypes
import hashlib
import json
import math
import statistics
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, "/private/tmp/dcops-pylink")
import pylink

ROOT = Path(__file__).resolve().parent
MAILBOX = 0x20001294


def main():
    global MAILBOX
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument("--reuse-upload", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    elf = ROOT / "firmware/out/cmsis-executorch/DevKit-E8/Debug/cmsis-executorch.elf"
    nm = "/Users/npow/.vcpkg/artifacts/2139c4c6/compilers.arm.arm.none.eabi.gcc/15.3.1/bin/arm-none-eabi-nm"
    symbols = {line.split()[2]: int(line.split()[0], 16)
               for line in subprocess.check_output([nm, "-n", str(elf)], text=True).splitlines()
               if len(line.split()) == 3 and line.split()[0] != "U"}
    MAILBOX = symbols["octo_benchmark"]
    library = pylink.library.Library(dllpath="/Applications/SEGGER/JLink_V986a/libjlinkarm.9.86.1.dylib")
    link = pylink.JLink(lib=library)
    link.open(serial_no=1219599493)
    link.set_tif(pylink.enums.JLinkInterfaces.SWD)
    link.connect("AE822FA0E5597LS0_M55_HP", speed=1000)
    link.exec_command("SetEnableFlashbank 0xA0000000=0")
    link.exec_command("SetAllowFlashCache=0")
    write = link._dll.JLINKARM_WriteMemEx
    write.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
    write.restype = ctypes.c_int
    verified_files = []

    def staging_job(address, size, operation):
        sequence = link.memory_read32(MAILBOX + 232, 1)[0] + 1
        link.memory_write32(MAILBOX + 216, [address, size])
        link.memory_write32(MAILBOX + 236, [operation])
        link.memory_write32(MAILBOX + 228, [sequence])
        deadline = time.monotonic() + 30
        while link.memory_read32(MAILBOX + 232, 1)[0] != sequence:
            if time.monotonic() >= deadline:
                raise TimeoutError("firmware SRAM transfer/CRC job timed out")
            time.sleep(0.001)
        return link.memory_read32(MAILBOX + 224, 1)[0]

    def upload(path, address):
        raw = path.read_bytes()
        expected_crc = zlib.crc32(raw)
        if args.reuse_upload:
            retained_crc = staging_job(address, len(raw), 2)
            print("retained CRC", path.name, hex(retained_crc), "expected", hex(expected_crc), flush=True)
            if retained_crc == expected_crc:
                verified_files.append({"file": path.name, "bytes_checked": len(raw), "crc32": expected_crc,
                                       "source_sha256": hashlib.sha256(raw).hexdigest(), "retained": True})
                return
        for offset in range(0, len(raw), 64 * 1024):
            block = raw[offset:offset + 64 * 1024]
            if len(block) % 4:
                block += bytes(4 - len(block) % 4)
            buffer = ctypes.create_string_buffer(block)
            count = write(symbols["temp_pool"], len(block), buffer, 4)
            if count != len(block):
                raise RuntimeError(f"write at {address + offset:#x}: {count}/{len(block)}")
            block_crc = staging_job(address + offset, len(block), 1)
            if block_crc != zlib.crc32(block):
                raise RuntimeError(f"SRAM-staged transfer CRC mismatch for {path.name} at {offset}")
            if offset % (1024 * 1024) == 0:
                print(f"upload {path.name}: {offset}/{len(raw)}", flush=True)
        first = bytes(link.memory_read8(address, min(64, len(raw))))
        last_offset = max(0, len(raw) - 64)
        last = bytes(link.memory_read8(address + last_offset, len(raw) - last_offset))
        if first != raw[:len(first)] or last != raw[last_offset:]:
            raise RuntimeError(f"upload boundary verification failed: {path.name}")
        print("uploaded", path.name, len(raw), flush=True)
        actual_crc = staging_job(address, len(raw), 2)
        if actual_crc != expected_crc:
            raise RuntimeError(f"full-file CRC mismatch for {path.name}")
        verified_files.append({"file": path.name, "bytes_checked": len(raw), "crc32": actual_crc,
                               "source_sha256": hashlib.sha256(raw).hexdigest(), "retained": False})

    try:
        words = link.memory_read32(MAILBOX, 8)
        print("initial mailbox", words, flush=True)
        if words[0] != 0x4F43544F:
            raise RuntimeError("benchmark firmware is not running")
        if args.upload:
            link.set_speed(15000)
            if words[1] != 2 or words[2] != 0:
                raise RuntimeError("firmware is not waiting for model upload")
            if not args.head_only and (ROOT / "backbone.pte").stat().st_size > 28 * 1024 * 1024:
                raise RuntimeError("backbone exceeds reserved model region")
            files = [("denoise.pte", 0xA1C00000), ("schedule.bin", 0xA1E8C000)]
            if args.head_only:
                files += [("embedding.bin", 0xA1E90000)]
            else:
                files += [("backbone.pte", 0xA0000000), ("primary.bin", 0xA1E00000),
                          ("wrist.bin", 0xA1E60000), ("language.bin", 0xA1E80000)]
            for name, address in files:
                upload(ROOT / name, address)
            link.set_speed(1000)
            link.memory_write32(MAILBOX + 12, [0 if args.head_only else (ROOT / "backbone.pte").stat().st_size,
                                              (ROOT / "denoise.pte").stat().st_size])
            link.memory_write32(MAILBOX + 8, [2 if args.head_only else 1])
        deadline = time.monotonic() + args.timeout
        previous = None
        heartbeat = time.monotonic()
        while time.monotonic() < deadline:
            words = link.memory_read32(MAILBOX, 26)
            state = (words[1], words[5], words[6], words[7])
            if state != previous:
                print("state/error/run/phase", state, "total_ms", words[20:26], flush=True)
                previous = state
            if time.monotonic() - heartbeat >= 30:
                npu = link.memory_read32(0x49042000, 9)
                print("NPU status/QREAD/QSIZE", hex(npu[1]), npu[6], npu[8],
                      "device_ticks", link.memory_read32(symbols["octo_ticks"], 1)[0], flush=True)
                heartbeat = time.monotonic()
            if words[1] >= 90:
                message = bytes(link.memory_read8(symbols["octo_last_error"], 2048)).split(b"\0", 1)[0].decode(errors="replace")
                print("ExecuTorch error:", message, flush=True)
                failure = {"mailbox_words": words, "runtime_message": message,
                           "transfer_verification": verified_files}
                if "octo_npu_snapshot" in symbols:
                    failure["npu_snapshot"] = link.memory_read32(symbols["octo_npu_snapshot"], 163)
                (ROOT / "board_failure.json").write_text(json.dumps(failure, indent=2))
                raise RuntimeError(f"board failure: {state}")
            if words[1] == 5:
                actions = struct.unpack("<28f", bytes(link.memory_read8(MAILBOX + 104, 112)))
                device_embedding = bytes(link.memory_read8(symbols["embedding"], 384 * 4))
                actual = struct.unpack("<384f", device_embedding)
                reference = struct.unpack("<384f", (ROOT / "embedding.bin").read_bytes())
                quantized_reference = struct.unpack("<384f", (ROOT / "quantized_embedding.bin").read_bytes())
                quantized_norm = math.sqrt(sum(a * a for a in actual) * sum(b * b for b in quantized_reference))
                norm = math.sqrt(sum(a * a for a in actual) * sum(b * b for b in reference))
                (ROOT / "device_embedding.bin").write_bytes(device_embedding)
                run_count = words[6] + 1
                totals = words[20:20 + run_count]
                measured = totals[1:] if run_count > 1 else totals
                report = json.loads((ROOT / "metadata.json").read_text())
                report.update({
                    "status": "completed_board_latency_smoke",
                    "board": "Alif AppKit-E8, M55 HP + Ethos-U85-256",
                    "runtime": "ExecuTorch 1.5.1",
                    "quantization": "INT8 weights and activations, PTQ, one bundled calibration sample",
                    "psram_bus_hz": 25000000,
                    "psram_read_latency": "fixed 8 clocks, MR0 bit 5 set on both x8 dies; controller WAIT=7, RXDS_VL_EN=0",
                    "runtime_metadata": "internal SRAM; planned tensors and bulk scratch in external PSRAM",
                    "npu_commands": "original Vela command bytes copied unchanged into internal SRAM",

                    "core_clock_hz": link.memory_read32(symbols["SystemCoreClock"], 1)[0],
                    "warmup_runs": 1 if run_count > 1 else 0,
                    "measured_runs": len(measured),
                    "timing_kind": "warm" if run_count > 1 else "single cold policy execution after model load",
                    "transfer_verification": verified_files,
                    "firmware_build": "GCC Debug configuration (-Og)",
                    "backbone_ms": words[8:8 + run_count],
                    "denoise_ms": words[14:14 + run_count],
                    "total_ms": totals,
                    "latency_median_ms": statistics.median(measured),
                    "warm_median_ms": statistics.median(measured) if run_count > 1 else None,
                    "actions": actions,
                    "action_sanity": {"elements": len(actions), "all_finite": all(math.isfinite(a) for a in actions)},
                    "embedding_sanity": {
                        "sample_count": 1,
                        "elements": len(actual),
                        "rmse_vs_quantized_graph": math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, quantized_reference)) / len(actual)),
                        "cosine_vs_quantized_graph": sum(a * b for a, b in zip(actual, quantized_reference)) / quantized_norm if quantized_norm else None,
                        "quantized_reference": json.loads((ROOT / "quantized_embedding_reference.json").read_text()),
                        "all_finite": all(math.isfinite(a) for a in actual),
                        "cosine_vs_fp32": sum(a * b for a, b in zip(actual, reference)) / norm if norm else None,
                        "rmse_vs_fp32": math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, reference)) / 384),
                        "scope": "Bundled calibration sample only; no robot-task quality evaluation.",
                    },
                    "scope": "Two supplied image frames to four 7D normalized actions; includes 20 diffusion steps and CPU quantization/sampling; excludes language encoding, camera capture, model upload/load, and robot execution. No policy-quality evaluation.",
                    "artifacts": {name: {"bytes": (ROOT / name).stat().st_size,
                                           "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
                                  for name in (["denoise.pte", "embedding.bin", "schedule.bin"] if args.head_only else
                                               ["backbone.pte", "denoise.pte", "primary.bin", "wrist.bin", "language.bin"])},
                })
                if args.head_only:
                    report["scope"] = "Diffusion head only: precomputed action embedding to four 7D normalized actions, 20 DDPM steps. Excludes vision/transformer; not end-to-end policy latency."
                (ROOT / ("head_board_report.json" if args.head_only else "board_report.json")).write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2), flush=True)
                return
            time.sleep(1)
        raise TimeoutError("board benchmark did not finish before deadline")
    finally:
        link.close()


if __name__ == "__main__":
    main()
