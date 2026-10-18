import ctypes
import hashlib
import json
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, "/private/tmp/dcops-pylink")
import pylink

ROOT = Path(__file__).resolve().parent
FILES = [
    ("backbone.pte", 0xA0000000), ("denoise.pte", 0xA1C00000),
    ("primary.bin", 0xA1E00000), ("wrist.bin", 0xA1E60000),
    ("language.bin", 0xA1E80000), ("schedule.bin", 0xA1E8C000),
]


def main():
    library = pylink.library.Library(dllpath="/Applications/SEGGER/JLink_V986a/libjlinkarm.9.86.1.dylib")
    link = pylink.JLink(lib=library)
    link.open(serial_no=1219599493)
    link.set_tif(pylink.enums.JLinkInterfaces.SWD)
    link.connect("AE822FA0E5597LS0_M55_HP", speed=15000)
    link.exec_command("SetEnableFlashbank 0xA0000000=0")
    link.exec_command("SetAllowFlashCache=0")
    read = link._dll.JLINKARM_ReadMemEx
    read.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
    read.restype = ctypes.c_int
    report = {"files": [], "scope": "Full byte-for-byte readback after all timed runs; verification traffic excluded from timings."}
    try:
        elf = ROOT / "firmware/out/cmsis-executorch/DevKit-E8/Debug/cmsis-executorch.elf"
        nm = "/Users/npow/.vcpkg/artifacts/2139c4c6/compilers.arm.arm.none.eabi.gcc/15.3.1/bin/arm-none-eabi-nm"
        symbols = {parts[2]: int(parts[0], 16)
                   for line in subprocess.check_output([nm, "-n", str(elf)], text=True).splitlines()
                   if len(parts := line.split()) == 3 and parts[0] != "U"}
        state = link.memory_read32(symbols["octo_benchmark"], 2)
        if state[0] != 0x4F43544F or state[1] not in [5, 92, 93, 94, 95]:
            raise RuntimeError("benchmark is not finished; refusing verification during timed inference")
        for name, address in FILES:
            source = (ROOT / name).read_bytes()
            digest = hashlib.sha256()
            mismatches = []
            mismatch_bytes = 0
            for offset in range(0, len(source), 64 * 1024):
                expected = source[offset:offset + 64 * 1024]
                buffer = ctypes.create_string_buffer(len(expected))
                count = read(address + offset, len(expected), buffer, 4)
                if count != len(expected):
                    raise RuntimeError(f"short read at {address + offset:#x}: {count}/{len(expected)}")
                actual = buffer.raw
                digest.update(actual)
                if actual != expected:
                    differing = sum(a != b for a, b in zip(actual, expected))
                    mismatch_bytes += differing
                    mismatches.append({"offset": offset, "bytes": len(expected), "differing_bytes": differing})
                if offset % (1024 * 1024) == 0:
                    print("verify", name, offset, "/", len(source), flush=True)
            result = {"file": name, "bytes_checked": len(source),
                      "expected_sha256": hashlib.sha256(source).hexdigest(),
                      "device_sha256": digest.hexdigest(),
                      "mismatch_bytes": mismatch_bytes, "mismatched_blocks": mismatches}
            report["files"].append(result)
            print(json.dumps(result), flush=True)
        report["passed"] = all(item["mismatch_bytes"] == 0 for item in report["files"])
        report["total_bytes_checked"] = sum(item["bytes_checked"] for item in report["files"])
        (ROOT / "board_bytes_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        if not report["passed"]:
            raise RuntimeError("device model or fixture differs from source; timing result needs a corrected rerun")
    finally:
        link.close()


if __name__ == "__main__":
    main()
