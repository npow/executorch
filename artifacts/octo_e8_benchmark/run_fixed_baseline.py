"""Acquire the shared board, cold-initialize PSRAM, and time one Octo chunk."""
import fcntl
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
lock_path = Path('/tmp/alif-e8-1219599493.lock')
with lock_path.open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print(f'ACQUIRED E8 lock pid={os.getpid()}; Octo owns original-app restoration', flush=True)
    jlink = ['/Applications/SEGGER/JLink_V986a/JLinkExe', '-device',
             'AE822FA0E5597LS0_M55_HP', '-if', 'SWD', '-speed', '1000',
             '-autoconnect', '1', '-CommandFile']
    # Pin reset can drop the debug connection; reconnect in a fresh process.
    with (root / 'fixed_baseline_pin_reset.log').open('w') as output:
        subprocess.run(jlink + [str(root / 'flash_pin_reset.jlink')],
                       stdout=output, stderr=subprocess.STDOUT, timeout=90)
    with (root / 'fixed_baseline_flash.log').open('w') as output:
        subprocess.run(jlink + [str(root / 'flash.jlink')],
                       stdout=output, stderr=subprocess.STDOUT, timeout=90, check=True)
    subprocess.run(['/Users/npow/alif/CMSIS-Executorch/.venv/bin/python',
                    str(root / 'run_board.py'), '--upload', '--timeout', '180'], check=True)
