import argparse
import ctypes
import hashlib
import json
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
sys.path.insert(0, '/private/tmp/dcops-pylink')
import pylink

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--kind',default='conv',choices=['conv','add','wide','chain','exact','exact3'])
parser.add_argument('--reuse-probe',action='store_true')
parser.add_argument('--keep-running',action='store_true')
parser.add_argument('--dma-chunk',type=int,default=0)
parser.add_argument('--modes',default='3,4,5,6,7,8,9,10')
args=parser.parse_args()
OUTPUT=ROOT/('probe' if args.kind=='conv' else 'probe_'+args.kind)
ELF=ROOT/'firmware/out/cmsis-executorch/DevKit-E8/Debug/cmsis-executorch.elf'
NM='/Users/npow/.vcpkg/artifacts/2139c4c6/compilers.arm.arm.none.eabi.gcc/15.3.1/bin/arm-none-eabi-nm'
symbols={p[2]:int(p[0],16) for line in subprocess.check_output([NM,'-n',str(ELF)],text=True).splitlines() if len(p:=line.split())==3 and p[0]!='U'}
mailbox=symbols['octo_benchmark']
link=pylink.JLink(lib=pylink.library.Library(dllpath='/Applications/SEGGER/JLink_V986a/libjlinkarm.9.86.1.dylib'))
results=[]
try:
 link.open(serial_no=1219599493)
 link.set_tif(pylink.enums.JLinkInterfaces.SWD)
 link.connect('AE822FA0E5597LS0_M55_HP',speed=1000)
 link.exec_command('SetEnableFlashbank 0xA0000000=0')
 link.exec_command('SetAllowFlashCache=0')
 words=link.memory_read32(mailbox,8)
 if not args.reuse_probe and words[:3]!=[0x4f43544f,2,0]:raise RuntimeError(f'Not idle: {words}')
 raw=(OUTPUT/'conv.pte').read_bytes()
 actual_crc=zlib.crc32(raw)
 if not args.reuse_probe:
  padded=raw+bytes((-len(raw))%4)
  link.set_speed(4000)
  write=link._dll.JLINKARM_WriteMemEx
  write.argtypes=[ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32]
  write.restype=ctypes.c_int
  buffer=ctypes.create_string_buffer(padded)
  if write(symbols['temp_pool'],len(padded),buffer,4)!=len(padded):raise RuntimeError('Short SRAM write')
  sequence=link.memory_read32(mailbox+232,1)[0]+1
  link.memory_write32(mailbox+216,[0xa3e00000,len(raw)])
  link.memory_write32(mailbox+236,[1])
  link.memory_write32(mailbox+228,[sequence])
  deadline=time.monotonic()+10
  while link.memory_read32(mailbox+232,1)[0]!=sequence:
   if time.monotonic()>deadline:raise TimeoutError('Probe upload timed out')
   time.sleep(.01)
  actual_crc=link.memory_read32(mailbox+224,1)[0]
  if actual_crc!=zlib.crc32(raw):raise RuntimeError(f'PSRAM model CRC mismatch {actual_crc:#x}')
  link.memory_write32(mailbox+12,[len(raw)])
  link.set_speed(1000)
 link.memory_write32(symbols['octo_dma_chunk_bytes'],[args.dma_chunk])
 reference=np.frombuffer((OUTPUT/'reference.bin').read_bytes(),dtype='<f4').reshape(json.loads((OUTPUT/'compile.json').read_text())['output_shape'])
 for mode in [int(v) for v in args.modes.split(',')]:
  link.memory_write32(mailbox+8,[mode])
  deadline=time.monotonic()+12
  while True:
   words=link.memory_read32(mailbox,26)
   if words[7]==mode and (words[1]==5 or words[1]>=90):break
   if time.monotonic()>deadline:raise TimeoutError(f'Mode{mode} monitor timed out: {words[:8]}')
   time.sleep(.05)
  placement=3 if mode>=11 else (mode-3)%4
  item={'mode':mode,'weights':'psram' if placement&1 else 'sram','scratch':'psram' if placement&2 else 'sram','memory_type':'normal' if mode>=7 else 'device','state':words[1],'error':words[5],'elapsed_ms':words[20]}
  snapshot=link.memory_read32(symbols['octo_npu_snapshot'],163)
  item['npu_elapsed_ms']=snapshot[1]-snapshot[0]
  item['npu_result']=snapshot[2]
  item['npu_status']=hex(snapshot[4])
  item['qread']=snapshot[9]
  item['qsize']=snapshot[11]
  item['mem_attr']=[hex(v) for v in snapshot[3+16:3+20]]
  item['npu_registers']=snapshot[3:]
  if words[1]==5:
   pointer=link.memory_read32(symbols['octo_probe_output'],1)[0]
   strides=link.memory_read32(symbols['octo_probe_strides'],4)
   link.set_speed(15000)
   read_started=time.monotonic()
   output_bytes=struct.pack('<'+str(reference.size)+'I',*link.memory_read32(pointer,reference.size))
   item["host_output_read_seconds"]=time.monotonic()-read_started
   link.set_speed(1000)
   actual=np.ndarray(reference.shape,dtype='<f4',buffer=output_bytes,strides=tuple(s*4 for s in strides))
   difference=actual-reference
   item.update(output_elements=actual.size,all_finite=bool(np.isfinite(actual).all()),max_abs_error=float(np.abs(difference).max()),rmse=float(np.sqrt(np.mean(difference**2))),cosine=float(np.dot(actual.ravel(),reference.ravel())/(np.linalg.norm(actual)*np.linalg.norm(reference))),output_sha256=hashlib.sha256(output_bytes).hexdigest(),output_strides=strides)
   (OUTPUT/f'mode{mode}_output.bin').write_bytes(output_bytes)
  else:
   item['runtime_message']=bytes(link.memory_read8(symbols['octo_last_error'],2048)).split(b'\0',1)[0].decode(errors='replace')
  item['dma_chunk_bytes']=args.dma_chunk
  results.append(item)
  report={'model_sha256':hashlib.sha256(raw).hexdigest(),'model_crc32':actual_crc,'results':results,'reference':'Host quantized PyTorch graph, deterministic synthetic convolution'}
  (OUTPUT/'board_results.json').write_text(json.dumps(report,indent=2)+'\n')
  (OUTPUT/f'board_results_dma{args.dma_chunk}.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({k:v for k,v in item.items() if k!='npu_registers'}),flush=True)
 if not args.keep_running:link.memory_write32(mailbox+8,[0])
finally:
 link.close()
