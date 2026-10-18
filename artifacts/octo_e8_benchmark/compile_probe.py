import argparse
import json
from pathlib import Path
import torch
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower

parser=argparse.ArgumentParser()
parser.add_argument('--kind',choices=['conv','add','wide','chain','exact','exact3'],default='conv')
args=parser.parse_args()
root=Path(__file__).resolve().parent
output_dir=root/('probe' if args.kind=='conv' else 'probe_'+args.kind)
output_dir.mkdir(exist_ok=True)
torch.manual_seed(17)
torch.set_num_threads(4)
class Probe(torch.nn.Module):
 def __init__(self):
  super().__init__()
  channels=128 if args.kind=='wide' else 32
  self.layers=torch.nn.Sequential(torch.nn.Conv2d(16,channels,3,padding=1),torch.nn.ReLU())
  if args.kind in ['exact','exact3']: self.layers=torch.nn.Sequential(torch.nn.Conv2d(6,32,3,padding=1,stride=2),torch.nn.ReLU(),torch.nn.Conv2d(32,96,3,padding=1,stride=2),torch.nn.ReLU())
  if args.kind=='exact3': self.layers.extend([torch.nn.Conv2d(96,176,3,padding=1,stride=2),torch.nn.ReLU()])
  if args.kind=='chain': self.layers.extend([torch.nn.Conv2d(channels,channels,3,padding=1),torch.nn.ReLU()])
  self.register_buffer('offset',torch.linspace(-.5,.5,channels).reshape(1,channels,1,1))
 def forward(self,x):
  result=self.layers(x)
  return result+self.offset if args.kind=='add' else result
model=Probe().eval()
shape=(2,6,256,256) if args.kind in ['exact','exact3'] else (1,16,64,64)
inputs=((torch.arange(torch.tensor(shape).prod().item(),dtype=torch.float32).remainder(127)-63).div(64).reshape(shape),)
spec=EthosUCompileSpec('ethos-u85-256',system_config='Ethos_U85_SRAM_OSPI',memory_mode='Dedicated_Sram',config_ini='/Users/npow/alif/CMSIS-Executorch/.cmsis/ensemble_vela.ini')
spec.dump_intermediate_artifacts_to(str(output_dir/'vela'))
module=torch.export.export(model,inputs,strict=False).module()
quantizer=EthosUQuantizer(spec).set_global(get_symmetric_quantization_config(is_per_channel=True))
prepared=prepare_pt2e(module,quantizer)
with torch.no_grad():prepared(*inputs)
quantized=convert_pt2e(prepared)
with torch.no_grad():reference=quantized(*inputs)
program=to_edge_transform_and_lower(torch.export.export(quantized,inputs,strict=False),partitioner=[EthosUPartitioner(spec)],compile_config=EdgeCompileConfig(_check_ir_validity=False)).to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
(output_dir/'conv.pte').write_bytes(program.buffer)
(output_dir/'reference.bin').write_bytes(reference.numpy().tobytes())
(output_dir/'input.bin').write_bytes(inputs[0].numpy().tobytes())
report={'pte_bytes':len(program.buffer),'planned_buffers':[p.non_const_buffer_sizes for p in program.executorch_program.execution_plan],'input_shape':list(inputs[0].shape),'output_shape':list(reference.shape),'source_seed':17,'scope':'Deterministic synthetic convolution for hardware/runtime diagnosis, not policy evaluation'}
(output_dir/'compile.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
