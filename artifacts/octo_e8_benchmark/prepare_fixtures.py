"""Write board inputs and a quantized reference from the exported sample."""
import hashlib
import json
from pathlib import Path

import torch
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e  # Registers quantized operators.

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(8)
    samples = torch.load(ROOT / 'inputs.pt', weights_only=True)
    tensors = dict(zip(('primary', 'wrist', 'language'), samples['backbone']))
    tensors['embedding'] = samples['embedding']
    tensors['schedule'] = torch.cat([samples[key].flatten() for key in ('betas', 'alphas', 'alpha_hats')])
    model = torch.export.load(ROOT / 'backbone_quantized.pt2').module()
    with torch.no_grad():
        output = model(*samples['backbone'])
    if isinstance(output, (tuple, list)):
        output = output[0]
    tensors['quantized_embedding'] = output
    for name, tensor in tensors.items():
        (ROOT / f'{name}.bin').write_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())
    raw = (ROOT / 'quantized_embedding.bin').read_bytes()
    reference = {
        'source_model_sha256': hashlib.sha256((ROOT / 'backbone_quantized.pt2').read_bytes()).hexdigest(),
        'sample_count': 1, 'elements': output.numel(),
        'sha256': hashlib.sha256(raw).hexdigest(),
        'all_finite': torch.isfinite(output).all().item(),
        'scope': 'PyTorch quantized graph output on the frozen bundled sample; no robot evaluation.',
    }
    (ROOT / 'quantized_embedding_reference.json').write_text(json.dumps(reference, indent=2) + '\n')


if __name__ == '__main__':
    main()
