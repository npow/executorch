import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "octo_ternary"
sys.path.insert(0, str(SOURCE / "octo-pytorch"))

from octo.model.octo_model_pt import OctoModelPt


class Backbone(torch.nn.Module):
    def __init__(self, transformer, observations, tasks):
        super().__init__()
        self.transformer = transformer
        self.observations = observations
        self.tasks = tasks

    def forward(self, primary, wrist, language):
        observations = {**self.observations, "image_primary": primary, "image_wrist": wrist}
        tasks = {**self.tasks, "language_instruction": language}
        outputs = self.transformer(
            observations, tasks, observations["timestep_pad_mask"],
            train=False, save_attention_mask=True,
        )
        return outputs["readout_action"].tokens[:, -1, 0]


class Denoise(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, embedding, actions, timestep):
        return self.model(embedding, actions, timestep)


def main():
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    checkpoint = SOURCE / "octo-pytorch-checkpoint"
    config = json.loads((checkpoint / "config.json").read_text())
    with (checkpoint / "example_batch.pickle").open("rb") as stream:
        batch = pickle.load(stream)
    model = OctoModelPt.from_config(config, batch, dataset_statistics=None).eval()
    weights = checkpoint / "300000/weights.pth"
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["state_dict"], strict=True)
    obs, tasks = batch["observation"], batch["task"]
    with torch.no_grad():
        language = model.module.octo_transformer.task_tokenizers.language.hf_model(
            **tasks["language_instruction"]
        ).last_hidden_state
    model.module.octo_transformer.task_tokenizers.language.hf_model = None
    backbone = Backbone(model.module.octo_transformer, obs, tasks).eval()
    inputs = (obs["image_primary"], obs["image_wrist"], language)
    with torch.no_grad():
        started = time.perf_counter()
        embedding = backbone(*inputs)
        print("backbone reference seconds", time.perf_counter() - started, flush=True)
    head = model.module.heads["action"]
    denoise = Denoise(head.diffusion_model).eval()
    denoise_inputs = (embedding, torch.randn(1, 28), torch.tensor([[19.0]]))
    metadata = {
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "policy_parameters": sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in denoise.parameters()),
        "observation_frames": 2,
        "action_shape": [1, 4, 7],
        "diffusion_steps": head.diffusion_steps,
        "language_encoder": "precomputed on host; excluded from timed policy",
        "input_source": "checkpoint bundled example batch; single sample, no task-quality claim",
    }
    torch.save({"backbone": inputs, "denoise": denoise_inputs, "embedding": embedding,
                "betas": head.betas, "alphas": head.alphas, "alpha_hats": head.alpha_hats}, ROOT / "inputs.pt")
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for name, module, args in [("backbone", backbone, inputs), ("denoise", denoise, denoise_inputs)]:
        print("export", name, flush=True)
        exported = torch.export.export(module, args, strict=False)
        torch.export.save(exported, ROOT / f"{name}.pt2")
        print("exported", name, len(list(exported.graph.nodes)), flush=True)


if __name__ == "__main__":
    main()
