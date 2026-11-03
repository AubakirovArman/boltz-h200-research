"""O2: retain identical FP32 conditioning casts for one diffusion invocation."""

import json
import os
from pathlib import Path

import torch

import fast_checkpoint_init


def install():
    fast_checkpoint_init.install()
    from boltz.model.modules.diffusionv2 import AtomDiffusion

    original = AtomDiffusion.sample
    records = []

    def sample(self, *args, **kwargs):
        conditioning = kwargs.get("diffusion_conditioning")
        if self.training or conditioning is None:
            return original(self, *args, **kwargs)
        cached = dict(conditioning)
        fields = []
        for name in ["q", "c", "atom_enc_bias", "token_trans_bias", "atom_dec_bias"]:
            value = conditioning[name]
            cached[name] = value.float()
            if value.dtype != torch.float32:
                fields.append({"name": name, "shape": list(value.shape),
                               "source_dtype": str(value.dtype), "fp32_bytes": cached[name].numel() * 4})
        kwargs["diffusion_conditioning"] = cached
        result = original(self, *args, **kwargs)
        records.append({"fields_hoisted": fields, "original_mapping_unchanged": True})
        (Path(os.environ["BOLTZ_RUN_DIRECTORY"]) / "conditioning-hoist.json").write_text(json.dumps(records, indent=2))
        return result

    AtomDiffusion.sample = sample


if __name__ == "__main__":
    install()
    import boltz.main as boltz_main
    boltz_main.cli()
