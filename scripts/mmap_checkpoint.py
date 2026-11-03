"""O3: mmap the same immutable local checkpoint instead of buffering its storage."""

import json
import os
import time
from pathlib import Path

import torch

import fast_checkpoint_init


def install(include_initialization=True):
    if include_initialization:
        fast_checkpoint_init.install()
    import pytorch_lightning.core.saving as saving

    original = saving.pl_load
    root = Path(__file__).resolve().parents[1]
    permitted = {(root / "cache" / name).resolve() for name in ["boltz2_conf.ckpt", "boltz2_aff.ckpt"]}
    rows = []

    def load(path_or_url, map_location=None, weights_only=False):
        if not isinstance(path_or_url, (str, Path)) or str(map_location) != "cpu":
            return original(path_or_url, map_location=map_location, weights_only=weights_only)
        if Path(path_or_url).resolve() not in permitted:
            return original(path_or_url, map_location=map_location, weights_only=weights_only)
        started = time.perf_counter()
        result = torch.load(path_or_url, map_location=map_location, weights_only=weights_only, mmap=True)
        rows.append({"checkpoint": str(path_or_url), "mmap": True, "metadata_load_s": time.perf_counter() - started})
        (Path(os.environ["BOLTZ_RUN_DIRECTORY"]) / "checkpoint-mmap.json").write_text(json.dumps(rows, indent=2))
        return result

    saving.pl_load = load


if __name__ == "__main__":
    install()
    import boltz.main as boltz_main
    boltz_main.cli()
