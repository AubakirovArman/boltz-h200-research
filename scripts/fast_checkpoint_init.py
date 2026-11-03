"""O1: avoid constructing discarded truncated-normal values during strict loading.

The pinned Boltz initializer draws scipy.stats.truncnorm's uniform samples and
then computes an expensive inverse CDF. A complete checkpoint overwrites those
parameters. Keep the exact RNG draws and ordinary Torch initialization, omit
only the discarded inverse CDF/copy, and restore the initializer after loading.
"""

import contextlib
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy
import torch

from boltz.model.layers import initialize


ORIGINAL = initialize.trunc_normal_init_


def rng_fingerprints():
    state = np.random.get_state()
    numpy_bytes = state[1].tobytes() + repr((state[0], state[2:])).encode()
    return {"numpy": hashlib.sha256(numpy_bytes).hexdigest(),
            "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}


@contextlib.contextmanager
def checkpoint_initialization(stats):
    assert scipy.__version__ == "1.13.1", "RNG equivalence is validated for the pinned SciPy"

    def discard_truncated_values(weights, scale=1.0, fan="fan_in"):
        if weights.device.type != "cpu" or scale <= 0 or fan not in {"fan_in", "fan_out", "fan_avg"}:
            return ORIGINAL(weights, scale=scale, fan=fan)
        if weights.ndim != 2:
            return ORIGINAL(weights, scale=scale, fan=fan)
        # This is precisely scipy.stats.rv_continuous._rvs' RNG call for
        # truncnorm in SciPy 1.13.1. No cached RNG states or precomputed weights.
        initialize.truncnorm.random_state.uniform(size=weights.numel())
        stats["truncated_initializers_skipped"] += 1
        stats["uniform_draws_preserved"] += weights.numel()
        stats.setdefault("_skipped_parameter_ids", []).append(id(weights))

    previous = initialize.trunc_normal_init_
    initialize.trunc_normal_init_ = discard_truncated_values
    try:
        yield
    finally:
        initialize.trunc_normal_init_ = previous


def install():
    import boltz.main as boltz_main

    original_load = boltz_main.Boltz2.load_from_checkpoint
    rows = []

    def load(*args, **kwargs):
        # Only the CLI's explicit strict=True loads may omit discarded values.
        if kwargs.get("strict") is not True:
            return original_load(*args, **kwargs)
        stats = {"checkpoint": str(args[0]), "truncated_initializers_skipped": 0,
                 "uniform_draws_preserved": 0, "rng_before_load": rng_fingerprints()}
        started = time.perf_counter()
        with checkpoint_initialization(stats):
            model = original_load(*args, **kwargs)
        parameter_ids = {id(parameter) for parameter in model.parameters()}
        skipped_ids = stats.pop("_skipped_parameter_ids", [])
        assert set(skipped_ids) <= parameter_ids, "A skipped initializer did not belong to a loaded parameter"
        stats["strict_checkpoint_covers_all_skipped_parameters"] = True
        stats.update(elapsed_s=time.perf_counter() - started, rng_after_load=rng_fingerprints())
        rows.append(stats)
        destination = Path(os.environ["BOLTZ_RUN_DIRECTORY"])
        (destination / "initialization.json").write_text(json.dumps(rows, indent=2))
        return model

    boltz_main.Boltz2.load_from_checkpoint = staticmethod(load)


if __name__ == "__main__":
    install()
    import boltz.main as boltz_main
    boltz_main.cli()
