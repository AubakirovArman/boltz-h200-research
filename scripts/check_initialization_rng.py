"""Targeted correctness proof for the strict checkpoint-load initialization shortcut."""

import json
from pathlib import Path

import numpy as np
import torch

from boltz.model.layers import initialize
from fast_checkpoint_init import checkpoint_initialization, rng_fingerprints


cases = []
for seed in [0, 17, 104729]:
    for shape, scale, fan in [((1, 1), 1, "fan_in"), ((31, 17), 2, "fan_out"),
                              ((64, 128), 1, "fan_avg"), ((512, 128), 1, "fan_in")]:
        weight = torch.empty(shape)
        np.random.seed(seed)
        torch.manual_seed(seed)
        initialize.trunc_normal_init_(weight, scale=scale, fan=fan)
        expected = rng_fingerprints()
        np.random.seed(seed)
        torch.manual_seed(seed)
        stats = {"truncated_initializers_skipped": 0, "uniform_draws_preserved": 0}
        with checkpoint_initialization(stats):
            initialize.trunc_normal_init_(weight, scale=scale, fan=fan)
        actual = rng_fingerprints()
        assert expected == actual, (seed, shape, scale, fan)
        cases.append({"seed": seed, "shape": shape, "scale": scale, "fan": fan, "rng_equal": True})


class Example(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(64, 128)
        self.down = torch.nn.Linear(128, 64)
        initialize.lecun_normal_init_(self.up.weight)
        initialize.he_normal_init_(self.down.weight)


torch.manual_seed(91)
np.random.seed(91)
checkpoint = Example().state_dict()
torch.manual_seed(0)
np.random.seed(0)
normal = Example()
normal.load_state_dict(checkpoint, strict=True)
expected_rng = rng_fingerprints()
torch.manual_seed(0)
np.random.seed(0)
stats = {"truncated_initializers_skipped": 0, "uniform_draws_preserved": 0}
with checkpoint_initialization(stats):
    fast = Example()
fast.load_state_dict(checkpoint, strict=True)
assert rng_fingerprints() == expected_rng
assert all(torch.equal(value, fast.state_dict()[key]) for key, value in normal.state_dict().items())
inputs = torch.randn(8, 64)
assert torch.equal(normal.down(normal.up(inputs)), fast.down(fast.up(inputs)))
report = {"rng_cases": cases, "strict_loaded_weights_equal": True,
          "subsequent_rng_equal": True, "forward_equal": True}
root = Path(__file__).resolve().parents[1]
(root / "reports/initialization-rng-proof.json").write_text(json.dumps(report, indent=2))
print(json.dumps({key: value for key, value in report.items() if key != "rng_cases"} | {"rng_cases_passed": len(cases)}, indent=2))
