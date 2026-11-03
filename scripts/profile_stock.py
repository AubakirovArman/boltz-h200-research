"""Diagnostic only: time loading and model stages without changing model math."""

import cProfile
import hashlib
import json
import os
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import torch

import boltz.main as boltz_main


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "reports/profile-stock"
DEST.mkdir(exist_ok=True)
loads = []
phases = []
original_load = boltz_main.Boltz2.load_from_checkpoint
original_predict = boltz_main.Trainer.predict


def rng_fingerprints():
    state = np.random.get_state()
    numpy_bytes = state[1].tobytes() + repr((state[0], state[2:])).encode()
    return {"numpy": hashlib.sha256(numpy_bytes).hexdigest(),
            "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}


def measured_load(*args, **kwargs):
    profile = cProfile.Profile()
    started = time.perf_counter()
    profile.enable()
    model = original_load(*args, **kwargs)
    profile.disable()
    row = {"checkpoint": str(args[0]), "elapsed_s": time.perf_counter() - started,
           "rng_after_load": rng_fingerprints(), "functions": []}
    for (filename, line, name), (primitive, total, own, cumulative, callers) in pstats.Stats(profile).stats.items():
        row["functions"].append({"file": filename, "line": line, "name": name, "calls": total,
                                 "own_s": own, "cumulative_s": cumulative})
    row["functions"].sort(key=lambda item: item["cumulative_s"], reverse=True)
    row["functions"] = row["functions"][:70]
    loads.append(row)
    profile.dump_stats(str(DEST / (Path(args[0]).stem + ".pstats")))
    (DEST / "loads.json").write_text(json.dumps(loads, indent=2))
    return model


def measured_predict(self, model, *args, **kwargs):
    phase = "affinity" if model.affinity_prediction else "structure"
    events = []
    pending = {}
    counts = {}
    handles = []
    profiler = None
    gpu_profile_rows = []

    def pre(name):
        def hook(module, arguments):
            nonlocal profiler
            counts[name] = counts.get(name, 0) + 1
            if phase == "structure" and name == "score_model" and counts[name] == 5:
                profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA],
                                                  record_shapes=True, profile_memory=False)
                profiler.__enter__()
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            pending[name] = (event, time.perf_counter())
        return hook

    def post(name):
        def hook(module, arguments, output):
            nonlocal profiler
            start, cpu_start = pending.pop(name)
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            events.append((name, start, end, time.perf_counter() - cpu_start))
            if phase == "structure" and name == "score_model" and counts[name] == 10:
                profiler.__exit__(None, None, None)
                for event in profiler.key_averages():
                    gpu_profile_rows.append({"name": event.key, "calls": event.count,
                                             "self_cuda_us": event.self_device_time_total,
                                             "self_cpu_us": event.self_cpu_time_total})
                gpu_profile_rows.sort(key=lambda item: item["self_cuda_us"], reverse=True)
                (DEST / "score-operator-profile.json").write_text(json.dumps(gpu_profile_rows, indent=2))
                profiler.export_chrome_trace(str(DEST / "score-trace.json"))
                profiler = None
        return hook

    modules = {"input_embedder": model.input_embedder, "msa_module": model.msa_module,
               "pairformer_module": model.pairformer_module,
               "diffusion_conditioning": model.diffusion_conditioning,
               "score_model": model.structure_module.score_model}
    for name in ["confidence_module", "affinity_module", "affinity_module1", "affinity_module2"]:
        if hasattr(model, name):
            modules[name] = getattr(model, name)
    for name, module in modules.items():
        handles.extend([module.register_forward_pre_hook(pre(name)), module.register_forward_hook(post(name))])
    started = time.perf_counter()
    result = original_predict(self, model, *args, **kwargs)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    for handle in handles:
        handle.remove()
    aggregates = {}
    for name, start, end, cpu_wall in events:
        row = aggregates.setdefault(name, {"calls": 0, "cuda_elapsed_s": 0.0, "cpu_wall_s": 0.0})
        row["calls"] += 1
        row["cuda_elapsed_s"] += start.elapsed_time(end) / 1000
        row["cpu_wall_s"] += cpu_wall
    phases.append({"phase": phase, "trainer_predict_wall_s": wall, "use_kernels": model.use_kernels,
                   "modules": aggregates})
    (DEST / "phases.json").write_text(json.dumps(phases, indent=2))
    return result


boltz_main.Boltz2.load_from_checkpoint = staticmethod(measured_load)
boltz_main.Trainer.predict = measured_predict
boltz_main.cli()
