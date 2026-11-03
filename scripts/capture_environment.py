"""Record the fixed environment, effective Click defaults and checkpoint configuration."""

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
import torch


ROOT = Path(__file__).resolve().parents[1]
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "GPU-6b984734-8ed6-11dd-3b9a-0b6c36ba5e11"
assert torch.cuda.device_count() == 1
from boltz import main as boltz_main


packages = {}
for name in ["boltz", "torch", "triton", "pytorch-lightning", "numpy", "rdkit", "scipy",
             "cuequivariance", "cuequivariance-torch", "cuequivariance-ops-torch-cu12", "cuequivariance-ops-cu12"]:
    packages[name] = importlib.metadata.version(name)
properties = torch.cuda.get_device_properties(0)
allowed = ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
           "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUEQ_DEFAULT_CONFIG",
           "CUEQ_DISABLE_AOT_TUNING", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"]
environment = {
    "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
    "cpu_logical_count": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
    "host_ram_total_bytes": psutil.virtual_memory().total,
    "host_ram_available_bytes": psutil.virtual_memory().available,
    "disk_free_bytes": shutil.disk_usage(ROOT).free,
    "packages": packages, "torch_cuda_runtime": torch.version.cuda,
    "torch_threads": torch.get_num_threads(), "visible_device_count": torch.cuda.device_count(),
    "gpu_name": properties.name, "gpu_memory_total_bytes": properties.total_memory,
    "compute_capability": [properties.major, properties.minor],
    "native_bf16": torch.cuda.is_bf16_supported(including_emulation=False),
    "environment_allowlist": {key: os.environ.get(key) for key in allowed},
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "upstream", text=True).strip(),
    "source_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT / "upstream", text=True),
    "gpu_telemetry": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version,mig.mode.current", "--format=csv"], text=True),
    "gpu_processes": subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"], text=True),
}
(ROOT / "manifests/environment.json").write_text(json.dumps(environment, indent=2))
(ROOT / "manifests/pip-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
(ROOT / "manifests/cli-help.txt").write_text(subprocess.check_output([str(ROOT / ".venv/bin/boltz"), "predict", "--help"], text=True))
for kernels in [False, True]:
    name = "baseline_b1" if kernels else "baseline_b0"
    args = [str(ROOT / "data/inputs/L3001.yaml"), "--seed", "0", "--cache", str(ROOT / "cache"),
            "--recycling_steps", "3", "--sampling_steps", "200", "--diffusion_samples", "1",
            "--sampling_steps_affinity", "200", "--diffusion_samples_affinity", "5",
            "--num_workers", "2", "--preprocessing-threads", "4", "--devices", "1", "--model", "boltz2"]
    if not kernels:
        args.append("--no_kernels")
    context = boltz_main.predict.make_context("predict", args)
    (ROOT / "configs" / (name + ".json")).write_text(json.dumps(context.params, indent=2, default=str))
checkpoints = {}
for name in ["boltz2_conf.ckpt", "boltz2_aff.ckpt"]:
    checkpoint = torch.load(ROOT / "cache" / name, map_location="cpu", weights_only=False, mmap=True)
    params = checkpoint.get("hyper_parameters", {})
    fields = ["compile_pairformer", "compile_structure", "compile_confidence", "compile_affinity",
              "compile_msa", "compile_templates", "use_kernels", "affinity_prediction", "confidence_prediction",
              "run_trunk_and_structure", "skip_run_structure", "affinity_ensemble", "affinity_mw_correction",
              "score_model_args", "affinity_model_args", "affinity_model_args1", "affinity_model_args2"]
    checkpoints[name] = {key: params.get(key, "not_saved_uses_constructor_default") for key in fields}
    checkpoints[name]["state_dict_keys"] = len(checkpoint["state_dict"])
    del checkpoint
(ROOT / "manifests/checkpoint-settings.json").write_text(json.dumps(checkpoints, indent=2, default=str))
print(json.dumps({"environment_ready": True, "gpu_name": properties.name,
                  "packages": packages, "checkpoint_settings": {key: {k: v for k, v in item.items() if not k.endswith("args") and not k.endswith("args1") and not k.endswith("args2")} for key, item in checkpoints.items()}}, indent=2))
