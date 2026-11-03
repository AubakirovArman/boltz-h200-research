"""Run issue #636's settings on the frozen 140 public pairs, on physical GPU 2."""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pynvml


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GPU_UUID = "GPU-6b984734-8ed6-11dd-3b9a-0b6c36ba5e11"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(target, label, no_kernels=False, resume=False):
    run_dir = HERE / "runs" / label / target
    if (run_dir / "result.json").exists():
        old = json.loads((run_dir / "result.json").read_text())
        if old["returncode"] == 0 and old["counts"]["affinity"] == old["expected"]:
            print(json.dumps({"retained_complete": target, "run": str(run_dir)}), flush=True)
            return
    existing = run_dir.exists()
    if existing:
        assert resume, "Partial run exists; use --resume to retain its evidence and outputs"
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    attempt = 1 + len(list(run_dir.glob("config.attempt*.json")))
    if existing and (run_dir / "config.json").exists():
        attempt += 1
    dataset = json.loads((HERE / "dataset.json").read_text())
    items = [item for item in dataset if item["target"] == target]
    inputs = HERE / "inputs" / target
    command = [sys.executable, "-m", "boltz.main", "predict", str(inputs),
               "--model", "boltz2", "--accelerator", "gpu", "--devices", "1",
               "--cache", str(ROOT / "cache"), "--out_dir", str(run_dir / "output"),
               "--recycling_steps", "10", "--sampling_steps", "200", "--diffusion_samples", "5",
               "--sampling_steps_affinity", "200", "--diffusion_samples_affinity", "5",
               "--seed", "0", "--num_workers", "2", "--preprocessing-threads", "1"]
    if no_kernels:
        command.append("--no_kernels")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": GPU_UUID, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "8", "NUMBA_NUM_THREADS": "8", "PYTHONUNBUFFERED": "1",
                "BOLTZ_CACHE": str(ROOT / "cache"), "CUEQ_DEFAULT_CONFIG": "1", "CUEQ_DISABLE_AOT_TUNING": "1"})
    config = {
        "command": command, "target": target, "expected": len(items), "gpu_uuid": GPU_UUID,
        "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "upstream", text=True).strip(),
        "upstream_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT / "upstream", text=True),
        "dataset_sha256": digest(HERE / "dataset.json"),
        "input_sha256": {item["Target ID"]: digest(ROOT / item["input"]) for item in items},
        "msa_sha256": digest(ROOT / "data/msa" / (target + ".a3m")),
        "settings": {"structure_recycles": 10, "structure_samples": 5, "structure_steps": 200,
                     "affinity_recycles_internal": 5, "affinity_samples": 5, "affinity_steps": 200,
                     "seed": 0, "molecular_weight_correction_in_cli": False,
                     "posthoc_variants": ["raw", "official_fixed_MW_correction"],
                     "structure_kernels": not no_kernels, "affinity_kernels": False},
        "scope": "Independent reconstruction; original issue's exact MSA, seed, ligand inputs and full predictions were not published.",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "attempt": attempt,
        "resuming_existing_outputs": existing,
        "resume_rng_note": "A resumed CLI starts from seed 0 again; subsequent sample RNG is not byte-identical to an uninterrupted run. Existing predictions are retained without selecting by score.",
    }
    assert not config["upstream_status"], "Official model source must be unmodified"
    if existing:
        original = json.loads((run_dir / "config.json").read_text())
        assert original["input_sha256"] == config["input_sha256"]
        assert original["msa_sha256"] == config["msa_sha256"]
        assert original["settings"] == config["settings"]
    (run_dir / ("config.json" if not existing else f"config.attempt{attempt}.json")).write_text(json.dumps(config, indent=2))
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID(GPU_UUID)
    started = time.monotonic()
    peak = 0
    last_print = -30
    with (run_dir / ("stdout.log" if not existing else f"stdout.attempt{attempt}.log")).open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT / "upstream", env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                gpu_rows = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                memory = sum(row.usedGpuMemory for row in gpu_rows if row.pid == process.pid)
                peak = max(peak, memory)
                elapsed = time.monotonic() - started
                if elapsed - last_print >= 30:
                    counts = {"structure": len({p.parent for p in (run_dir / "output").rglob("confidence_*.json")}),
                              "affinity": len(list((run_dir / "output").rglob("affinity_*.json")))}
                    progress = {"target": target, "label": label, "pid": process.pid,
                                "elapsed_s": elapsed, "counts": counts, "expected": len(items),
                                "gpu_peak_mib": peak / 2**20, "status": "running"}
                    (run_dir / "progress.json").write_text(json.dumps(progress, indent=2))
                    print(json.dumps(progress), flush=True)
                    last_print = elapsed
                if elapsed > 4 * 3600:
                    process.terminate()
                    raise TimeoutError("Four-hour bound exceeded for one target series")
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    count_affinity = len(list((run_dir / "output").rglob("affinity_*.json")))
    confidence_files = list((run_dir / "output").rglob("confidence_*.json"))
    count_confidence = len({p.parent for p in confidence_files})
    result = {"target": target, "label": label, "returncode": process.returncode,
              "elapsed_s": time.monotonic() - started, "gpu_peak_sampled_mib": peak / 2**20,
              "counts": {"affinity": count_affinity, "structure": count_confidence,
                         "confidence_files": len(confidence_files)}, "expected": len(items)}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    assert result["returncode"] == 0 and count_affinity == len(items), result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="issue-settings-csv-seed0")
    parser.add_argument("--targets", nargs="+", choices=["L1000", "L3000"], default=["L1000", "L3000"])
    parser.add_argument("--no-kernels", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for target in args.targets:
        run(target, args.label, args.no_kernels, args.resume)
