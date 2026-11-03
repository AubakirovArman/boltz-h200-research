"""Run an unmodified Boltz CLI with external, per-process resource monitoring."""

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pynvml
import yaml


ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-6b984734-8ed6-11dd-3b9a-0b6c36ba5e11"


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--input", default="L3001")
    parser.add_argument("--no-kernels", action="store_true")
    parser.add_argument("--launcher", help="Separate optimization/diagnostic launcher; stock is default")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    run = ROOT / "runs" / args.label
    run.mkdir(exist_ok=False)
    input_path = ROOT / "data/inputs" / (args.input + ".yaml")
    if not input_path.is_file():
        input_path = ROOT / "data/inputs" / args.input
    input_files = [input_path] if input_path.is_file() else list(input_path.glob("*.yaml"))
    assert input_files, "No benchmark inputs found"
    payloads = {path.stem: yaml.safe_load(path.read_text()) for path in input_files}
    affinity_inputs = {name for name, payload in payloads.items()
                       if any("affinity" in item for item in payload.get("properties", []))}
    input_hashes = {path.name: sha256(path) for path in input_files}
    input_digest = (sha256(input_path) if input_path.is_file()
                    else hashlib.sha256(json.dumps(input_hashes, sort_keys=True).encode()).hexdigest())
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": GPU_UUID, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8", "NUMBA_NUM_THREADS": "8", "PYTHONUNBUFFERED": "1",
        "BOLTZ_CACHE": str(ROOT / "cache"),
        "BOLTZ_RUN_DIRECTORY": str(run),
        "CUEQ_DEFAULT_CONFIG": "1", "CUEQ_DISABLE_AOT_TUNING": "1",
    })
    # No allocator, dtype, TF32, model, MSA or scientific work changes.
    command = [sys.executable]
    command += [args.launcher] if args.launcher else ["-m", "boltz.main"]
    command += [
        "predict", str(input_path), "--model", "boltz2", "--accelerator", "gpu", "--devices", "1",
        "--recycling_steps", "3", "--sampling_steps", "200", "--diffusion_samples", "1",
        "--sampling_steps_affinity", "200", "--diffusion_samples_affinity", "5", "--seed", "0",
        "--num_workers", "2", "--preprocessing-threads", "1" if input_path.is_dir() else "4",
        "--cache", str(ROOT / "cache"),
        "--out_dir", str(run / "output"),
    ]
    if args.no_kernels:
        command.append("--no_kernels")
    whitelist = ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMBA_NUM_THREADS", "CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING",
                 "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "NVIDIA_TF32_OVERRIDE",
                 "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"]
    config = {
        "label": args.label, "command": command, "gpu_uuid": GPU_UUID,
        "runner_sha256": sha256(Path(__file__)),
        "launcher_sha256": sha256(Path(args.launcher)) if args.launcher else None,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "upstream", text=True).strip(),
        "source_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT / "upstream", text=True),
        "input_sha256": input_digest, "input_files": input_hashes,
        "input_directory_order": [path.name for path in input_files],
        "msa_sha256": {entry["protein"]["msa"]: sha256(Path(entry["protein"]["msa"]))
                       for payload in payloads.values()
                       for entry in payload["sequences"] if "protein" in entry},
        "environment": {key: env.get(key) for key in whitelist},
        "monitoring_interval_s": 0.1,
        "cache_state": "Checkpoints, CCD and MSA predownloaded; fresh output/preprocessing directory; ordinary persistent kernel caches.",
        "task": {"structure_recycles": 3, "structure_steps": 200, "structure_samples": 1,
                 "affinity_recycles_internal": 5, "affinity_steps": 200, "affinity_samples": 5,
                 "affinity_mw_correction": False, "structure_kernels": not args.no_kernels,
                 "affinity_kernels": False, "cli_msa_subsampling": False, "max_msa_seqs": 8192},
    }
    (run / "config.json").write_text(json.dumps(config, indent=2))
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID(GPU_UUID)
    initial_processes = [{"pid": item.pid, "used_gpu_memory_bytes": item.usedGpuMemory}
                         for item in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)]
    started = time.perf_counter()
    utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    process = subprocess.Popen(command, cwd=ROOT / "upstream", env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stages = []

    def capture():
        tail = b""
        markers = {
            b"Running structure prediction for": "structure_announced",
            b"Predicting property: affinity": "affinity_announced",
            b"Running affinity prediction for": "affinity_inputs_announced",
        }
        seen = set()
        with (run / "stdout.log").open("wb") as out:
            while chunk := os.read(process.stdout.fileno(), 65536):
                out.write(chunk)
                out.flush()
                tail = (tail + chunk)[-8192:]
                for marker, name in markers.items():
                    if name not in seen and marker in tail:
                        seen.add(name)
                        stages.append({"name": name, "elapsed_s": time.perf_counter() - started})

    reader = threading.Thread(target=capture, daemon=True)
    reader.start()
    tree = psutil.Process(process.pid)
    own_peak = 0
    device_peak = 0
    host_peak = 0
    other_peak = 0
    observed_own_cuda = False
    known_own_pids = {process.pid}
    timed_out = False
    with (run / "monitor.jsonl").open("w") as monitor:
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            try:
                members = [tree, *tree.children(recursive=True)]
                known_own_pids.update(member.pid for member in members)
                rss = sum(member.memory_info().rss for member in members if member.is_running())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                rss = 0
            gpu_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            own_memory = sum(item.usedGpuMemory for item in gpu_processes if item.pid in known_own_pids)
            other_memory = sum(item.usedGpuMemory for item in gpu_processes if item.pid not in known_own_pids)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            observed_own_cuda |= any(item.pid in known_own_pids for item in gpu_processes)
            own_peak, device_peak = max(own_peak, own_memory), max(device_peak, memory.used)
            host_peak, other_peak = max(host_peak, rss), max(other_peak, other_memory)
            monitor.write(json.dumps({"elapsed_s": elapsed, "own_gpu_bytes": own_memory,
                                      "other_gpu_bytes": other_memory, "device_gpu_bytes": memory.used,
                                      "tree_rss_bytes": rss, "gpu_utilization_percent": util.gpu,
                                      "other_pids": [item.pid for item in gpu_processes if item.pid not in known_own_pids]}) + "\n")
            monitor.flush()
            if elapsed > args.timeout:
                timed_out = True
                for member in reversed(members):
                    try:
                        member.terminate()
                    except psutil.NoSuchProcess:
                        pass
                break
            time.sleep(0.1)
    try:
        returncode = process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        returncode = process.wait()
    elapsed = time.perf_counter() - started
    reader.join(timeout=10)
    files = []
    values = {}
    for path in sorted((run / "output").rglob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(run)), "bytes": path.stat().st_size,
                          "sha256": sha256(path)})
            if path.suffix == ".json" and path.parent.name in payloads:
                values[path.name] = json.loads(path.read_text())
    confidence = [key for key in values if key.startswith("confidence_")]
    affinity = [key for key in values if key.startswith("affinity_")]
    structures = [row for row in files if row["path"].endswith((".cif", ".pdb")) and "/predictions/" in row["path"]]
    checks = {
        "exit_zero": returncode == 0, "not_timed_out": not timed_out,
        "expected_structure_count": len(structures) == len(payloads),
        "expected_confidence_count": len(confidence) == len(payloads),
        "affinity_present_if_requested": len(affinity) == len(affinity_inputs),
        "numeric_json_finite": finite(values), "own_cuda_process_seen": observed_own_cuda,
    }
    result = {
        "label": args.label, "started_utc": utc, "end_to_end_s": elapsed,
        "gpu_memory_sampled_peak_mib": own_peak / 2**20 if observed_own_cuda else None,
        "device_memory_sampled_peak_mib": device_peak / 2**20,
        "other_processes_sampled_peak_mib": other_peak / 2**20,
        "host_process_tree_rss_peak_mib": host_peak / 2**20,
        "returncode": returncode, "completion_status": "success" if all(checks.values()) else "incomplete_or_error",
        "output_checks": checks, "outputs": values, "files": files,
        "stage_log_events": stages, "background_processes_before": initial_processes,
        "stage_timing_note": "Log arrival markers only; not exact GPU stage timings.",
    }
    (run / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key not in {"files", "outputs"}}, indent=2), flush=True)
    raise SystemExit(0 if result["completion_status"] == "success" else 1)


if __name__ == "__main__":
    main()
