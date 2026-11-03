"""Finite validation plan: repeated single input, then stock directory batching."""

import json
import subprocess
import sys
import time
from pathlib import Path

from compare_outputs import compare
from validate_outputs import validate


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = str(ROOT / "scripts/optimized_cli.py")
STATUS = ROOT / "reports/final-validation-progress.json"
steps = [
    ("combined-l3001-first", "L3001", True, "b1-l3001-first"),
    ("combined-l3001-repeat1", "L3001", True, "b1-l3001-first"),
    ("combined-l3001-repeat2", "L3001", True, "b1-l3001-first"),
    ("b1-l3001-final-control", "L3001", False, "b1-l3001-first"),
]
steps += [("b1-fixed-set", "fixed_set", False, None),
          ("combined-fixed-set", "fixed_set", True, "b1-fixed-set")]

completed = []
for index, (label, input_name, optimized, reference) in enumerate(steps):
    progress = {"status": "running", "current": label, "step": index + 1,
                "total": len(steps), "completed": completed}
    STATUS.write_text(json.dumps(progress, indent=2))
    print("Starting or validating " + label, flush=True)
    result_path = ROOT / "runs" / label / "result.json"
    if (ROOT / "runs" / label).exists() and not result_path.exists():
        # A previously started benchmark was kept running while only its
        # orchestration process was replaced by this batched validation plan.
        deadline = time.monotonic() + 1800
        while not result_path.exists():
            assert time.monotonic() < deadline, "Waiting for existing run timed out"
            time.sleep(2)
    if not result_path.exists():
        command = [sys.executable, str(ROOT / "scripts/run_benchmark.py"), label, "--input", input_name]
        if optimized:
            command += ["--launcher", LAUNCHER]
        with (ROOT / "runs" / (label + "-console.log")).open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    validation = validate(label)
    assert validation["complete"] and validation["all_prediction_arrays_and_coordinates_finite"]
    if reference:
        difference = compare(reference, label)
        destination = ROOT / "reports" / (label + "-comparison.json")
        destination.write_text(json.dumps(difference, indent=2))
        assert difference["both_complete"] and difference["same_json_metric_keys"]
        assert difference["all_json_metrics_equal"] and difference["all_prediction_arrays_equal"], label
    result = json.loads((ROOT / "runs" / label / "result.json").read_text())
    row = {"run": label, "input": input_name, "optimized": optimized,
           "end_to_end_s": result["end_to_end_s"],
           "gpu_memory_sampled_peak_mib": result["gpu_memory_sampled_peak_mib"],
           "validated": True, "exact_match_to_reference": reference}
    completed.append(row)
    print(json.dumps(row), flush=True)

STATUS.write_text(json.dumps({"status": "complete", "completed": completed}, indent=2))
print("All planned baseline/candidate comparisons completed with exact output equality.", flush=True)
