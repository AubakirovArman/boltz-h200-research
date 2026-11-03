"""Compare actual prediction values, not ZIP timestamps or filesystem metadata."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def numbers(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(numbers(child, prefix + "/" + key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(numbers(child, prefix + "/" + str(index)))
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def compare_arrays(first, second, prefix, report):
    if first.shape != second.shape or first.dtype != second.dtype:
        report[prefix] = {"compatible": False, "shape_a": list(first.shape), "shape_b": list(second.shape),
                          "dtype_a": str(first.dtype), "dtype_b": str(second.dtype)}
        return False
    if first.dtype.names:
        equal = True
        for name in first.dtype.names:
            equal &= compare_arrays(first[name], second[name], prefix + "/" + name, report)
        return equal
    if first.dtype.hasobject:
        def plain(value):
            if isinstance(value, np.ndarray):
                return plain(value.tolist())
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {str(key): plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            return value
        equal = plain(first) == plain(second)
    else:
        equal = np.array_equal(first, second)
    row = {"equal": bool(equal), "shape": list(first.shape)}
    if np.issubdtype(first.dtype, np.number):
        a, b = first.astype(np.float64), second.astype(np.float64)
        row["bit_equal"] = first.tobytes(order="C") == second.tobytes(order="C")
        row.update({"all_finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
                    "max_abs_difference": float(np.max(np.abs(a - b))) if a.size else 0,
                    "mean_abs_difference": float(np.mean(np.abs(a - b))) if a.size else 0})
    report[prefix] = row
    return equal


def compare(first_name, second_name):
    first = ROOT / "runs" / first_name
    second = ROOT / "runs" / second_name
    a, b = (json.loads((root / "result.json").read_text()) for root in [first, second])
    x, y = numbers(a["outputs"]), numbers(b["outputs"])
    metrics = {key: {"baseline": x[key], "candidate": y[key], "abs_difference": abs(x[key] - y[key])}
               for key in sorted(x.keys() & y.keys())}
    snapshots = []
    for root in [first, second]:
        snapshots.append({str(p).split("/predictions/", 1)[1]: p
                          for p in (root / "output").rglob("*.npz") if "/predictions/" in str(p)})
    arrays = {}
    all_equal = snapshots[0].keys() == snapshots[1].keys()
    for key in sorted(snapshots[0].keys() & snapshots[1].keys()):
        # These are our own recorded Boltz output files, not user-supplied archives.
        # StructureV2 includes an object-encoded symmetry dictionary.
        with np.load(snapshots[0][key], allow_pickle=True) as left, np.load(snapshots[1][key], allow_pickle=True) as right:
            all_equal &= set(left.files) == set(right.files)
            for name in sorted(set(left.files) & set(right.files)):
                all_equal &= compare_arrays(left[name], right[name], key + "/" + name, arrays)
    report = {
        "baseline": first_name, "candidate": second_name,
        "both_complete": a["completion_status"] == b["completion_status"] == "success",
        "speedup": a["end_to_end_s"] / b["end_to_end_s"],
        "end_to_end_s": {"baseline": a["end_to_end_s"], "candidate": b["end_to_end_s"]},
        "sampled_peak_mib": {"baseline": a["gpu_memory_sampled_peak_mib"], "candidate": b["gpu_memory_sampled_peak_mib"]},
        "same_json_metric_keys": x.keys() == y.keys(),
        "all_json_metrics_equal": x == y,
        "all_json_numbers_bit_equal": x.keys() == y.keys() and all(
            np.float64(x[key]).tobytes() == np.float64(y[key]).tobytes() for key in x),
        "all_prediction_arrays_equal": bool(all_equal),
        "all_numeric_prediction_arrays_bit_equal": all(
            row.get("bit_equal", True) for row in arrays.values()),
        "json_metrics": metrics, "array_comparisons": arrays,
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in {"json_metrics", "array_comparisons"}}, indent=2))
