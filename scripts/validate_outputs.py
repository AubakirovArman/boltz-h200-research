"""Check saved numerical arrays and atom coordinates from our own CLI runs."""

import argparse
import json
from pathlib import Path

import gemmi
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def is_finite(value):
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            return all(is_finite(value[name]) for name in value.dtype.names)
        if value.dtype.hasobject:
            return all(is_finite(item) for item in value.flat)
        if np.issubdtype(value.dtype, np.number):
            return bool(np.isfinite(value).all())
        return True
    if isinstance(value, dict):
        return all(is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(is_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def validate(label):
    root = ROOT / "runs" / label
    result = json.loads((root / "result.json").read_text())
    predictions = [p for p in (root / "output").rglob("*") if "/predictions/" in str(p) and p.is_file()]
    rows = []
    for path in predictions:
        if path.suffix == ".npz":
            # Files were generated locally by the recorded stock/candidate run.
            with np.load(path, allow_pickle=True) as values:
                rows.append({"file": str(path.relative_to(root)), "all_finite": all(is_finite(values[key]) for key in values.files),
                             "arrays": list(values.files)})
        elif path.suffix == ".cif":
            block = gemmi.cif.read_file(str(path)).sole_block()
            xyz = np.array([[float(value) for value in block.find_values("_atom_site.Cartn_" + axis)]
                            for axis in ["x", "y", "z"]])
            rows.append({"file": str(path.relative_to(root)), "atoms": xyz.shape[1],
                         "all_finite": xyz.size > 0 and bool(np.isfinite(xyz).all())})
    report = {"run": label, "complete": result["completion_status"] == "success",
              "all_prediction_arrays_and_coordinates_finite": bool(rows) and all(row["all_finite"] for row in rows),
              "files": rows}
    (root / "validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    args = parser.parse_args()
    report = validate(args.label)
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))
    raise SystemExit(0 if report["complete"] and report["all_prediction_arrays_and_coordinates_finite"] else 1)
