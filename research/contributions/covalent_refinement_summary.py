"""Recalculate the PR #717 refinement tables from all retained samples."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def metrics(rows):
    distances = [c["distance_A"] for r in rows for c in r["closures"]]
    return {
        "structures": len(rows),
        "closure_mean_A": float(np.mean(distances)) if distances else None,
        "closure_min_A": min(distances) if distances else None,
        "closure_max_A": max(distances) if distances else None,
        "closure_within_radii_buffer": sum(c["within_covalent_radii_buffer"] for r in rows for c in r["closures"]),
        "closure_at_most_2_A": sum(c["distance_A"] <= 2 for r in rows for c in r["closures"]),
        "modified_bond_violations": sum(len(r["geometry"]["bond_violations"]) for r in rows),
        "modified_angle_distance_violations": sum(len(r["geometry"]["angle_violations"]) for r in rows),
        "peptide_bond_violations": sum(b["outside_buffered_amide_reference"] for r in rows for b in r["geometry"]["backbone"]),
        "max_physical_energy": {
            key: max(r["geometry"]["energies"].get(key, 0.) for r in rows)
            for key in sorted(set(k for r in rows for k in r["geometry"]["energies"]))
        },
    }


def original_rows(result, audits):
    by_file = {Path(r["file"]).stem: r for r in audits}
    rows = []
    for r in result["rows"]:
        if not r["guided"]:
            continue
        d = r["closure_distance_A"]
        rows.append({"case": "original", "seed": r["seed"], "sample": r["sample"],
                     "closures": [{"distance_A": d, "within_covalent_radii_buffer": 1.58375 <= d <= 2.03625}],
                     "geometry": by_file[Path(r["file"]).stem]})
    return rows


def compare_controls(root, candidate_folder, baseline_folder, cases):
    candidate = read(root / candidate_folder / "result.json")["rows"]
    baseline = read(root / baseline_folder / "result.json")["rows"]
    lookup = {(r["case"], r["seed"], r["sample"]): r for r in baseline}
    comparisons = []
    for row in candidate:
        if row["case"] not in cases:
            continue
        old = lookup[(row["case"], row["seed"], row["sample"])]
        x = np.load(root / candidate_folder / row["file"])["atoms"]["coords"]
        y = np.load(root / baseline_folder / old["file"])["atoms"]["coords"]
        comparisons.append({"case": row["case"], "seed": row["seed"], "sample": row["sample"],
            "bit_equal": bool(np.array_equal(x, y)),
            "max_atom_displacement_A": float(np.linalg.norm(x-y, axis=-1).max()),
            "unaligned_atom_rmsd_A": float(np.sqrt(np.square(x-y).sum(axis=-1).mean()))})
    return comparisons


def main(root, output):
    old_audits = read(root / "original-all-geometry.json")
    baseline = original_rows(read(root.parent / "baseline/result.json"),
                            [r for r in old_audits if "/baseline/" in r["file"]])
    draft = original_rows(read(root.parent / "solution/result.json"),
                         [r for r in old_audits if "/solution/" in r["file"]])
    original = original_rows(read(root / "candidate-v3/result.json"), read(root / "candidate-v3-all-geometry.json"))
    panel = read(root / "candidate-v3-panel/result.json")["rows"]
    holdout = read(root / "final-holdout/result.json")["rows"]
    final = original + [r for r in panel if not r["case"].endswith("_control")] + holdout
    tables = {"original_upstream_seeds_0_1_2": metrics(baseline),
              "original_initial_draft_seeds_0_1_2": metrics(draft),
              "original_final_seeds_0_1_2": metrics(original)}
    for case in ("original", "ace_cys", "ace_cy3_aly"):
        tables[case + "_final_all_seeds"] = metrics([r for r in final if r["case"] == case])
    tables["final_all_intra_chain_cases"] = metrics(final)
    tables["final_prespecified_seed_2026"] = metrics(holdout)
    controls = compare_controls(root, "candidate-v3-panel", "upstream-panel", ["linear_aly_control", "interchain_control"])
    repeat = compare_controls(root, "upstream-control-repeat", "upstream-panel", ["linear_aly_control"])
    prior_unsteered = [r["coordinates_sha256"] for r in read(root.parent / "baseline/result.json")["rows"] if not r["guided"]]
    new_unsteered = [r["coordinates_sha256"] for r in read(root / "candidate-v3/result.json")["rows"] if not r["guided"]]
    stages = ["force-sweep", "candidate-v1", "candidate-panel", "candidate-v2", "candidate-v2-panel",
              "upstream-panel", "candidate-v3", "candidate-v3-panel", "final-holdout", "upstream-control-repeat"]
    counts = {name: len(list((root/name).rglob("seed*.npz"))) for name in stages}
    counts["native_cli"] = len(list((root/"native-cli-final").glob("*/predictions/*/*_model_0.cif")))
    result = {"tables": tables, "controls": controls, "baseline_repeat": repeat,
              "unsteered_bit_equal": prior_unsteered == new_unsteered,
              "completed_gpu_structures_by_stage": counts, "total_new_gpu_structures": sum(counts.values()),
              "scope": "Focused geometry checks; all samples included. No experimental fold/affinity accuracy or universal chemistry claim."}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"tables": tables, "total_new_gpu_structures": result["total_new_gpu_structures"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.root, arguments.output)
