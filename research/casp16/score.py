"""Compare the four predeclared scoring variants without rerunning Boltz."""

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def correlation(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(pearsonr(x, y).statistic)


def mean_correlation(groups, field, weighted):
    values = [(correlation(np.array([r[field] for r in rows]), np.array([r["true_log10_ic50_um"] for r in rows])), len(rows))
              for rows in groups.values()]
    if any(value is None for value, _ in values):
        return None
    return float(np.average([value for value, _ in values], weights=[n if weighted else 1 for _, n in values]))


def score(run_root, bootstrap=2500):
    dataset = json.loads((HERE / "dataset.json").read_text())
    predictions = {}
    for path in run_root.rglob("affinity_*.json"):
        name = path.stem.removeprefix("affinity_")
        if name in predictions:
            raise ValueError("Duplicate prediction for " + name)
        predictions[name] = (path, json.loads(path.read_text()))
    missing = [item["Target ID"] for item in dataset if item["Target ID"] not in predictions]
    rows = []
    for item in dataset:
        name = item["Target ID"]
        if name not in predictions:
            continue
        path, values = predictions[name]
        raw = float(values["affinity_pred_value"])
        corrected = 1.03525938 * raw - 0.59992683 * item["molecular_weight"] ** 0.3 + 2.83288489
        row = {"id": name, "target": item["target"], "task": item["Task"],
               "true_ic50_um": float(item["IC50"]),
               "true_log10_ic50_um": float(np.log10(float(item["IC50"]))),
               "mw": item["molecular_weight"], "raw": raw, "mw_corrected": corrected,
               "mw_only": -item["molecular_weight"] ** 0.3,
               "model1": float(values["affinity_pred_value1"]),
               "model2": float(values["affinity_pred_value2"]),
               "probability_binary": float(values["affinity_probability_binary"]),
               "prediction_file": str(path.relative_to(ROOT))}
        assert all(np.isfinite(value) for value in row.values() if isinstance(value, float)), name
        assert abs(raw - (row["model1"] + row["model2"]) / 2) < 1e-5, "Expected raw ensemble without MW correction"
        rows.append(row)
    groups = {target: [row for row in rows if row["target"] == target]
              for target in ["L1000", "L3000"]}
    groups = {target: values for target, values in groups.items() if values}
    fields = ["raw", "mw_corrected", "mw_only", "model1", "model2", "probability_binary"]
    per_target = {}
    for target, values in groups.items():
        truth = np.array([row["true_log10_ic50_um"] for row in values])
        per_target[target] = {"n": len(values), "metrics": {field: {
            "pearson_r": correlation(np.array([row[field] for row in values]), truth),
            "kendall_tau": float(kendalltau(np.array([row[field] for row in values]), truth).statistic)
        } for field in fields}}
    summary = {"coverage": {"expected": len(dataset), "available": len(rows), "missing": missing,
                             "complete": not missing}, "per_target": per_target,
               "aggregate": {field: {"uniform_target_mean": mean_correlation(groups, field, False),
                                     "compound_weighted_target_mean": mean_correlation(groups, field, True)}
                             for field in fields},
               "correction_parameters_source": "Official pinned source; no coefficients fitted to CASP16",
               "bootstrapping": None}
    if not missing and bootstrap:
        rng = np.random.default_rng(20261103)
        distributions = {f"{field}/{weight}": [] for field in ["raw", "mw_corrected"] for weight in ["uniform", "compound_weighted"]}
        for _ in range(bootstrap):
            sampled = {target: [values[i] for i in rng.integers(0, len(values), size=len(values))]
                       for target, values in groups.items()}
            for field in ["raw", "mw_corrected"]:
                for weight in ["uniform", "compound_weighted"]:
                    value = mean_correlation(sampled, field, weight == "compound_weighted")
                    if value is not None:
                        distributions[f"{field}/{weight}"].append(value)
        summary["bootstrapping"] = {"replicates": bootstrap, "seed": 20261103,
                                    "method": "Target-stratified paired resampling; percentile intervals",
                                    "ci95": {key: np.quantile(values, [0.025, 0.975]).tolist()
                                             for key, values in distributions.items()}}
        a = np.asarray(distributions["mw_corrected/compound_weighted"])
        b = np.asarray(distributions["raw/compound_weighted"])
        if len(a) == len(b):
            summary["bootstrapping"]["mw_effect_weighted_ci95"] = np.quantile(a - b, [0.025, 0.975]).tolist()
    for task in ["PA", "A"]:
        values = [row for row in rows if row["target"] == "L3000" and row["task"] == task]
        if values:
            summary.setdefault("l3000_subsets_diagnostic", {})[task] = {
                "n": len(values), **{field: correlation(np.array([row[field] for row in values]),
                                                       np.array([row["true_log10_ic50_um"] for row in values]))
                                    for field in ["raw", "mw_corrected"]}}
    output = run_root / "scoring-summary.json"
    output.write_text(json.dumps(summary, indent=2))
    if rows:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        (run_root / "predictions-and-labels.csv").write_text(stream.getvalue())
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2500)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = score(args.run_root.resolve(), args.bootstrap)
    if args.require_complete:
        assert result["coverage"]["complete"], result["coverage"]
    print(json.dumps(result, indent=2))
