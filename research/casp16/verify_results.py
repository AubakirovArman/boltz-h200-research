"""Independently verify ID joins, literal source coefficients and reported R."""

import argparse
import ast
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def main(run_root):
    reference = json.loads((run_root / "scoring-summary.json").read_text())
    assert reference["coverage"]["complete"]
    source = ast.parse((ROOT / "upstream/src/boltz/model/models/boltz2.py").read_text())
    block = next(node for node in ast.walk(source) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "self.affinity_mw_correction")
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in block.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in {"model_coef", "mw_coef", "bias"}}
    assert constants == {"model_coef": 1.03525938, "mw_coef": -.59992683, "bias": 2.83288489}
    assert any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
               and isinstance(node.right, ast.Constant) and node.right.value == .3
               for node in ast.walk(block))
    predictions = {}
    total_json_files = 0
    for pattern in ["affinity_*.json", "confidence_*.json"]:
        for path in run_root.rglob(pattern):
            values = json.loads(path.read_text())
            assert finite(values), path
            total_json_files += 1
            if pattern.startswith("affinity"):
                name = path.stem.removeprefix("affinity_")
                assert name not in predictions, name
                predictions[name] = values
    labels = {}
    calculations = {}
    max_precision_difference = 0.
    for target, expected in [("L1000", 17), ("L3000", 123)]:
        rows = list(csv.DictReader((HERE / "raw" / (target + "_exper_affinity.csv")).read_text(encoding="utf-8-sig").splitlines()))
        assert len(rows) == expected
        values, weights, truth = [], [], []
        for row in rows:
            name = row["Target ID"]
            assert name not in labels
            labels[name] = row
            prediction = predictions[name]
            raw = float(prediction["affinity_pred_value"])
            assert abs(raw - .5 * (prediction["affinity_pred_value1"] + prediction["affinity_pred_value2"])) < 1e-5
            smiles = row["ligand_smiles"] if target == "L1000" else row["Structure"]
            mw = Descriptors.MolWt(Chem.MolFromSmiles(smiles))
            ic50 = float(row["CHYMASE_IC50_COLI_NH"] if target == "L1000" else row["IC50"])
            values.append(raw)
            weights.append(mw)
            truth.append(math.log10(ic50))
        values, weights, truth = np.asarray(values), np.asarray(weights), np.asarray(truth)
        corrected = constants["model_coef"] * values + constants["mw_coef"] * weights ** .3 + constants["bias"]
        # Match the published floating-point calculation separately from float64 scoring.
        fp32 = (constants["model_coef"] * torch.tensor(values, dtype=torch.float32)
                + constants["mw_coef"] * torch.tensor(weights, dtype=torch.float32).pow(.3)
                + constants["bias"]).numpy()
        fp_error = float(np.max(np.abs(corrected - fp32)))
        max_precision_difference = max(max_precision_difference, fp_error)
        calculations[target] = {"n": expected,
            "raw": float(np.corrcoef(values, truth)[0, 1]),
            "mw_corrected": float(np.corrcoef(corrected, truth)[0, 1]),
            "mw_corrected_fp32": float(np.corrcoef(fp32, truth)[0, 1]),
            "max_fp32_vs_float64_prediction_difference": fp_error}
        for field in ["raw", "mw_corrected"]:
            assert abs(calculations[target][field] - reference["per_target"][target]["metrics"][field]["pearson_r"]) < 1e-12
        assert abs(calculations[target]["mw_corrected"] - calculations[target]["mw_corrected_fp32"]) < 1e-6
    assert set(predictions) == set(labels) and len(labels) == 140
    assert total_json_files == 140 * 6
    for field in ["raw", "mw_corrected"]:
        uniform = sum(calculations[target][field] for target in calculations) / 2
        weighted = sum(calculations[target][field] * calculations[target]["n"] for target in calculations) / 140
        assert abs(uniform - reference["aggregate"][field]["uniform_target_mean"]) < 1e-12
        assert abs(weighted - reference["aggregate"][field]["compound_weighted_target_mean"]) < 1e-12
    result = {"passed": True, "complete_id_join": 140, "finite_json_files": total_json_files,
              "source_constants_verified": constants, "independent_numpy_correlations": calculations,
              "max_fp32_vs_float64_prediction_difference": max_precision_difference}
    (ROOT / "reports/casp16-issue636/independent-verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    main(args.run_root.resolve())
