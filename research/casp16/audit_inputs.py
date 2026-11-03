"""Recompute input-identity and scoring-protocol evidence from frozen sources."""

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.stats import pearsonr


HERE = Path(__file__).resolve().parent


def main():
    dataset = json.loads((HERE / "dataset.json").read_text())
    differences = []
    for row in dataset:
        if row["csv_smiles_matches_affinity_tsv"]:
            continue
        molecule = Chem.MolFromSmiles(row["Structure"])
        identity = Chem.MolToSmiles(molecule, isomericSmiles=False)
        candidates = [item for item in row["official_tsv_rows"] if "A" in item["Task"]]
        same_graph = any(Chem.MolToSmiles(Chem.MolFromSmiles(item["SMILES"]), isomericSmiles=False) == identity
                         for item in candidates)
        differences.append({"id": row["Target ID"], "target": row["target"],
                            "type": "stereochemistry" if same_graph else "different_molecular_graph",
                            "csv_smiles": row["Structure"], "official_rows": row["official_tsv_rows"]})
    (HERE / "input-structure-audit.json").write_text(json.dumps(differences, indent=2))
    comments = json.loads((HERE / "sources/issue636-comments.json").read_text())
    comment = next(item for item in comments if item["url"].endswith("6127069488"))
    labels = {row["Target ID"]: row for row in dataset}
    rows = []
    for line in comment["body"].splitlines():
        if not re.match(r"\| L1\d+ \|", line):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        row = labels[cells[0]]
        raw = float(cells[3])
        mw = row["molecular_weight"]
        rows.append({"id": cells[0], "ic50_um": float(row["IC50"]), "prediction_raw_rounded": raw,
                     "mw": mw, "mw_corrected": 1.03525938 * raw - .59992683 * mw ** .3 + 2.83288489})
    assert len(rows) == 17
    y = np.log10([row["ic50_um"] for row in rows])
    public = {"n": len(rows),
              "raw_r": float(pearsonr(y, [row["prediction_raw_rounded"] for row in rows]).statistic),
              "corrected_r": float(pearsonr(y, [row["mw_corrected"] for row in rows]).statistic),
              "source": comment["url"],
              "scope": "Recalculation of rounded public comment predictions, not a new model experiment", "rows": rows,
              "issue_rounded_values_aggregation": {"uniform": (.44 + .59) / 2,
                                                   "weighted": (17 * .44 + 123 * .59) / 140}}
    (HERE / "comment-l1000-recalculation.json").write_text(json.dumps(public, indent=2))
    obsolete = {row["Target ID"]: row for row in csv.DictReader(
        (HERE / "raw/L1000_exper_affinity_obsolete.csv").read_text(encoding="utf-8-sig").splitlines())}
    facts = {"reported_issue_correlations": {
        "L1000": .44, "L3000": .59, "counts": {"L1000": 17, "L3000": 123},
        "uniform_mean": (.44 + .59) / 2, "compound_weighted_mean": (17 * .44 + 123 * .59) / 140},
        "public_L1000_predictions": {key: value for key, value in public.items() if key != "rows"},
        "input_differences": {"total": len(differences), **dict(Counter(row["type"] for row in differences))},
        "unit_conversion_checks": {}, "mw_only_baseline": {}, "obsolete_label_diagnostic": {}}
    for target in ["L1000", "L3000"]:
        group = [row for row in dataset if row["target"] == target]
        truth = np.log10([float(row["IC50"]) for row in group])
        energy = np.array([float(row["binding_affinity"]) for row in group])
        facts["unit_conversion_checks"][target] = {
            "correlation_log10_um_vs_official_energy": float(pearsonr(truth, energy).statistic)}
        facts["mw_only_baseline"][target] = {"n": len(group),
            "pearson_r": float(pearsonr(truth, [-row["molecular_weight"] ** .3 for row in group]).statistic)}
    for name in ["CHYMASE_IC50_COLI_NH", "CHYMASE_IC50_CATHEP"]:
        truth = np.log10([float(obsolete[row["id"]][name]) for row in rows])
        facts["obsolete_label_diagnostic"][name] = {
            field: float(pearsonr(truth, [row[field] for row in rows]).statistic)
            for field in ["prediction_raw_rounded", "mw_corrected"]}
    (HERE / "confirmed-protocol-findings.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps({key: value for key, value in facts.items() if key != "public_L1000_predictions"}, indent=2))


if __name__ == "__main__":
    main()
