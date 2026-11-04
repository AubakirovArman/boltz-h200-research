"""Audit native CLI CIF output independently of the prediction writer."""
import argparse
import json
import math
from pathlib import Path

import gemmi
import numpy as np
import torch

from covalent_geometry_audit import prepare, inspect
from covalent_refinement_validation import cases


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def main(root, output):
    torch.set_num_threads(2)
    rows = []
    for case in ("original", "ace_cy3_aly"):
        paths = list(root.glob(f"*/predictions/{case}/{case}_model_0.cif"))
        assert len(paths) == 1, (case, paths)
        path = paths[0]
        structure = gemmi.read_structure(str(path))
        observed = {(chain.name, residue.seqid.num, atom.name): np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    for chain in structure[0] for residue in chain for atom in residue}
        target, features = prepare(cases()[case])
        xyz = np.zeros((len(target.structure.atoms), 3), dtype=np.float32)
        for chain in target.structure.chains:
            for residue in target.structure.residues[chain["res_idx"]:chain["res_idx"]+chain["res_num"]]:
                for i in range(int(residue["atom_idx"]), int(residue["atom_idx"]+residue["atom_num"])):
                    key = (str(chain["name"]), int(residue["res_idx"])+1, str(target.structure.atoms[i]["name"]))
                    xyz[i] = observed.pop(key)
        assert not observed
        assert np.isfinite(xyz).all()
        geometry = inspect(xyz, target, features)
        confidence = json.loads((path.parent/f"confidence_{case}_model_0.json").read_text())
        assert finite(confidence)
        pair = features["connected_atom_index"][0, :, 0].tolist()
        closure = float(np.linalg.norm(xyz[pair[0]]-xyz[pair[1]]))
        lower = float(features["connected_atom_lower_bounds"][0, 0])
        upper = float(features["connected_atom_upper_bounds"][0, 0])
        rows.append({"case": case, "cif": str(path), "atoms": len(xyz),
                     "closure_A": closure, "lower_A": lower, "upper_A": upper,
                     "closure_in_range": lower <= closure <= upper,
                     "geometry": geometry, "confidence_finite": True})
    result = {"rows": rows, "all_closures_in_range": all(r["closure_in_range"] for r in rows),
              "modified_bond_violations": sum(len(r["geometry"]["bond_violations"]) for r in rows),
              "modified_angle_distance_violations": sum(len(r["geometry"]["angle_violations"]) for r in rows),
              "peptide_bond_violations": sum(b["outside_buffered_amide_reference"] for r in rows for b in r["geometry"]["backbone"])}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
