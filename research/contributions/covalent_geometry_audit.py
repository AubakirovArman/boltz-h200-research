"""Inspect all saved covalent pilot poses, including internal and peptide bonds."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

from covalent_guidance_pilot import (
    ROOT, SCHEMA, Boltz2Featurizer, Boltz2Tokenizer, BoltzSteeringParams, Input,
    asdict, collate, const, load_canonicals, load_molecules, parse_boltz_schema, save,
)
from boltz.data.types import StructureV2
from boltz.model.potentials.potentials import get_potentials


def prepare(schema=SCHEMA):
    molecules = load_canonicals(ROOT / "cache/mols")
    extra = [m["ccd"] for s in schema["sequences"] for m in s.get("protein", {}).get("modifications", [])]
    molecules.update(load_molecules(ROOT / "cache/mols", extra))
    target = parse_boltz_schema("geometry_audit", schema, molecules, ROOT / "cache/mols", boltz_2=True)
    tokenized = Boltz2Tokenizer().tokenize(Input(
        structure=target.structure, msa={}, record=target.record,
        residue_constraints=target.residue_constraints, extra_mols=target.extra_mols))
    torch.manual_seed(0)
    features = Boltz2Featurizer().process(tokenized, molecules=molecules,
        random=np.random.default_rng(42), training=False, max_atoms=None,
        max_tokens=None, max_seqs=const.max_msa_seqs, pad_to_max_seqs=False,
        single_sequence_prop=0., compute_frames=True, compute_constraint_features=True)
    return target, collate([features])


def inspect(xyz, target, feats):
    coords = torch.zeros((1, feats["atom_pad_mask"].shape[1], 3))
    coords[0, :len(xyz)] = torch.as_tensor(xyz.copy())
    flags = BoltzSteeringParams(fk_steering=True, physical_guidance_update=True)
    energies = {}
    for potential in get_potentials(asdict(flags), boltz2=True):
        parameters = potential.compute_parameters(.005)
        energies[type(potential).__name__] = float(potential.compute(coords, feats, parameters)[0])
    bounds = target.residue_constraints.rdkit_bounds_constraints
    ds = np.linalg.norm(xyz[bounds["atom_idxs"][:, 0]] - xyz[bounds["atom_idxs"][:, 1]], axis=1)
    output = {"energies": energies}
    for label, mask in (("bond", bounds["is_bond"]), ("angle", bounds["is_angle"])):
        lo, hi = bounds["lower_bound"] * .875, bounds["upper_bound"] * 1.125
        bad = mask & ((ds < lo) | (ds > hi))
        output[label + "_violations"] = [
            {"indices": [int(a) for a in bounds[i]["atom_idxs"]],
             "atoms": [str(target.structure.atoms[a]["name"]) for a in bounds[i]["atom_idxs"]],
             "distance_A": float(ds[i]), "lo_A": float(lo[i]), "hi_A": float(hi[i])}
            for i in np.flatnonzero(bad)]
    # Same RDKit amide C--N reference and buffer for every exported pose.
    amide = Chem.MolFromSmiles("CC(=O)NC")
    amide_bounds = GetMoleculeBoundsMatrix(amide)
    lo, hi = float(amide_bounds[3, 1] * .875), float(amide_bounds[1, 3] * 1.125)
    backbone = []
    residues = target.structure.residues
    for chain in target.structure.chains:
        start, end = int(chain["res_idx"]), int(chain["res_idx"] + chain["res_num"])
        for i in range(start, end - 1):
            left, right = residues[i], residues[i + 1]
            a = {str(target.structure.atoms[j]["name"]): j for j in range(int(left["atom_idx"]), int(left["atom_idx"] + left["atom_num"]))}
            b = {str(target.structure.atoms[j]["name"]): j for j in range(int(right["atom_idx"]), int(right["atom_idx"] + right["atom_num"]))}
            if "C" in a and "N" in b:
                d = float(np.linalg.norm(xyz[a["C"]] - xyz[b["N"]]))
                backbone.append({"residues": [i + 1, i + 2], "distance_A": d,
                                 "outside_buffered_amide_reference": not lo <= d <= hi})
    output["backbone_amide_reference_A"] = [lo, hi]
    output["backbone"] = backbone
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    target, features = prepare()
    results = []
    for folder in args.inputs:
        for p in sorted(folder.rglob("*.npz")):
            structure = StructureV2.load(p)
            row = {"file": str(p), **inspect(structure.atoms["coords"], target, features)}
            results.append(row)
    save(args.output, results)
    print(json.dumps({"poses": len(results), "bond_violations": sum(len(r["bond_violations"]) for r in results),
                      "angle_violations": sum(len(r["angle_violations"]) for r in results),
                      "backbone_violations": sum(b["outside_buffered_amide_reference"] for r in results for b in r["backbone"])}))
