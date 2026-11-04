"""Diagnostic force-balance sweep; every sample is retained, no production edits."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from covalent_guidance_pilot import (
    ROOT, SCHEMA, Boltz2, Boltz2DiffusionParams, Boltz2Featurizer,
    Boltz2Tokenizer, BoltzSteeringParams, Coords, Input, MSAModuleArgs,
    PairformerArgsV2, collate, const, load_canonicals, load_molecules,
    parse_boltz_schema, save, tensor_digest, writer_module,
)
from boltz.model.potentials import potentials as potentials_module
from boltz.model.modules import diffusionv2


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    variants = [("connection_0.03", 0.03, 0.01),
                ("balanced_0.03", 0.03, 0.03),
                ("geometry_0.15", 0.15, 0.15)]
    settings = {"variants": variants, "seeds": [0, 1], "samples": 3,
                "recycles": 3, "steps": 200, "all_samples_retained": True,
                "purpose": "Diagnostic weight sweep; not final production settings."}
    save(out / "settings.json", settings)
    (out / "input.yaml").write_text(yaml.safe_dump(SCHEMA, sort_keys=False))
    save(out / "source.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__), Path(diffusionv2.__file__),
                                        Path(potentials_module.__file__))})
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    molecules = load_canonicals(ROOT / "cache/mols")
    molecules.update(load_molecules(ROOT / "cache/mols", ["ACE", "CY3"]))
    target = parse_boltz_schema("ace_cy3", SCHEMA, molecules, ROOT / "cache/mols", boltz_2=True)
    tokenized = Boltz2Tokenizer().tokenize(Input(
        structure=target.structure, msa={}, record=target.record,
        residue_constraints=target.residue_constraints, extra_mols=target.extra_mols))
    featurizer = Boltz2Featurizer()
    explicit = target.structure.bonds[target.structure.bonds["type"] == const.bond_type_ids["COVALENT"]][0]
    endpoints = [int(explicit["atom_1"]), int(explicit["atom_2"])]
    bounds = target.residue_constraints.rdkit_bounds_constraints
    bond_bounds = bounds[bounds["is_bond"]]
    torch.manual_seed(0)
    start = time.monotonic()
    flags = BoltzSteeringParams(fk_steering=True, physical_guidance_update=True)
    model = Boltz2.load_from_checkpoint(
        ROOT / "cache/boltz2_conf.ckpt", map_location="cpu", strict=True,
        ema=False, use_kernels=True, diffusion_process_args=asdict(Boltz2DiffusionParams()),
        pairformer_args=asdict(PairformerArgsV2()), msa_args=asdict(MSAModuleArgs(use_paired_feature=True)),
        steering_args=asdict(flags)).eval().cuda()
    original_get = diffusionv2.get_potentials
    rows = []
    for variant, connection_weight, geometry_weight in variants:
        folder = out / variant
        folder.mkdir()
        def configured_get(args, boltz2=False):
            result = original_get(args, boltz2=boltz2)
            for potential in result:
                if isinstance(potential, potentials_module.ConnectionsPotential):
                    potential.parameters["guidance_weight"] = connection_weight
                elif isinstance(potential, potentials_module.PoseBustersPotential):
                    potential.parameters["guidance_weight"] = geometry_weight
            return result
        diffusionv2.get_potentials = configured_get
        for seed in settings["seeds"]:
            torch.manual_seed(seed)
            features = featurizer.process(tokenized, molecules=molecules, random=np.random.default_rng(42),
                training=False, max_atoms=None, max_tokens=None, max_seqs=const.max_msa_seqs,
                pad_to_max_seqs=False, single_sequence_prop=0., compute_frames=True, compute_constraint_features=True)
            feats_cpu = collate([features])
            save(folder / f"features_seed{seed}.json", {k: tensor_digest(v) for k, v in feats_cpu.items() if isinstance(v, torch.Tensor)})
            feats = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in feats_cpu.items()}
            torch.manual_seed(seed)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(feats, recycling_steps=3, num_sampling_steps=200,
                    diffusion_samples=3, max_parallel_samples=1, run_confidence_sequentially=True)
            for sample, coords in enumerate(prediction["sample_atom_coords"].cpu().numpy()):
                xyz = coords[:len(target.structure.atoms)]
                assert np.isfinite(xyz).all()
                distances = np.linalg.norm(xyz[bond_bounds["atom_idxs"][:, 0]] - xyz[bond_bounds["atom_idxs"][:, 1]], axis=1)
                lo, hi = bond_bounds["lower_bound"] * .875, bond_bounds["upper_bound"] * 1.125
                atoms = target.structure.atoms.copy()
                atoms["coords"] = xyz
                structure = replace(target.structure, atoms=atoms, coords=np.array([(tuple(x),) for x in xyz], dtype=Coords))
                name = f"seed{seed}_sample{sample}"
                structure.dump(folder / (name + ".npz"))
                (folder / (name + ".cif")).write_text(writer_module.to_mmcif(structure, boltz2=True))
                rows.append({"variant": variant, "seed": seed, "sample": sample,
                    "closure_A": float(np.linalg.norm(xyz[endpoints[0]] - xyz[endpoints[1]])),
                    "bond_violations": int(((distances < lo) | (distances > hi)).sum()),
                    "max_bond_excess_A": float(np.maximum(0, np.maximum(lo - distances, distances - hi)).max()),
                    "plddt": float(prediction["complex_plddt"][sample]), "file": f"{variant}/{name}.npz"})
            save(out / "progress.json", {"rows": rows, "elapsed_s": time.monotonic() - start})
            print(json.dumps(rows[-3:]), flush=True)
    diffusionv2.get_potentials = original_get
    save(out / "result.json", {"settings": settings, "rows": rows, "elapsed_s": time.monotonic() - start})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
