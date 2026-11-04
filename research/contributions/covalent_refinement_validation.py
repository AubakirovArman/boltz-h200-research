"""Frozen expanded covalent-guidance panel; all seeds and structures are saved."""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from rdkit import Chem
import torch
import yaml

from covalent_guidance_pilot import (
    ROOT, SCHEMA, Boltz2, Boltz2DiffusionParams, Boltz2Featurizer, Boltz2Tokenizer,
    BoltzSteeringParams, Coords, Input, MSAModuleArgs, PairformerArgsV2, collate,
    const, load_canonicals, load_molecules, parse_boltz_schema, save, tensor_digest,
    writer_module,
)
from covalent_geometry_audit import inspect
import boltz.data.feature.featurizerv2 as feature_module
import boltz.data.parse.schema as schema_module
import boltz.model.modules.diffusionv2 as diffusion_module
import boltz.model.potentials.potentials as potential_module


def cases():
    original = copy.deepcopy(SCHEMA)
    cys = copy.deepcopy(original)
    cys["sequences"][0]["protein"]["modifications"] = [{"position": 1, "ccd": "ACE"}]
    aly = copy.deepcopy(original)
    aly["sequences"][0]["protein"]["modifications"].append({"position": 8, "ccd": "ALY"})
    linear = copy.deepcopy(aly)
    del linear["constraints"]
    inter = {"version": 1, "sequences": [
        {"protein": {"id": "A", "sequence": "GACGYYPR", "msa": "empty"}},
        {"protein": {"id": "B", "sequence": "GAVACGYY", "msa": "empty"}},
    ], "constraints": [{"bond": {"atom1": ["A", 3, "SG"], "atom2": ["B", 5, "SG"]}}]}
    return {"original": original, "ace_cys": cys, "ace_cy3_aly": aly,
            "linear_aly_control": linear, "interchain_control": inter}


def main(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    settings = {"label": args.label, "cases": args.cases, "seeds": args.seeds,
                "recycles": 3, "steps": 200, "samples": 3, "max_parallel_samples": 1,
                "guidance": "native --use_potentials flags", "all_samples_included": True,
                "scope": "Single-sequence geometry study, not a speed or structural-accuracy benchmark.",
                "original": "Exact issue #675 sequence and atom references, with msa: empty.",
                "others": "Explicitly synthetic generalization/control cases; not the unprovided ALY reporter input."}
    save(out / "settings.json", settings)
    paths = [Path(__file__), Path(__file__).with_name("covalent_geometry_audit.py"),
             Path(__file__).with_name("covalent_guidance_pilot.py")]
    paths += [Path(m.__file__) for m in (feature_module, schema_module, diffusion_module, potential_module, writer_module)]
    save(out / "source.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    source = Path(feature_module.__file__).resolve().parents[4]
    save(out / "git.json", {"source": str(source),
        "commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "-C", str(source), "diff"])).hexdigest()})
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    molecules = load_canonicals(ROOT / "cache/mols")
    molecules.update(load_molecules(ROOT / "cache/mols", ["ACE", "CY3", "ALY"]))
    torch.manual_seed(0)
    start = time.monotonic()
    model = Boltz2.load_from_checkpoint(
        ROOT / "cache/boltz2_conf.ckpt", map_location="cpu", strict=True,
        ema=False, use_kernels=True, diffusion_process_args=asdict(Boltz2DiffusionParams()),
        pairformer_args=asdict(PairformerArgsV2()), msa_args=asdict(MSAModuleArgs(use_paired_feature=True)),
        steering_args=asdict(BoltzSteeringParams(fk_steering=True, physical_guidance_update=True))).eval().cuda()
    rows = []
    for case in args.cases:
        schema = cases()[case]
        folder = out / case
        folder.mkdir()
        (folder / "input.yaml").write_text(yaml.safe_dump(schema, sort_keys=False))
        target = parse_boltz_schema(case, schema, molecules, ROOT / "cache/mols", boltz_2=True)
        tokenized = Boltz2Tokenizer().tokenize(Input(
            structure=target.structure, msa={}, record=target.record,
            residue_constraints=target.residue_constraints, extra_mols=target.extra_mols))
        explicit = target.structure.bonds[target.structure.bonds["type"] == const.bond_type_ids["COVALENT"]]
        for seed in args.seeds:
            torch.manual_seed(seed)
            features = Boltz2Featurizer().process(tokenized, molecules=molecules,
                random=np.random.default_rng(42), training=False, max_atoms=None, max_tokens=None,
                max_seqs=const.max_msa_seqs, pad_to_max_seqs=False, single_sequence_prop=0.,
                compute_frames=True, compute_constraint_features=True)
            feats_cpu = collate([features])
            save(folder / f"features_seed{seed}.json", {k: tensor_digest(v) for k,v in feats_cpu.items() if isinstance(v, torch.Tensor)})
            feats = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k,v in feats_cpu.items()}
            torch.manual_seed(seed)
            begin = time.monotonic()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(feats, recycling_steps=3, num_sampling_steps=200,
                    diffusion_samples=3, max_parallel_samples=1, run_confidence_sequentially=True)
            for sample, coords in enumerate(prediction["sample_atom_coords"].cpu().numpy()):
                xyz = coords[:len(target.structure.atoms)]
                assert np.isfinite(xyz).all()
                atoms = target.structure.atoms.copy()
                atoms["coords"] = xyz
                structure = replace(target.structure, atoms=atoms, coords=np.array([(tuple(x),) for x in xyz], dtype=Coords))
                name = f"seed{seed}_sample{sample}"
                structure.dump(folder / (name + ".npz"))
                (folder / (name + ".cif")).write_text(writer_module.to_mmcif(structure, boltz2=True))
                geometry = inspect(xyz, target, feats_cpu)
                closures = []
                for link in explicit:
                    i,j = int(link["atom_1"]), int(link["atom_2"])
                    distance = float(np.linalg.norm(xyz[i] - xyz[j]))
                    elements = feats_cpu["ref_element"][0, [i,j]].argmax(-1).tolist()
                    radius = sum(Chem.GetPeriodicTable().GetRcovalent(e) for e in elements)
                    closures.append({"atoms": [i,j], "distance_A": distance,
                        "within_existing_2A_limit": distance <= 2.,
                        "within_covalent_radii_buffer": .875 * radius <= distance <= 1.125 * radius})
                rows.append({"case": case, "seed": seed, "sample": sample, "closures": closures,
                    "geometry": geometry, "plddt": float(prediction["complex_plddt"][sample]),
                    "coordinates_sha256": hashlib.sha256(xyz.tobytes()).hexdigest(),
                    "file": f"{case}/{name}.npz"})
            save(out / "progress.json", {"rows": rows, "elapsed_s": time.monotonic() - start})
            print(json.dumps({"case": case, "seed": seed, "forward_s": time.monotonic()-begin,
                "closures": [[b["distance_A"] for b in r["closures"]] for r in rows[-3:]],
                "bond_violations": [len(r["geometry"]["bond_violations"]) for r in rows[-3:]],
                "angle_violations": [len(r["geometry"]["angle_violations"]) for r in rows[-3:]]}), flush=True)
    save(out / "result.json", {"settings": settings, "rows": rows, "elapsed_s": time.monotonic() - start,
                                "peak_torch_allocated_MiB": torch.cuda.max_memory_allocated()/2**20})
    print(json.dumps({"completed": True, "structures": len(rows)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cases", choices=list(cases()), nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3,4,5])
    main(parser.parse_args())
