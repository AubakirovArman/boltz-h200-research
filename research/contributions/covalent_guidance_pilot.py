"""Fixed small #675 pilot: native model/settings, all samples, no MSA."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from boltz.data import const
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.module.inferencev2 import collate
from boltz.data.mol import load_canonicals, load_molecules
from boltz.data.parse import schema as schema_module
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import Coords, Input
from boltz.data.write import mmcif as writer_module
from boltz.main import Boltz2DiffusionParams, BoltzSteeringParams, MSAModuleArgs, PairformerArgsV2
from boltz.model.models.boltz2 import Boltz2
import boltz.data.feature.featurizerv2 as features_module

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = {'recycles': 3, 'diffusion_steps': 200, 'samples': 3, 'max_parallel_samples': 1,
            'seeds': [0, 1, 2], 'unsteered_control_seeds': [0], 'all_samples_included': True,
            'msa': 'empty', 'templates': False,
            'guidance': 'Native CLI --use_potentials: FK steering and physical guidance enabled; default contact guidance retained.',
            'rng': 'Set torch seed separately before feature construction and before each forward; model loaded once.',
            'scope': 'Direct released model/feature pipeline, not a CLI runtime benchmark.'}
SCHEMA = {'version': 1, 'sequences': [{'protein': {'id': 'B', 'sequence': 'GYYPRVARLIFVAIYC',
           'msa': 'empty', 'cyclic': False, 'modifications': [{'position': 1, 'ccd': 'ACE'}, {'position': 16, 'ccd': 'CY3'}]}}],
          'constraints': [{'bond': {'atom1': ['B', 1, 'CH3'], 'atom2': ['B', 16, 'SG']}}]}


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def tensor_digest(value):
    a = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'settings.json', SETTINGS)
    (out / 'input.yaml').write_text(yaml.safe_dump(SCHEMA, sort_keys=False))
    sources = [Path(features_module.__file__), Path(schema_module.__file__), Path(writer_module.__file__)]
    save(out / 'source.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    molecules = load_canonicals(ROOT / 'cache/mols')
    molecules.update(load_molecules(ROOT / 'cache/mols', ['ACE', 'CY3']))
    target = parse_boltz_schema('ace_cy3', SCHEMA, molecules, ROOT / 'cache/mols', boltz_2=True)
    inputs = Input(structure=target.structure, msa={}, record=target.record,
                   residue_constraints=target.residue_constraints, extra_mols=target.extra_mols)
    tokenized = Boltz2Tokenizer().tokenize(inputs)
    featurizer = Boltz2Featurizer()
    torch.manual_seed(0)
    start = time.monotonic()
    model = Boltz2.load_from_checkpoint(ROOT / 'cache/boltz2_conf.ckpt', map_location='cpu', strict=True,
        ema=False, use_kernels=True, diffusion_process_args=asdict(Boltz2DiffusionParams()),
        pairformer_args=asdict(PairformerArgsV2()), msa_args=asdict(MSAModuleArgs(use_paired_feature=True)),
        steering_args=asdict(BoltzSteeringParams())).eval().cuda()
    print(json.dumps({'model_loaded_s': time.monotonic() - start}), flush=True)
    explicit = [b for b in target.structure.bonds if int(b['type']) == const.bond_type_ids['COVALENT']]
    assert len(explicit) == 1
    endpoints = [int(explicit[0]['atom_1']), int(explicit[0]['atom_2'])]
    bounds = target.residue_constraints.rdkit_bounds_constraints
    bond_bounds = bounds[bounds['is_bond']]
    rows = []
    features_record = {}
    for seed in SETTINGS['seeds']:
        torch.manual_seed(seed)
        feats_cpu = featurizer.process(tokenized, molecules=molecules, random=np.random.default_rng(42), training=False,
            max_atoms=None, max_tokens=None, max_seqs=const.max_msa_seqs, pad_to_max_seqs=False,
            single_sequence_prop=0., compute_frames=True, compute_constraint_features=True)
        feats = collate([feats_cpu])
        features_record[str(seed)] = {k: tensor_digest(v) for k, v in feats.items() if isinstance(v, torch.Tensor)}
        features_record[str(seed)]['connected_atom_pairs'] = feats['connected_atom_index'].tolist()
        features_record[str(seed)]['connected_chain_pairs'] = feats['connected_chain_index'].tolist()
        feats = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in feats.items()}
        for guided in ([True, False] if seed == 0 else [True]):
            flags = BoltzSteeringParams()
            flags.fk_steering = guided
            flags.physical_guidance_update = guided
            model.steering_args = asdict(flags)
            torch.manual_seed(seed)
            begin = time.monotonic()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(feats, recycling_steps=SETTINGS['recycles'],
                    num_sampling_steps=SETTINGS['diffusion_steps'], diffusion_samples=SETTINGS['samples'],
                    max_parallel_samples=1, run_confidence_sequentially=True)
            coords = prediction['sample_atom_coords'].cpu().numpy()
            assert len(coords) == 3 and np.isfinite(coords).all()
            for sample in range(3):
                xyz = coords[sample, :len(target.structure.atoms)]
                distance = float(np.linalg.norm(xyz[endpoints[0]] - xyz[endpoints[1]]))
                modified_distances = np.linalg.norm(xyz[bond_bounds['atom_idxs'][:, 0]] - xyz[bond_bounds['atom_idxs'][:, 1]], axis=1)
                violations = (modified_distances < bond_bounds['lower_bound'] * .875) | (modified_distances > bond_bounds['upper_bound'] * 1.125)
                atoms = target.structure.atoms.copy()
                atoms['coords'] = xyz
                structure = replace(target.structure, atoms=atoms, coords=np.array([(tuple(x),) for x in xyz], dtype=Coords))
                name = f"seed{seed}_{'guided' if guided else 'unsteered'}_sample{sample}"
                structure.dump(out / (name + '.npz'))
                (out / (name + '.cif')).write_text(writer_module.to_mmcif(structure, boltz2=True))
                rows.append({'seed': seed, 'guided': guided, 'sample': sample, 'closure_distance_A': distance,
                             'within_existing_connection_upper_bound_2A': distance <= 2.,
                             'modified_bonds_outside_rdkit_bounds_with_12_5_percent_buffer': int(violations.sum()),
                             'modified_bond_count': len(bond_bounds),
                             'plddt': float(prediction['complex_plddt'][sample].item()),
                             'coordinates_sha256': hashlib.sha256(xyz.tobytes()).hexdigest(), 'file': name + '.cif'})
            save(out / 'progress.json', {'completed_structures': len(rows), 'last_seed': seed, 'guided': guided,
                                         'last_forward_s': time.monotonic() - begin})
            print(json.dumps({'seed': seed, 'guided': guided, 'closure_distances_A': [r['closure_distance_A'] for r in rows[-3:]],
                              'forward_s': time.monotonic() - begin}), flush=True)
    save(out / 'features.json', features_record)
    result = {'settings': SETTINGS, 'structures': len(rows), 'rows': rows, 'elapsed_including_load_s': time.monotonic() - start,
              'peak_torch_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
              'limitations': ['One supplied peptide construct, single-sequence mode, no binding pose/affinity accuracy claim.',
                             'Guidance can improve a declared connection without fixing all internal modified-residue geometry.',
                             'All samples retained; no seed/pose selection.']}
    save(out / 'result.json', result)
    print(json.dumps({'completed': True, 'structures': len(rows), 'elapsed_s': result['elapsed_including_load_s']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
