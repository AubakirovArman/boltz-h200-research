"""Independent CPU audit of the published 36-structure ion pilot."""
import hashlib
import itertools
import json
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial.transform import Rotation


def verify(here=None):
    here = Path(here or Path(__file__).resolve().parent)
    plan = json.loads((here / 'plan.json').read_text())
    refs = json.loads((here / 'references.json').read_text())
    analysis = json.loads((here / 'analysis.json').read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(here / 'study.py') == plan['runner_sha256']
    assert sha(here / 'references.json') == plan['references_sha256']
    for name, item in plan['inputs'].items():
        assert sha(here / 'inputs' / (name + '.yaml')) == item['sha256']
    for pdb, ref in refs.items():
        assert sha(here / 'raw' / (pdb + '.cif')) == ref['source_sha256']
    errors = []
    identities = set()
    for row in analysis['rows']:
        key = (row['case'], row['seed'], row['rank'])
        assert key not in identities
        identities.add(key)
        suffix = row['prediction_file'].split('research/ions/', 1)[1]
        structure = gemmi.read_structure(str(here / suffix))
        ref = refs[row['pdb']]
        protein = structure[0]['A']
        sequence = ''.join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in protein)
        assert sequence == ref['sequence'], (key, sequence)
        atoms = {str(r.seqid.num) + ':' + a.name: np.array(a.pos.tolist())
                 for r in protein for a in r}
        ca = sorted((k for k in ref['protein_atoms'] if k.endswith(':CA')),
                    key=lambda k: int(k.split(':')[0]))
        assert len(ca) == row['ca_count'] and set(ca) <= atoms.keys()
        mobile = np.array([atoms[k] for k in ca])
        target = np.array([ref['protein_atoms'][k]['xyz'] for k in ca])
        rotation, _ = Rotation.align_vectors(target - target.mean(0), mobile - mobile.mean(0))
        aligned = rotation.apply(mobile - mobile.mean(0)) + target.mean(0)
        rmsd = np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1)))
        errors.append(abs(float(rmsd) - row['ca_rmsd_A']))
        metals = np.array([a.pos.tolist() for c in structure[0] for r in c
                           if r.name == ref['metal'] for a in r])
        if row['with_ions']:
            assert len(metals) == len(ref['metals_xyz'])
            aligned_m = rotation.apply(metals - mobile.mean(0)) + target.mean(0)
            expected_m = np.array(ref['metals_xyz'])
            # Exhaustive permutations independently check the original Hungarian assignment.
            order = min(itertools.permutations(range(len(metals))),
                        key=lambda p: np.sum((aligned_m[list(p)] - expected_m) ** 2))
            metal_rmsd = np.sqrt(np.mean(np.sum((aligned_m[list(order)] - expected_m) ** 2, axis=1)))
            errors.append(abs(float(metal_rmsd) - row['metal_rmsd_after_protein_alignment_A']))
            distances = []
            for j, donors in enumerate(ref['reference_donors']):
                for k in donors:
                    observed = np.linalg.norm(atoms[k] - metals[order[j]])
                    expected = np.linalg.norm(np.array(ref['protein_atoms'][k]['xyz']) - expected_m[j])
                    distances.append(abs(observed - expected))
            errors.append(abs(float(np.mean(distances)) - row['reference_donor_distance_mae_A']))
        else:
            assert len(metals) == 0
    assert len(identities) == 36
    assert np.isfinite(errors).all() and max(errors) < 1e-10, max(errors)
    for pdb in refs:
        for flag, label in [(False, 'without_ions'), (True, 'with_ions')]:
            group = [r['ca_rmsd_A'] for r in analysis['rows']
                     if r['pdb'] == pdb and r['with_ions'] == flag]
            assert len(group) == 9
            summary = analysis['summary'][pdb][label]
            assert abs(np.mean(group) - summary['ca_rmsd_mean_A']) < 1e-12
            assert abs(np.median(group) - summary['ca_rmsd_median_A']) < 1e-12
    return {'structures_verified': len(identities), 'input_and_reference_hashes_match': True,
            'protein_sequences_match': True, 'independent_rotation_and_exhaustive_ion_assignment': True,
            'max_metric_difference_A': float(max(errors)), 'aggregate_means_and_medians_match': True}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
