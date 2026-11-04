"""Compare two exports of the same frozen Boltz predictions, without inference."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import gemmi
import numpy as np

from boltz.data.types import StructureV2
from boltz.data.write import pdb
from boltz.data.write.mmcif import to_mmcif

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / 'data/pdb-format'


def metrics(text, expected, file_format):
    structure = gemmi.read_pdb_string(text) if file_format == 'pdb' else gemmi.make_structure_from_block(gemmi.cif.read_string(text).sole_block())
    atoms = [a for c in structure[0] for r in c for a in r]
    xyz = np.array([a.pos.tolist() for a in atoms])
    chain_ids = [c.name for c in structure[0]]
    result = {'atoms': len(atoms), 'chains': chain_ids, 'unique_chain_ids': len(chain_ids) == len(set(chain_ids))}
    if xyz.shape != expected.shape:
        result['shape_matches'] = False
    else:
        distances = np.linalg.norm(xyz - expected, axis=1)
        result.update({'shape_matches': True, 'unaligned_rmsd_A': float(np.sqrt(np.mean(distances**2))),
                       'max_atom_displacement_A': float(distances.max())})
    return result


def run(input_directory):
    path = ROOT / 'upstream/src/boltz/data/write/pdb.py'
    spec = importlib.util.spec_from_file_location('original_pdb_writer', path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    rows = []
    paths = sorted(input_directory.glob('*/pre_affinity_*.npz'))
    if not paths:
        paths = sorted(input_directory.glob('*.npz'))
    assert len(paths) == 17
    for path in paths:
        source = StructureV2.load(path)
        assert len(source.chains) == 2 and source.atoms['is_present'].all()
        expected = source.atoms['coords'].copy()
        for label, names in [('native', ['A', 'B']), ('long', ['prot1', 'lig1']),
                             ('same_initial', ['A1', 'A2']), ('reserved', ['A', 'prot1'])]:
            chains = source.chains.copy()
            chains['name'] = names
            structure = replace(source, chains=chains)
            baseline = original.to_pdb(structure, boltz2=True)
            fixed = pdb.to_pdb(structure, boltz2=True)
            cif = to_mmcif(structure, boltz2=True)
            result = {'id': path.stem.removeprefix('pre_affinity_'), 'case': label,
                      'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'original_pdb': metrics(baseline, expected, 'pdb'),
                      'proposed_pdb': metrics(fixed, expected, 'pdb'),
                      'mmcif': metrics(cif, expected, 'cif')}
            assert result['proposed_pdb']['max_atom_displacement_A'] < .00087
            assert result['proposed_pdb']['unique_chain_ids']
            assert result['mmcif']['max_atom_displacement_A'] < 9e-6
            assert np.array_equal(source.atoms['coords'], expected)
            if label == 'native':
                assert baseline == fixed
                result['native_pdb_byte_identical'] = True
            rows.append(result)
    return {'predictions': len(paths), 'naming_cases_per_prediction': 4, 'export_comparisons': len(rows),
            'no_inference_rerun': True, 'rows': rows,
            'max_original_long_name_displacement_A': max(r['original_pdb']['max_atom_displacement_A'] for r in rows if r['case'] != 'native'),
            'max_proposed_displacement_A': max(r['proposed_pdb']['max_atom_displacement_A'] for r in rows),
            'native_pdb_byte_identical_in_all_cases': True,
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(pdb.__file__), ROOT / 'upstream/src/boltz/data/write/pdb.py']},
            'limitations': ['Original issue #696 supplies no YAML/coordinates/seed; this demonstrated writer defect cannot establish the exact cause of those images.',
                           'PDB uses 3 coordinate decimals, mmCIF uses 5; small rounding differences are expected.',
                           'Independent model runs can differ through sampling; all comparisons here reuse the exact same stored coordinates.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--inputs', type=Path, default=INPUTS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve previous results.'
    result = run(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
