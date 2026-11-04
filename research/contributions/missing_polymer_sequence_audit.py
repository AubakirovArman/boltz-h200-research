"""Reproduce Boltz #669 with its author's files; no GPU or model weights."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import urllib.request
import warnings

import gemmi
import numpy as np
from scipy.spatial import cKDTree
import yaml

from boltz.data.mol import load_canonicals
from boltz.data.parse import mmcif, pdb
from boltz.data.parse.schema import parse_boltz_schema

ROOT = Path(__file__).resolve().parents[2]
URLS = {
    't1.yaml': ('https://github.com/user-attachments/files/26494950/t1.yaml', '6e2af667b012d32e7ae74dd0bf13ce9a44d8f456664413dc59b839dd401e2083'),
    'vav1_pymol2.pdb': ('https://github.com/user-attachments/files/26494953/vav1_pymol2.pdb', 'fe42bbb3257ea82e758c73884311e6cbc75e4dd24d78501a104d862343640da6'),
}


def run(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in URLS.items():
        target = directory / name
        if not target.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                target.write_bytes(response.read())
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
    sources = {str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
               for m in (mmcif, pdb)}
    result = {'inputs': {n: {'url': u, 'sha256': h} for n, (u, h) in URLS.items()},
              'runtime_sources_sha256': sources}
    stage = 'parse_pdb'
    try:
        molecules = load_canonicals(ROOT / 'cache/mols')
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            parsed = pdb.parse_pdb(str(directory / 'vav1_pymol2.pdb'), mols=molecules,
                                  moldir=str(ROOT / 'cache/mols'), use_assembly=False, compute_interfaces=False)
            result['template_chains'] = [{'name': str(c['name']), 'residues': int(c['res_num']), 'atoms': int(c['atom_num'])}
                                         for c in parsed.data.chains]
            result['template_sequence_lengths'] = {k: len(v) for k, v in parsed.sequences.items()}
            source = gemmi.read_structure(str(directory / 'vav1_pymol2.pdb'))
            source.remove_hydrogens()
            source.remove_alternative_conformations()
            xyz = np.array([a.pos.tolist() for c in source[0] for r in c for a in r])
            observed = parsed.data.coords['coords'][parsed.data.atoms['is_present']]
            distances, _ = cKDTree(xyz).query(observed)
            result['max_parsed_to_source_atom_distance_A'] = float(distances.max())
            assert distances.max() < 1e-4
            stage = 'parse_full_yaml'
            schema = yaml.safe_load((directory / 't1.yaml').read_text())
            schema['templates'][0]['pdb'] = str(directory / 'vav1_pymol2.pdb')
            # This installed RDKit exposes a separate upstream affinity-MW
            # import bug before template parsing. Preserve and report it, then
            # isolate the template workflow without changing protein/ligand input.
            try:
                target = parse_boltz_schema('t1', schema, molecules, ROOT / 'cache/mols', boltz_2=True)
                result['full_author_yaml_completed'] = True
            except AttributeError as error:
                if "Descriptors" not in str(error):
                    raise
                result['full_author_yaml_completed'] = False
                result['unrelated_affinity_error'] = str(error)
                schema['properties'] = []
                stage = 'parse_yaml_template_workflow_without_affinity_property'
                target = parse_boltz_schema('t1', schema, molecules, ROOT / 'cache/mols', boltz_2=True)
                result['template_isolation_change'] = 'Disable affinity property only; retain original protein and ligand sequences and template settings.'
            result['query_chains'] = [c.chain_name for c in target.record.chains]
            result['template_records'] = [asdict(t) for t in target.record.templates]
            assert target.record.templates
            assert {t.query_chain for t in target.record.templates} == {'E', 'F'}
            result['warnings'] = list(dict.fromkeys(str(w.message) for w in emitted))
        result['completed'] = True
    except Exception as error:
        result.update({'completed': False, 'failed_stage': stage,
                       'error_type': type(error).__name__, 'error': str(error)})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'reports/contributions/issue669')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve previous results.'
    result = run(args.inputs.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
