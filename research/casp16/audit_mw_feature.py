"""Audit CSV molecular mass versus the exact pinned CLI preprocessing feature.

Does not change inputs, model outputs or the original scoring files.
"""

import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / 'runs/issue-settings-csv-seed0'


def main():
    examples = []
    for smiles in ['CC', 'c1ccccc1', 'CC(=O)O']:
        source = Chem.MolFromSmiles(smiles)
        removed = Chem.RemoveHs(Chem.AddHs(source), sanitize=False)
        stale = Descriptors.MolWt(removed)
        removed.UpdatePropertyCache(strict=False)
        examples.append({'smiles': smiles, 'source_MolWt': Descriptors.MolWt(source),
                         'RemoveHs_sanitize_false_MolWt': stale,
                         'after_UpdatePropertyCache_MolWt': Descriptors.MolWt(removed)})
    rows = list(csv.DictReader((ROOT/'reports/casp16-issue636/predictions-and-labels.csv').open()))
    per_target = {}
    values = []
    for target in ['L1000', 'L3000']:
        processed = BASE/target/'output'/('boltz_results_'+target)/'processed'
        records = json.loads((processed/'manifest.json').read_text())['records']
        mw = {r['id']: r['affinity']['mw'] for r in records}
        group = [r for r in rows if r['target']==target]
        truth = [float(r['true_log10_ic50_um']) for r in group]
        corrected = []
        for r in group:
            prediction = 1.03525938*float(r['raw']) - .59992683*mw[r['id']]**.3 + 2.83288489
            corrected.append(prediction)
            values.append({'id':r['id'], 'target':target, 'csv_molwt':float(r['mw']),
                           'processed_manifest_mw':mw[r['id']], 'raw':float(r['raw']),
                           'cli_mw_corrected_fp64':prediction})
        per_target[target]={'n':len(group),'r':float(np.corrcoef(corrected,truth)[0,1])}
    result = {
        'source':'parse/schema.py:1274 -> Record.affinity.mw -> featurizerv2.py:2308 -> Boltz2 MW correction',
        'prior_published_csv_mw_weighted_r':0.6410222018383349,
        'processed_cli_mw_per_target':per_target,
        'processed_cli_mw_weighted_r':sum(v['n']*v['r'] for v in per_target.values())/140,
        'preprocessing_finding':'With the pinned RDKit, AddHs followed by RemoveHs(sanitize=False) leaves a molecular-weight property-cache difference; UpdatePropertyCache restores the implicit-hydrogen contribution in the examples. Chemical standardization can additionally change the source molecule.',
        'minimal_examples':examples,
        'note':'Same official correction coefficients; different MW inputs. The old CSV-MW calculation remains valid as such, but is not an exact reconstruction of the CLI MW feature. No inference, fitting or source change.',
        'pairs':values,
    }
    output=ROOT/'reports/casp16-issue636/MW_INPUT_CLARIFICATION.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))


if __name__=='__main__':
    main()
