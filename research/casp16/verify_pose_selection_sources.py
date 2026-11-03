"""Check the selector against raw JSONs and the independent confidence rule."""

import csv
import json
import math
import statistics
from pathlib import Path


HERE=Path(__file__).resolve().parent
BASE=HERE/'runs/issue-settings-csv-seed0'
CONTROL=HERE/'experiments/affinity-seed1-fixed-structures-20261103'
RANK1=HERE/'experiments/pose-rank1-seed1-20261103'
EXPERIMENT=HERE/'experiments/ligand-iptm-selector-seed1-20261103'


def predictions(root):
    result={}
    for p in root.glob('*/output/*/predictions/*/affinity_*.json'):
        name=p.stem.removeprefix('affinity_');assert name not in result
        values=json.loads(p.read_text());assert all(math.isfinite(float(v)) for v in values.values())
        assert abs(values['affinity_pred_value']-.5*(values['affinity_pred_value1']+values['affinity_pred_value2']))<1e-5
        result[name]=values['affinity_pred_value']
    return result


def main():
    control=predictions(CONTROL);rank1=predictions(RANK1);fresh=predictions(EXPERIMENT)
    rows=list(csv.DictReader((EXPERIMENT/'all-predictions.csv').open()));assert len(rows)==140
    plan=json.loads((EXPERIMENT/'plan.json').read_text());mw={};scores={}
    for target in ['L1000','L3000']:
        records=json.loads((BASE/target/'output'/('boltz_results_'+target)/'processed/manifest.json').read_text())['records']
        mw.update({r['id']:r['affinity']['mw'] for r in records})
    assert set(control)==set(rank1)=={r['id'] for r in rows}
    assert set(fresh)=={name for name,r in plan['selection'].items() if r['rank']>=2}
    for r in rows:
        name=r['id'];target=r['target'];source=BASE/target/'output'/('boltz_results_'+target)/'predictions'/name
        candidates=[json.loads((source/f'confidence_{name}_model_{i}.json').read_text())['ligand_iptm'] for i in range(5)]
        chosen=max(range(5),key=lambda i:(candidates[i],-i))
        assert chosen==int(r['pose_rank'])==plan['selection'][name]['rank']
        value=control[name] if chosen==0 else rank1[name] if chosen==1 else fresh[name]
        assert abs(value-float(r['selected']))<1e-15
        assert abs(control[name]-float(r['control']))<1e-15
        assert abs(.5*(control[name]+value)-float(r['ensemble']))<1e-15
        assert abs(mw[name]-float(r['processed_mw']))<1e-12
    for method in ['selected','ensemble']:
        total=0.
        for target in ['L1000','L3000']:
            group=[r for r in rows if r['target']==target]
            x=[1.03525938*float(r[method])-.59992683*mw[r['id']]**.3+2.83288489 for r in group]
            y=[math.log10(float(r['true_ic50_um'])) for r in group]
            total+=len(group)/140*statistics.correlation(x,y)
        scores[method]=total
        assert abs(total-json.loads((EXPERIMENT/'analysis.json').read_text())['weighted_r'][method+'_processed'])<1e-12
    result={'passed':True,'pairs':140,'new_predictions':len(fresh),'pose_choice_recomputed_from_all_five_confidences':True,
            'cached_raw_predictions_verified':True,'exact_cli_mw_verified':True,'weighted_r':scores,
            'scope':'Scientific source audit, not an official upstream test suite.'}
    (EXPERIMENT/'source-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
