"""Independent ID, resource and numeric checks of the conditional ensemble study."""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE/'runs/issue-settings-csv-seed0'
EXPERIMENT = HERE/'experiments/affinity-seed1-fixed-structures-20261103'


def finite(value):
    if isinstance(value,float):
        return math.isfinite(value)
    if isinstance(value,dict):
        return all(finite(v) for v in value.values())
    if isinstance(value,list):
        return all(finite(v) for v in value)
    return True


def main():
    reference=json.loads((EXPERIMENT/'analysis.json').read_text())
    plan=json.loads((EXPERIMENT/'plan.json').read_text())
    resources=json.loads((EXPERIMENT/'result.json').read_text())
    assert resources['completed'] and len(resources['jobs'])==4
    for job in resources['jobs']:
        assert job['returncode']==0 and job['ids_match'] and not job['structure_prediction_launched']
    for name,expected in plan['baseline_checksums'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    predictions={}
    for job in plan['jobs']:
        config=json.loads((EXPERIMENT/job['job']/'config.json').read_text())
        paths=list(Path(config['output']).rglob('affinity_*.json'))
        assert {p.stem.removeprefix('affinity_') for p in paths}==set(job['ids'])
        for p in paths:
            name=p.stem.removeprefix('affinity_'); assert name not in predictions
            value=json.loads(p.read_text()); assert finite(value)
            predictions[name]=value
    rows=list(csv.DictReader((ROOT/'reports/casp16-issue636/predictions-and-labels.csv').open()))
    assert len(rows)==140 and {r['id'] for r in rows}==set(predictions)
    calculations={}
    for target in ['L1000','L3000']:
        records=json.loads((BASE/target/'output'/('boltz_results_'+target)/'processed/manifest.json').read_text())['records']
        mw={r['id']:r['affinity']['mw'] for r in records}
        group=[r for r in rows if r['target']==target]
        labels=[math.log10(float(r['true_ic50_um'])) for r in group]
        values={k:[] for k in ['baseline_processed','repeat_processed','ensemble_processed','baseline_csv','repeat_csv','ensemble_csv']}
        for r in group:
            old=json.loads((ROOT/r['prediction_file']).read_text()); fresh=predictions[r['id']]
            raw0=(old['affinity_pred_value1']+old['affinity_pred_value2'])/2
            raw1=(fresh['affinity_pred_value1']+fresh['affinity_pred_value2'])/2
            assert abs(raw0-float(r['raw']))<1e-5 and abs(raw1-fresh['affinity_pred_value'])<1e-5
            # Use literal stored ensemble outputs for the independent score;
            # heads above independently check their interpretation.
            for mode,raw in [('baseline',old['affinity_pred_value']),('repeat',fresh['affinity_pred_value']),('ensemble',(old['affinity_pred_value']+fresh['affinity_pred_value'])/2)]:
                for definition,mass in [('processed',mw[r['id']]),('csv',float(r['mw']))]:
                    values[mode+'_'+definition].append(1.03525938*raw-.59992683*mass**.3+2.83288489)
        calculations[target]={'n':len(group),'r':{k:statistics.correlation(v,labels) for k,v in values.items()}}
    for k in calculations['L1000']['r']:
        weighted=sum(v['n']*v['r'][k] for v in calculations.values())/140
        assert abs(weighted-reference['aggregate'][k]['weighted'])<1e-12,k
    result={'passed':True,'unique_pairs':140,'new_affinity_json_finite':140,'cli_jobs':4,
            'independent_pearson_agreement_tolerance':1e-12,'baseline_files_unchanged':len(plan['baseline_checksums']),
            'baseline_source_files_byte_identical':True,'structure_stage_was_reused':True,
            'scope':'Research completeness and arithmetic checks, not official upstream tests.',
            'independent_calculations':calculations}
    (EXPERIMENT/'independent-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
