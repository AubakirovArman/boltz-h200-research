"""Secondary diagnostics for the frozen ensemble experiment, after completion."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE/'experiments/affinity-seed1-fixed-structures-20261103'


def main():
    rows=list(csv.DictReader((EXPERIMENT/'all-predictions.csv').open()))
    assert len(rows)==140 and len({r['id'] for r in rows})==140
    diagnostics={}
    for target in ['L1000','L3000']:
        group=[r for r in rows if r['target']==target]
        y=np.array([float(r['true_log10_ic50_um']) for r in group])
        mw=np.array([float(r['processed_mw']) for r in group])
        values={}
        for mode,key in [('baseline','raw'),('repeat','seed1_raw'),('ensemble','ensemble_raw')]:
            raw=np.array([float(r[key]) for r in group])
            values[mode]=1.03525938*raw-.59992683*mw**.3+2.83288489
        metrics={mode:{'pearson_r':float(np.corrcoef(pred,y)[0,1]),
                       'kendall_tau':float(kendalltau(pred,y).statistic),
                       'mae_log10_ic50_um':float(np.abs(pred-y).mean())}
                 for mode,pred in values.items()}
        # Center/standardize solely to diagnose shared errors, never to change
        # the submitted predictions or optimize weights against benchmark labels.
        z_y=(y-y.mean())/y.std()
        residuals={}
        for mode in ['baseline','repeat']:
            z=(values[mode]-values[mode].mean())/values[mode].std()
            residuals[mode]=z_y-metrics[mode]['pearson_r']*z
        diagnostics[target]={
            'n':len(group),'metrics':metrics,
            'baseline_repeat_prediction_correlation':float(np.corrcoef(values['baseline'],values['repeat'])[0,1]),
            'shared_standardized_residual_correlation':float(np.corrcoef(residuals['baseline'],residuals['repeat'])[0,1]),
            'mean_absolute_between_run_difference_log10':float(np.abs(values['repeat']-values['baseline']).mean()),
        }
    result={'scope':'Secondary diagnostics, not a fitted prediction method or a new primary endpoint.',
            'per_target':diagnostics,
            'explanation':'High shared residual correlation suggests that seed averaging leaves a substantial common error component. It does not identify its physical cause.'}
    path=EXPERIMENT/'secondary-diagnostics.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
