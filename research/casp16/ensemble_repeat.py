"""Repeat stock affinity on frozen structures and evaluate a fixed equal ensemble.

This is a conditional affinity-stage experiment, not an independent full
structure+affinity replicate or a speed benchmark. The published seed-0 outputs
are read-only inputs. No coefficients, seeds or molecules are selected by score.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / 'runs/issue-settings-csv-seed0'
EXPERIMENT = HERE / 'experiments/affinity-seed1-fixed-structures-20261103'
GPU = 'GPU-6b984734-8ed6-11dd-3b9a-0b6c36ba5e11'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def baseline_files():
    return sorted(BASE.rglob('affinity_*.json')) + sorted(BASE.rglob('pre_affinity_*.npz'))


def prepare():
    assert not EXPERIMENT.exists(), 'Experiment already exists; preserve it'
    assert subprocess.check_output(['git', 'status', '--short'], cwd=ROOT/'upstream', text=True) == ''
    EXPERIMENT.mkdir(parents=True)
    dataset = json.loads((HERE/'dataset.json').read_text())
    jobs = []
    for target, shards in [('L1000', 1), ('L3000', 3)]:
        ids = sorted(r['Target ID'] for r in dataset if r['target'] == target)
        source = BASE/target/'output'/('boltz_results_'+target)
        records = json.loads((source/'processed/manifest.json').read_text())['records']
        for shard in range(shards):
            selected = ids[shard::shards]
            name = target + f'-shard{shard+1}'
            work = EXPERIMENT/name
            inputs = work/'inputs'/name
            output = work/'output'/('boltz_results_'+name)
            inputs.mkdir(parents=True)
            for name_id in selected:
                shutil.copy2(HERE/'inputs'/target/(name_id+'.yaml'), inputs/(name_id+'.yaml'))
            # Make an independent copy, not writable hardlinks into seed-0 data.
            shutil.copytree(source/'processed', output/'processed')
            for record in (output/'processed/records').glob('*.json'):
                if record.stem not in selected:
                    record.unlink()
            save(output/'processed/manifest.json', {'records':[r for r in records if r['id'] in selected]})
            for name_id in selected:
                pred = output/'predictions'/name_id
                pred.mkdir(parents=True)
                shutil.copy2(source/'predictions'/name_id/('pre_affinity_'+name_id+'.npz'), pred)
            command = [sys.executable, '-m', 'boltz.main', 'predict', str(inputs),
                       '--model','boltz2','--accelerator','gpu','--devices','1',
                       '--cache',str(ROOT/'cache'),'--out_dir',str(work/'output'),
                       '--recycling_steps','10','--sampling_steps','200','--diffusion_samples','5',
                       '--sampling_steps_affinity','200','--diffusion_samples_affinity','5',
                       '--seed','1','--num_workers','2','--preprocessing-threads','1']
            config = {'job':name,'target':target,'ids':selected,'seed':1,'command':command,
                      'output':str(output),'phase':'affinity_only_on_fixed_seed0_structures',
                      'source_structure_sha256':{i:digest(source/'predictions'/i/('pre_affinity_'+i+'.npz')) for i in selected},
                      'upstream_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'upstream',text=True).strip(),
                      'gpu_uuid':GPU,'monitor_interval_s':0.5,
                      'cached_structure_reuse':True,'performance_comparison':False,
                      'note':'Stock CLI skips structure prediction because all pre_affinity inputs already exist. Affinity is newly calculated; no affinity output is copied.'}
            save(work/'config.json', config)
            jobs.append(config)
    save(EXPERIMENT/'plan.json', {
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'question':'Does fixed equal averaging of seed0 and seed1 affinity improve R on all 140 fixed pairs?',
        'phase':'Conditional affinity repeat with source structures fixed; not a full independent replicate',
        'jobs':[{k:j[k] for k in ['job','target','ids','seed']} for j in jobs],
        'seeds':[0,1],'ensemble_weights':[0.5,0.5],
        'primary_mw':'Exact processed manifest affinity.mw passed to the pinned CLI featurizer',
        'secondary_mw':'RDKit MolWt of original CSV SMILES, for comparison with the earlier published 0.641022',
        'mw_discrepancy_found_before_new_outputs':'Boltz standardizes SMILES before computing the manifest MW; this differs from original CSV molecular weight. Both fixed definitions will be reported.',
        'bootstrap_replicates':2500,'bootstrap_seed':20261103,
        'selection':'All 140 rows; sorted IDs round-robin into 3 L3000 jobs and 1 L1000 job. No label-based selection.',
        'fit':'No fitting, seed selection, head selection or optimized ensemble weights',
        'baseline_checksums':{str(p.relative_to(ROOT)):digest(p) for p in baseline_files()},
        'runner_sha256':digest(Path(__file__)),
    })
    print(json.dumps({'prepared':str(EXPERIMENT),'jobs':len(jobs),'pairs':sum(len(j['ids']) for j in jobs)},indent=2))


def run():
    import pynvml
    plan = json.loads((EXPERIMENT/'plan.json').read_text())
    assert digest(Path(__file__)) == plan['runner_sha256'], 'Freeze code before launch'
    assert not (EXPERIMENT/'supervisor.json').exists(), 'Do not duplicate a running experiment'
    started = time.monotonic()
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByUUID(GPU)
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
                'OPENBLAS_NUM_THREADS':'8','NUMBA_NUM_THREADS':'8','PYTHONUNBUFFERED':'1',
                'BOLTZ_CACHE':str(ROOT/'cache'),'CUEQ_DEFAULT_CONFIG':'1','CUEQ_DISABLE_AOT_TUNING':'1'})
    active = []
    for item in plan['jobs']:
        work = EXPERIMENT/item['job']
        config = json.loads((work/'config.json').read_text())
        log = (work/'stdout.log').open('wb')
        process = subprocess.Popen(config['command'], cwd=ROOT/'upstream', env=env,
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True)
        log.close()
        active.append({'process':process,'config':config,'work':work,'started':time.monotonic(),'peak':0})
    save(EXPERIMENT/'supervisor.json', {'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'children':{j['config']['job']:j['process'].pid for j in active},'phase':'affinity_only','gpu_uuid':GPU})
    last_write = 0
    results = []
    try:
        while active:
            memory = {p.pid:p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu)}
            for job in list(active):
                process = job['process']; config = job['config']; work = job['work']
                job['peak'] = max(job['peak'],memory.get(process.pid,0))
                if process.poll() is not None:
                    found = list(Path(config['output']).rglob('affinity_*.json'))
                    ids = sorted(p.stem.removeprefix('affinity_') for p in found)
                    log = (work/'stdout.log').read_text(errors='replace')
                    result = {'job':config['job'],'returncode':process.returncode,'pairs':len(found),
                              'expected':len(config['ids']),'ids_match':ids==sorted(config['ids']),
                              'elapsed_s':time.monotonic()-job['started'],'gpu_peak_sampled_mib':job['peak']/2**20,
                              'structure_prediction_launched':'Running structure prediction for' in log,
                              'phase':'affinity_only_on_fixed_seed0_structures'}
                    save(work/'result.json',result); results.append(result); active.remove(job)
                    print(json.dumps(result),flush=True)
                    assert result['returncode']==0 and result['ids_match'] and not result['structure_prediction_launched'],result
            if time.monotonic()-last_write>=15:
                progress = {'elapsed_s':time.monotonic()-started,'completed_jobs':results,
                            'active':[{'job':j['config']['job'],'pid':j['process'].pid,
                                       'pairs':len(list(Path(j['config']['output']).rglob('affinity_*.json'))),
                                       'expected':len(j['config']['ids']),'gpu_peak_mib':j['peak']/2**20} for j in active]}
                save(EXPERIMENT/'progress.json',progress)
                print(json.dumps(progress),flush=True);last_write=time.monotonic()
            if time.monotonic()-started>7200:
                raise TimeoutError('Two-hour safety bound for affinity-only experiment')
            time.sleep(0.5)
        for name, sha in plan['baseline_checksums'].items():
            assert digest(ROOT/name)==sha, 'Baseline was changed: '+name
        save(EXPERIMENT/'result.json', {'completed':True,'jobs':results,'elapsed_s':time.monotonic()-started,
                                      'original_baseline_unchanged':True,'phase':'affinity_only'})
        analyze()
    finally:
        for job in active:
            if job['process'].poll() is None:
                os.killpg(job['process'].pid,15)


def analyze():
    import numpy as np
    from scipy.stats import pearsonr
    rows=list(csv.DictReader((ROOT/'reports/casp16-issue636/predictions-and-labels.csv').open()))
    new={}
    for job in json.loads((EXPERIMENT/'plan.json').read_text())['jobs']:
        config=json.loads((EXPERIMENT/job['job']/'config.json').read_text())
        for path in Path(config['output']).rglob('affinity_*.json'):
            name=path.stem.removeprefix('affinity_')
            assert name not in new,name
            values=json.loads(path.read_text())
            value=float(values['affinity_pred_value'])
            assert abs(value-(values['affinity_pred_value1']+values['affinity_pred_value2'])/2)<1e-5
            new[name]=value
    assert set(new)=={r['id'] for r in rows} and len(rows)==140,'Require complete fixed dataset'
    mw={}
    for target in ['L1000','L3000']:
        manifest=json.loads((BASE/target/'output'/('boltz_results_'+target)/'processed/manifest.json').read_text())
        mw.update({r['id']:r['affinity']['mw'] for r in manifest['records']})
    per_target={}; table=[]
    for r in rows:
        old=float(r['raw']); fresh=new[r['id']]
        q={**r,'processed_mw':mw[r['id']],'seed1_raw':fresh,'ensemble_raw':(old+fresh)/2}
        table.append(q)
    variants={}; arrays={}
    for target in ['L1000','L3000']:
        subset=[r for r in table if r['target']==target]
        y=np.array([float(r['true_log10_ic50_um']) for r in subset])
        weights={kind:np.array([float(r[col]) for r in subset]) for kind,col in [('processed','processed_mw'),('csv','mw')]}
        pred={}
        for run,col in [('baseline','raw'),('repeat','seed1_raw'),('ensemble','ensemble_raw')]:
            raw=np.array([float(r[col]) for r in subset]);pred[run+'_raw']=raw
            for kind,w in weights.items(): pred[run+'_'+kind]=1.03525938*raw-.59992683*w**.3+2.83288489
        assert all(np.isfinite(p).all() for p in pred.values()) and np.isfinite(y).all()
        per_target[target]={'n':len(subset),'r':{k:float(pearsonr(v,y).statistic) for k,v in pred.items()}}
        arrays[target]=(y,pred)
    for k in per_target['L1000']['r']:
        variants[k]={'uniform':sum(g['r'][k] for g in per_target.values())/2,
                     'weighted':sum(g['n']*g['r'][k] for g in per_target.values())/140}
    rng=np.random.default_rng(20261103); draws=[]
    for _ in range(2500):
        value={k:0. for k in ['baseline_processed','repeat_processed','ensemble_processed']}
        for target,(y,pred) in arrays.items():
            idx=rng.integers(0,len(y),size=len(y))
            for k in value: value[k]+=len(y)/140*float(np.corrcoef(pred[k][idx],y[idx])[0,1])
        draws.append([value[k] for k in ['baseline_processed','repeat_processed','ensemble_processed']])
    draws=np.array(draws)
    effect=variants['ensemble_processed']['weighted']-variants['baseline_processed']['weighted']
    result={'phase':'Conditional affinity repeat on fixed source structures; NOT a full independent structure replicate',
            'pairs':140,'per_target':per_target,'aggregate':variants,'ensemble_weighted_delta_processed_mw':effect,
            'bootstrap':{'resamples':2500,'seed':20261103,
                         'ensemble_r_ci95':np.quantile(draws[:,2],[.025,.975]).tolist(),
                         'ensemble_delta_ci95':np.quantile(draws[:,2]-draws[:,0],[.025,.975]).tolist()},
            'reach_0_8':variants['ensemble_processed']['weighted']>=.8,
            'note':'All ensemble weights and seeds fixed before prediction; no parameter fitting.'}
    save(EXPERIMENT/'analysis.json',result)
    with (EXPERIMENT/'all-predictions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    report=EXPERIMENT/'REPORT_RU.md'
    lines=['# Повтор affinity и усреднение двух прогнозов','',
           'Условный повтор на сохранённых структурах; это не полный повтор структуры с другим seed.',
           'Все 140 молекул; seed=1 на каждый из четырёх заранее заданных процессов; веса ансамбля 0.5/0.5.','',
           '| Вариант | Взвешенный R, MW из обработанного входа CLI | R, MW из исходного CSV |','|---|---:|---:|']
    for run,label in [('baseline','Исходный расчёт'),('repeat','Новый affinity'),('ensemble','Среднее двух')]:
        lines.append(f"| {label} | {variants[run+'_processed']['weighted']:.6f} | {variants[run+'_csv']['weighted']:.6f} |")
    lines+=['',f'Изменение R от усреднения: {effect:+.6f}.',
            '95% парный bootstrap-интервал изменения: '+str(result['bootstrap']['ensemble_delta_ci95'])+'.','',
            'Масса CLI берётся из processed manifest после стандартной химической обработки. Ранее опубликованные 0.641022 рассчитаны с RDKit MolWt исходных CSV SMILES; с массой именно CLI исходный результат 0.641163. Формула одна, определение входной массы различалось.','',
            'Новый seed также меняет случайные величины стадии affinity, но исходные pre_affinity структуры зафиксированы. Условное улучшение не доказывает переносимость на новые белки. Целевой порог 0.8 проверяется без исключения молекул или выбора лучшего прогона.']
    report.write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','analyze'])
    action=parser.parse_args().action
    {'prepare':prepare,'run':run,'analyze':analyze}[action]()
