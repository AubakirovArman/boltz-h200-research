"""Evaluate a predeclared ligand-specific confidence selector using cached poses.

Select the highest ligand_iptm among five poses without affinity labels. Reuse
rank0/rank1 seed1 affinity values from the two existing experiments and calculate
only the 23 remaining selected poses. Resource times are not an E2E benchmark.
"""

import argparse
import csv
import datetime
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import ensemble_repeat as shared
import pose_repeat as geometry


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=HERE/'runs/issue-settings-csv-seed0'
CONTROL=HERE/'experiments/affinity-seed1-fixed-structures-20261103'
RANK1=HERE/'experiments/pose-rank1-seed1-20261103'
EXPERIMENT=HERE/'experiments/ligand-iptm-selector-seed1-20261103'


def make_pose(target,name,rank):
    src=BASE/target/'output'/('boltz_results_'+target)/'predictions'/name
    with np.load(src/('pre_affinity_'+name+'.npz'),allow_pickle=True) as f:
        data={k:f[k].copy() for k in f.files}
    ref=data['coords']['coords'].astype(np.float64);keys=[None]*len(ref);ca=[];lig=[]
    for c in data['chains']:
        for r in data['residues'][int(c['res_idx']):int(c['res_idx']+c['res_num'])]:
            for i in range(int(r['atom_idx']),int(r['atom_idx']+r['atom_num'])):
                keys[i]=(str(c['name']),str(int(r['res_idx'])+1),str(data['atoms'][i]['name']))
                if c['mol_type']==0 and data['atoms'][i]['name']=='CA':ca.append(i)
                if c['mol_type']==3:lig.append(i)
    assert None not in keys and len(set(keys))==len(keys)
    old=geometry.cif_coordinates(src/(name+'_model_0.cif'),keys)
    assert abs(old-ref).max()<1e-4
    xyz=geometry.cif_coordinates(src/(name+f'_model_{rank}.cif'),keys)
    ca=np.array(ca);lig=np.array(lig)
    distances=np.linalg.norm(ref[ca,None,:]-ref[lig][None,:,:],axis=2).min(axis=1)
    pocket=ca[distances<=12.]
    if len(pocket)<3:pocket=ca[np.argsort(distances)[:3]]
    x=xyz[pocket];y=ref[pocket];u,_,vt=np.linalg.svd((x-x.mean(0)).T@(y-y.mean(0)))
    rotation=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
    assert abs(np.linalg.det(rotation)-1)<1e-10
    aligned=((xyz-x.mean(0))@rotation+y.mean(0)).astype(np.float32)
    data['atoms']['coords']=aligned;data['coords']['coords']=aligned
    return data,{'id':name,'pose_rank':rank,'atoms':len(ref),'source_cif_sha256':shared.digest(src/(name+f'_model_{rank}.cif')),
                 'pocket_ligand_rmsd_A':float(np.sqrt(np.mean(np.sum((aligned[lig]-ref[lig])**2,axis=1))))}


def prepare():
    assert not EXPERIMENT.exists()
    assert subprocess.check_output(['git','status','--short'],cwd=ROOT/'upstream',text=True)==''
    dataset=json.loads((HERE/'dataset.json').read_text());selection={};conf_hashes={}
    for row in dataset:
        name=row['Target ID'];t=row['target'];pred=BASE/t/'output'/('boltz_results_'+t)/'predictions'/name
        candidates=[]
        for rank in range(5):
            p=pred/(f'confidence_{name}_model_{rank}.json');value=json.loads(p.read_text())
            score=float(value['ligand_iptm']);assert math.isfinite(score)
            candidates.append((score,rank));conf_hashes[str(p.relative_to(ROOT))]=shared.digest(p)
        # A lower composite-confidence rank breaks ties deterministically.
        best=max(candidates,key=lambda x:(x[0],-x[1]))
        selection[name]={'target':t,'rank':best[1],'ligand_iptm':best[0],
                         'reused_from':('rank0_control' if best[1]==0 else 'rank1_experiment' if best[1]==1 else 'new_inference')}
    fresh=[name for name,v in selection.items() if v['rank']>=2]
    EXPERIMENT.mkdir(parents=True)
    jobs=[];audit=[]
    for target in ['L1000','L3000']:
        ids=sorted(name for name in fresh if selection[name]['target']==target)
        if not ids:continue
        work=EXPERIMENT/target;inputs=work/'inputs'/target;inputs.mkdir(parents=True)
        source=BASE/target/'output'/('boltz_results_'+target)
        out=work/'output'/('boltz_results_'+target)
        shutil.copytree(source/'processed',out/'processed')
        for p in (out/'processed/records').glob('*.json'):
            if p.stem not in ids:p.unlink()
        records=json.loads((source/'processed/manifest.json').read_text())['records']
        shared.save(out/'processed/manifest.json',{'records':[r for r in records if r['id'] in ids]})
        for name in ids:
            shutil.copy2(HERE/'inputs'/target/(name+'.yaml'),inputs/(name+'.yaml'))
            data,meta=make_pose(target,name,selection[name]['rank']);audit.append(meta)
            pred=out/'predictions'/name;pred.mkdir(parents=True)
            np.savez_compressed(pred/('pre_affinity_'+name+'.npz'),**data)
        command=[sys.executable,'-m','boltz.main','predict',str(inputs),'--model','boltz2','--accelerator','gpu','--devices','1',
                 '--cache',str(ROOT/'cache'),'--out_dir',str(work/'output'),'--recycling_steps','10','--sampling_steps','200',
                 '--diffusion_samples','5','--sampling_steps_affinity','200','--diffusion_samples_affinity','5','--seed','1',
                 '--num_workers','2','--preprocessing-threads','1']
        config={'target':target,'ids':ids,'command':command,'output':str(out),'seed':1,'phase':'affinity_only_selected_ligand_confidence_pose'}
        shared.save(work/'config.json',config);jobs.append(config)
    shared.save(EXPERIMENT/'geometry-audit.json',audit)
    shared.save(EXPERIMENT/'plan.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'question':'Does pose selection by maximum ligand_iptm improve affinity versus the composite-confidence rank0 choice?',
        'selection':selection,'jobs':[{'target':j['target'],'ids':j['ids']} for j in jobs],
        'fixed_rule':'Highest ligand_iptm among all five original poses; ties choose smaller rank. No affinity labels or new affinity values used.',
        'pairs':140,'new_inference_pairs':len(fresh),'reused_pairs':140-len(fresh),
        'seed':1,'cached_values_note':'Rank0/rank1 values come from the same fixed seed1 shard controls. Fresh ranks2-4 use two smaller seed1 CLI calls; RNG streams differ. This is an exploratory selector test, not exact matched-noise isolation.',
        'ensemble':'Fixed equal average of rank0 control and selected prediction, including identical values when rank0 is selected.',
        'bootstrap':{'resamples':2500,'seed':20261103},'no_label_fitting':True,'no_performance_claim':True,
        'confidence_sha256':conf_hashes,'runner_sha256':shared.digest(Path(__file__)),
        'geometry_dependency_sha256':shared.digest(Path(geometry.__file__))})
    print(json.dumps({'prepared':str(EXPERIMENT),'pairs':140,'fresh':len(fresh),'jobs':len(jobs)},indent=2))


def run():
    import pynvml
    plan=json.loads((EXPERIMENT/'plan.json').read_text())
    assert shared.digest(Path(__file__))==plan['runner_sha256']
    assert shared.digest(Path(geometry.__file__))==plan['geometry_dependency_sha256']
    assert not (EXPERIMENT/'supervisor.json').exists()
    shared.save(EXPERIMENT/'supervisor.json',{'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':shared.GPU_UUID if hasattr(shared,'GPU_UUID') else shared.GPU,
        'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','NUMBA_NUM_THREADS':'8',
        'PYTHONUNBUFFERED':'1','BOLTZ_CACHE':str(ROOT/'cache'),'CUEQ_DEFAULT_CONFIG':'1','CUEQ_DISABLE_AOT_TUNING':'1'})
    pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByUUID(shared.GPU);results=[];start=time.monotonic()
    for job in plan['jobs']:
        work=EXPERIMENT/job['target'];config=json.loads((work/'config.json').read_text());peak=0;began=time.monotonic();last=0
        with (work/'stdout.log').open('wb') as log:
            process=subprocess.Popen(config['command'],cwd=ROOT/'upstream',env=env,stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                mem={p.pid:p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)}
                peak=max(peak,mem.get(process.pid,0))
                if time.monotonic()-last>15:
                    progress={'status':'running','target':job['target'],'pid':process.pid,'expected':len(job['ids']),
                              'new_predictions':len(list(EXPERIMENT.glob('*/output/*/predictions/*/affinity_*.json'))),
                              'fresh_expected':plan['new_inference_pairs'],'elapsed_s':time.monotonic()-start,'peak_mib':peak/2**20}
                    shared.save(EXPERIMENT/'progress.json',progress);print(json.dumps(progress),flush=True);last=time.monotonic()
                assert time.monotonic()-start<7200,'Two-hour safety bound'
                time.sleep(.5)
            found={p.stem.removeprefix('affinity_') for p in Path(config['output']).rglob('affinity_*.json')}
            text=(work/'stdout.log').read_text(errors='replace')
            result={'target':job['target'],'pairs':len(found),'returncode':process.returncode,'elapsed_s':time.monotonic()-began,
                    'gpu_peak_sampled_mib':peak/2**20,'structure_launched':'Running structure prediction for' in text}
            shared.save(work/'result.json',result);results.append(result)
            assert process.returncode==0 and found==set(job['ids']) and not result['structure_launched']
        finally:
            if process.poll() is None:os.killpg(process.pid,15)
    # Only one extra GPU child is active at a time. Rank1 data are shared with
    # the larger parallel experiment and may finish later than the 23 fresh cases.
    needed=[name for name,v in plan['selection'].items() if v['rank']==1]
    while True:
        available={p.stem.removeprefix('affinity_') for p in RANK1.glob('*/output/*/predictions/*/affinity_*.json')}
        if set(needed)<=available:break
        assert time.monotonic()-start<7200,'Rank1 dependency did not finish'
        shared.save(EXPERIMENT/'progress.json',{'status':'waiting_for_shared_rank1_outputs','available':len(set(needed)&available),
                    'needed':len(needed),'new_predictions':plan['new_inference_pairs'],'elapsed_s':time.monotonic()-start})
        time.sleep(10)
    shared.save(EXPERIMENT/'result.json',{'completed':True,'fresh_pairs':plan['new_inference_pairs'],'jobs':results,'elapsed_s':time.monotonic()-start,
                                       'phase':'exploratory ligand-confidence selector; cached affinity values reused'})
    analyze()


def analyze():
    plan=json.loads((EXPERIMENT/'plan.json').read_text())
    for path,sha in plan['confidence_sha256'].items():assert shared.digest(ROOT/path)==sha
    rows=list(csv.DictReader((CONTROL/'all-predictions.csv').open()));assert len(rows)==140
    rank1={p.stem.removeprefix('affinity_'):json.loads(p.read_text())['affinity_pred_value'] for p in RANK1.glob('*/output/*/predictions/*/affinity_*.json')}
    fresh={p.stem.removeprefix('affinity_'):json.loads(p.read_text())['affinity_pred_value'] for p in EXPERIMENT.glob('*/output/*/predictions/*/affinity_*.json')}
    table=[]
    for row in rows:
        name=row['id'];rank=plan['selection'][name]['rank'];control=float(row['seed1_raw'])
        selected=control if rank==0 else float(rank1[name]) if rank==1 else float(fresh[name])
        assert math.isfinite(selected)
        table.append({**row,'pose_rank':rank,'original':float(row['raw']),'control':control,'selected':selected,'ensemble':.5*(control+selected)})
    modes=['original','control','selected','ensemble'];per={};arrays={}
    for target in ['L1000','L3000']:
        group=[r for r in table if r['target']==target];truth=np.array([float(r['true_log10_ic50_um']) for r in group]);pred={}
        for mode in modes:
            for kind,key in [('processed','processed_mw'),('csv','mw')]:
                pred[mode+'_'+kind]=np.array([1.03525938*r[mode]-.59992683*float(r[key])**.3+2.83288489 for r in group])
        per[target]={'n':len(group),'r':{k:float(np.corrcoef(v,truth)[0,1]) for k,v in pred.items()}};arrays[target]=(truth,pred)
    aggregate={k:sum(g['n']*g['r'][k] for g in per.values())/140 for k in per['L1000']['r']}
    rng=np.random.default_rng(20261103);draws=[]
    for _ in range(2500):
        value={m:0. for m in modes}
        for y,pred in arrays.values():
            idx=rng.integers(0,len(y),len(y))
            for m in modes:value[m]+=len(y)/140*float(np.corrcoef(pred[m+'_processed'][idx],y[idx])[0,1])
        draws.append([value[m] for m in modes])
    draws=np.array(draws)
    delta=lambda a,b:np.quantile(draws[:,modes.index(a)]-draws[:,modes.index(b)],[.025,.975]).tolist()
    summary={'pairs':140,'fresh_pairs':plan['new_inference_pairs'],'per_target':per,'weighted_r':aggregate,
        'bootstrap':{'replicates':2500,'seed':20261103,'selected_vs_control_delta_ci95':delta('selected','control'),
                     'selected_vs_original_delta_ci95':delta('selected','original'),'ensemble_vs_original_delta_ci95':delta('ensemble','original')},
        'caveat':plan['cached_values_note'],'no_label_based_selection':True}
    shared.save(EXPERIMENT/'analysis.json',summary)
    with (EXPERIMENT/'all-predictions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    # Independent Pearson implementation on the completed table.
    for key in aggregate:
        m,kind=key.rsplit('_',1);mass='processed_mw' if kind=='processed' else 'mw'
        value=0.
        for t in ['L1000','L3000']:
            group=[r for r in table if r['target']==t]
            x=[1.03525938*r[m]-.59992683*float(r[mass])**.3+2.83288489 for r in group]
            y=[math.log10(float(r['true_ic50_um'])) for r in group]
            value+=len(group)/140*statistics.correlation(x,y)
        assert abs(value-aggregate[key])<1e-12,key
    shared.save(EXPERIMENT/'independent-verification.json',{'passed':True,'pairs':140,'fresh_pairs':len(fresh),'independent_pearson_tolerance':1e-12,
        'scope':'Exploratory research consistency checks, not official tests.'})
    lines=['# Выбор позы по ligand_iptm','',
        'Поза заранее выбирается по максимальному ligand_iptm среди пяти исходных предсказаний; экспериментальные значения affinity не используются. Все 140 пар сохранены. Для rank0/rank1 переиспользованы уже полученные seed1-прогнозы, дополнительно рассчитаны 23 позы rank2–4. Это не измерение скорости полного решения.','',
        '| Вариант | Взвешенный R, MW CLI | Взвешенный R, MW CSV |','|---|---:|---:|']
    for mode,label in zip(modes,['Исторический исходный','Контроль rank0, seed1','Выбор по ligand_iptm','Среднее контроля и выбранной позы']):
        lines.append(f"| {label} | {aggregate[mode+'_processed']:.6f} | {aggregate[mode+'_csv']:.6f} |")
    lines+=['','95% интервал изменения выбранной позы против контроля: '+str(summary['bootstrap']['selected_vs_control_delta_ci95']),
            '95% интервал изменения выбранной позы против исторического исходного: '+str(summary['bootstrap']['selected_vs_original_delta_ci95']),
            '',plan['cached_values_note'],'',
            'Малое различие R нельзя уверенно приписать только позе: последовательности RNG новых малых CLI отличаются от контроля. Структурная гипотеза исследовательская; независимый набор белков не проверялся.']
    (EXPERIMENT/'REPORT_RU.md').write_text('\n'.join(lines)+'\n')
    shared.save(EXPERIMENT/'progress.json',{'status':'completed','pairs':140,'fresh_pairs':len(fresh),'analysis_complete':True})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','analyze'])
    {'prepare':prepare,'run':run,'analyze':analyze}[p.parse_args().action]()
