"""Test confidence-rank-1 poses against the completed rank-0 affinity control.

All 140 complexes are retained. The alternative pose and equal ensemble rule
are fixed before inference. No upstream model source or benchmark label is used
to select poses or weights. CIF rounding and rigid alignment are audited.
"""

import argparse
import csv
import datetime
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

import ensemble_repeat as shared


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE/'runs/issue-settings-csv-seed0'
CONTROL = HERE/'experiments/affinity-seed1-fixed-structures-20261103'
EXPERIMENT = HERE/'experiments/pose-rank1-seed1-20261103'
MODES = ['original','control','pose1','matched_ensemble','original_pose1_ensemble']


def cif_coordinates(path, keys):
    import gemmi
    table=gemmi.cif.read(str(path)).sole_block().find('_atom_site.',[
        'label_asym_id','auth_seq_id','label_atom_id','Cartn_x','Cartn_y','Cartn_z'])
    lookup={}
    for entry in table:
        row=list(entry); key=tuple(row[:3])
        assert key not in lookup,(path,key)
        lookup[key]=[float(v) for v in row[3:]]
    assert set(lookup)==set(keys) and len(lookup)==len(keys),(path,len(lookup),len(keys))
    result=np.array([lookup[k] for k in keys])
    assert np.isfinite(result).all(),path
    return result


def alternative_pose(target,name):
    source=BASE/target/'output'/('boltz_results_'+target)/'predictions'/name
    with np.load(source/('pre_affinity_'+name+'.npz'),allow_pickle=True) as loaded:
        data={k:loaded[k].copy() for k in loaded.files}
    ref=data['coords']['coords'].astype(np.float64)
    assert np.array_equal(data['atoms']['coords'],data['coords']['coords'])
    assert data['atoms']['is_present'].all()
    keys=[None]*len(ref);ca=[];ligand=[]
    for chain in data['chains']:
        for residue in data['residues'][int(chain['res_idx']):int(chain['res_idx']+chain['res_num'])]:
            for i in range(int(residue['atom_idx']),int(residue['atom_idx']+residue['atom_num'])):
                keys[i]=(str(chain['name']),str(int(residue['res_idx'])+1),str(data['atoms'][i]['name']))
                if chain['mol_type']==0 and data['atoms'][i]['name']=='CA':ca.append(i)
                if chain['mol_type']==3:ligand.append(i)
    assert None not in keys and len(set(keys))==len(keys)
    rank0=cif_coordinates(source/(name+'_model_0.cif'),keys)
    rank1=cif_coordinates(source/(name+'_model_1.cif'),keys)
    roundtrip=float(abs(rank0-ref).max())
    assert roundtrip<1e-4,(name,roundtrip)
    ca=np.asarray(ca);ligand=np.asarray(ligand)
    assert len(ca)>=3 and len(ligand)>0
    distance=np.sqrt(np.sum((ref[ca,None,:]-ref[ligand][None,:,:])**2,axis=2)).min(axis=1)
    pocket=ca[distance<=12.]
    if len(pocket)<3:pocket=ca[np.argsort(distance)[:3]]
    x=rank1[pocket];y=ref[pocket]
    u,_,vt=np.linalg.svd((x-x.mean(0)).T@(y-y.mean(0)))
    rotation=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
    assert abs(np.linalg.det(rotation)-1)<1e-10
    aligned=(rank1-x.mean(0))@rotation+y.mean(0)
    # Applying one proper rigid transform to every atom preserves all distances.
    sample=np.unique(np.linspace(0,len(ref)-1,min(len(ref),32),dtype=int))
    before=np.linalg.norm(rank1[sample,None,:]-rank1[sample][None,:,:],axis=2)
    after=np.linalg.norm(aligned[sample,None,:]-aligned[sample][None,:,:],axis=2)
    assert float(abs(before-after).max())<1e-9
    pose=aligned.astype(np.float32)
    data['atoms']['coords']=pose
    data['coords']['coords']=pose
    rmsd=lambda index:float(np.sqrt(np.mean(np.sum((pose[index]-ref[index])**2,axis=1))))
    meta={'id':name,'target':target,'atom_count':len(ref),'ca_atoms':len(ca),'ligand_atoms':len(ligand),
          'alignment_pocket_ca_atoms':len(pocket),'rank0_cif_max_roundtrip_error_A':roundtrip,
          'rank1_pocket_ligand_rmsd_A':rmsd(ligand),'rank1_pocket_ca_rmsd_A':rmsd(pocket),
          'rank1_global_ca_rmsd_after_pocket_alignment_A':rmsd(ca),
          'rank1_cif_sha256':shared.digest(source/(name+'_model_1.cif')),
          'rank0_cif_sha256':shared.digest(source/(name+'_model_0.cif')),
          'source_pre_affinity_sha256':shared.digest(source/('pre_affinity_'+name+'.npz'))}
    return data,meta


def prepare():
    assert not EXPERIMENT.exists(),'Preserve existing experiment'
    assert json.loads((CONTROL/'independent-verification.json').read_text())['passed']
    assert subprocess.check_output(['git','status','--short'],cwd=ROOT/'upstream',text=True)==''
    jobs=json.loads((CONTROL/'plan.json').read_text())['jobs']
    # Audit all atom mappings before creating GPU jobs or inspecting new scores.
    poses={};audit=[]
    for job in jobs:
        for name in job['ids']:
            poses[name],meta=alternative_pose(job['target'],name);audit.append(meta)
    assert len(poses)==140
    EXPERIMENT.mkdir(parents=True)
    baseline={str(p.relative_to(ROOT)):shared.digest(p) for p in shared.baseline_files()}
    for p in CONTROL.glob('*/output/*/predictions/*/affinity_*.json'):
        baseline[str(p.relative_to(ROOT))]=shared.digest(p)
    for job in jobs:
        oldwork=CONTROL/job['job'];work=EXPERIMENT/job['job']
        original=json.loads((oldwork/'config.json').read_text());oldout=Path(original['output'])
        inputs=work/'inputs'/job['job'];inputs.parent.mkdir(parents=True)
        shutil.copytree(oldwork/'inputs'/job['job'],inputs)
        output=work/'output'/('boltz_results_'+job['job'])
        shutil.copytree(oldout/'processed',output/'processed')
        order=[r['id'] for r in json.loads((oldout/'processed/manifest.json').read_text())['records']]
        assert [p.stem for p in (output/'processed/records').glob('*.json')]==order,'Maintain control input order'
        for name in job['ids']:
            dest=output/'predictions'/name;dest.mkdir(parents=True)
            np.savez_compressed(dest/('pre_affinity_'+name+'.npz'),**poses[name])
        command=[s.replace(str(oldwork),str(work)) for s in original['command']]
        config={**original,'command':command,'output':str(output),'phase':'affinity_on_confidence_rank1_pose',
                'control_input_order':order,'pose_rank':1,'alignment':'Proper rigid alignment to baseline pocket CA within 12 A',
                'control_config_sha256':shared.digest(oldwork/'config.json'),
                'source_structure_sha256':{name:shared.digest(output/'predictions'/name/('pre_affinity_'+name+'.npz')) for name in job['ids']},
                'note':'Rank 1 from the existing five structure samples, chosen by confidence before affinity labels. Same seed and input order as completed rank-0 control; original files read-only.'}
        shared.save(work/'config.json',config)
    shared.save(EXPERIMENT/'geometry-audit.json',{'complete_pairs':140,'atom_mapping':'Exact chain/residue/atom-name join',
                'only_coordinates_changed':True,'rank':1,'entries':audit})
    shared.save(EXPERIMENT/'plan.json',{
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'question':'Does the second confidence-ranked structure improve affinity or a fixed two-pose ensemble?',
        'jobs':jobs,'seed':1,'pose_rank':1,'no_label_based_selection':True,
        'primary_comparison':'Pose rank1 versus completed seed1 rank0 control on all 140 pairs',
        'ensemble':'Equal 0.5/0.5 rank0 control and rank1 predictions; no fitted weights',
        'secondary_comparison':'Original historical prediction and equal original/rank1 ensemble',
        'molecular_weights':'Report exact CLI processed feature as primary and original CSV MolWt as secondary',
        'coordinate_precision':'CIF stores five decimals. Rank0 round-trip is checked within 1e-4 A. Rank1 receives a proper rigid pocket alignment, then original FP32 dtype.',
        'bootstrap_replicates':2500,'bootstrap_seed':20261103,
        'runner_sha256':shared.digest(Path(shared.__file__)),'adapter_sha256':shared.digest(Path(__file__)),
        'baseline_checksums':baseline,'source_status':'unmodified',
    })
    print(json.dumps({'prepared':str(EXPERIMENT),'pairs':140,'jobs':len(jobs),
          'median_ligand_rmsd_A':float(np.median([r['rank1_pocket_ligand_rmsd_A'] for r in audit])),
          'max_rank0_roundtrip_A':max(r['rank0_cif_max_roundtrip_error_A'] for r in audit)},indent=2))


def read_new():
    values={}
    for job in json.loads((EXPERIMENT/'plan.json').read_text())['jobs']:
        config=json.loads((EXPERIMENT/job['job']/'config.json').read_text())
        output=Path(config['output'])
        order=[r['id'] for r in json.loads((output/'processed/manifest.json').read_text())['records']]
        assert order==config['control_input_order'],'Input-order confound'
        for path in output.rglob('affinity_*.json'):
            name=path.stem.removeprefix('affinity_');assert name not in values
            result=json.loads(path.read_text());raw=float(result['affinity_pred_value'])
            assert all(math.isfinite(float(v)) for v in result.values()),path
            assert math.isfinite(raw) and abs(raw-.5*(result['affinity_pred_value1']+result['affinity_pred_value2']))<1e-5
            values[name]=raw
    return values


def analyze():
    from scipy.stats import pearsonr
    fresh=read_new()
    original=list(csv.DictReader((ROOT/'reports/casp16-issue636/predictions-and-labels.csv').open()))
    control={r['id']:r for r in csv.DictReader((CONTROL/'all-predictions.csv').open())}
    assert len(fresh)==140 and set(fresh)=={r['id'] for r in original}
    table=[]
    for row in original:
        name=row['id'];raw0=float(row['raw']);ctl=float(control[name]['seed1_raw']);alt=fresh[name]
        table.append({**row,'processed_mw':float(control[name]['processed_mw']),
                      'original':raw0,'control':ctl,'pose1':alt,'matched_ensemble':.5*(ctl+alt),
                      'original_pose1_ensemble':.5*(raw0+alt)})
    per_target={};arrays={}
    for target in ['L1000','L3000']:
        rows=[r for r in table if r['target']==target]
        y=np.array([float(r['true_log10_ic50_um']) for r in rows]);pred={}
        for mode in MODES:
            for label,key in [('processed','processed_mw'),('csv','mw')]:
                pred[mode+'_'+label]=np.array([1.03525938*r[mode]-.59992683*float(r[key])**.3+2.83288489 for r in rows])
        per_target[target]={'n':len(rows),'r':{k:float(pearsonr(p,y).statistic) for k,p in pred.items()},
                            'rank0_rank1_prediction_correlation':float(np.corrcoef(pred['control_processed'],pred['pose1_processed'])[0,1])}
        arrays[target]=(y,pred)
    aggregate={k:{'weighted':sum(g['n']*g['r'][k] for g in per_target.values())/140,
                  'uniform':sum(g['r'][k] for g in per_target.values())/2} for k in per_target['L1000']['r']}
    rng=np.random.default_rng(20261103);draws=[]
    for _ in range(2500):
        result={k:0. for k in MODES}
        for y,pred in arrays.values():
            idx=rng.integers(0,len(y),len(y))
            for mode in MODES:result[mode]+=len(y)/140*float(np.corrcoef(pred[mode+'_processed'][idx],y[idx])[0,1])
        draws.append([result[m] for m in MODES])
    draws=np.array(draws);delta=lambda a,b:np.quantile(draws[:,MODES.index(a)]-draws[:,MODES.index(b)],[.025,.975]).tolist()
    summary={'pairs':140,'phase':'Affinity on alternative precomputed pose, matched seed1 control',
             'per_target':per_target,'aggregate':aggregate,'bootstrap':{'resamples':2500,'seed':20261103,
             'pose1_vs_control_delta_ci95':delta('pose1','control'),
             'matched_ensemble_vs_control_delta_ci95':delta('matched_ensemble','control'),
             'matched_ensemble_vs_original_delta_ci95':delta('matched_ensemble','original')},
             'no_fitting_or_label_based_pose_selection':True}
    shared.save(EXPERIMENT/'analysis.json',summary)
    with (EXPERIMENT/'all-predictions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['# Проверка второй предсказанной позы комплекса','',
        'Все 140 пар, confidence rank=1 из пяти ранее полученных структур. Выбор позы не зависит от экспериментальных меток. Affinity seed=1 и порядок входов совпадают с предыдущим контролем rank=0. Исходные структуры и прогнозы не изменены.','',
        '| Вариант | Взвешенный R, MW CLI | Взвешенный R, MW CSV |','|---|---:|---:|']
    labels=['Исходный исторический расчёт','Контроль rank0, seed1','Вторая поза rank1, seed1','Среднее rank0/rank1, seed1','Среднее исходного расчёта и rank1']
    for mode,label in zip(MODES,labels):lines.append(f"| {label} | {aggregate[mode+'_processed']['weighted']:.6f} | {aggregate[mode+'_csv']['weighted']:.6f} |")
    lines+=['','95% парный bootstrap-интервал изменения rank1 относительно согласованного контроля: '+str(summary['bootstrap']['pose1_vs_control_delta_ci95']),
            '95% интервал изменения среднего двух поз относительно контроля: '+str(summary['bootstrap']['matched_ensemble_vs_control_delta_ci95']),
            '95% интервал изменения среднего двух поз относительно исходного расчёта: '+str(summary['bootstrap']['matched_ensemble_vs_original_delta_ci95']),
            '', 'Это проверка структуры/cropping на той же модели, не дообучение и не E2E benchmark. CIF имеет округление до пяти десятичных знаков; точность переноса rank0 проверена до 1e-4 Å. Новая поза выровнена одним жёстким поворотом и сдвигом по Cα исходного кармана в радиусе 12 Å. Химия, атомные соответствия, веса модели, dtype и вычисления affinity сохранены.']
    (EXPERIMENT/'REPORT_RU.md').write_text('\n'.join(lines)+'\n')
    verify()
    resources=json.loads((EXPERIMENT/'result.json').read_text()) if (EXPERIMENT/'result.json').exists() else None
    if resources:shared.save(EXPERIMENT/'progress.json',{'status':'completed','predictions':140,'active':[],
                 'completed_jobs':resources['jobs'],'elapsed_s':resources['elapsed_s'],'analysis_complete':True})
    print(json.dumps(summary,indent=2),flush=True)


def verify():
    reference=json.loads((EXPERIMENT/'analysis.json').read_text())
    table=list(csv.DictReader((EXPERIMENT/'all-predictions.csv').open()))
    assert len(table)==140 and len({r['id'] for r in table})==140
    fresh=read_new();assert all(abs(float(r['pose1'])-fresh[r['id']])<1e-15 for r in table)
    check={}
    for target in ['L1000','L3000']:
        rows=[r for r in table if r['target']==target];truth=[math.log10(float(r['true_ic50_um'])) for r in rows]
        check[target]={'n':len(rows),'r':{}}
        for mode in MODES:
            for kind,key in [('processed','processed_mw'),('csv','mw')]:
                p=[1.03525938*float(r[mode])-.59992683*float(r[key])**.3+2.83288489 for r in rows]
                check[target]['r'][mode+'_'+kind]=statistics.correlation(p,truth)
    for key in check['L1000']['r']:
        r=sum(g['n']*g['r'][key] for g in check.values())/140
        assert abs(r-reference['aggregate'][key]['weighted'])<1e-12,key
    plan=json.loads((EXPERIMENT/'plan.json').read_text())
    for path,sha in plan['baseline_checksums'].items():assert shared.digest(ROOT/path)==sha,path
    shared.save(EXPERIMENT/'independent-verification.json',{'passed':True,'pairs':140,'matched_control_input_order':True,
        'independent_pearson_tolerance':1e-12,'preserved_source_files':len(plan['baseline_checksums']),
        'scope':'Research consistency checks, not official upstream tests.','per_target':check})


def run():
    plan=json.loads((EXPERIMENT/'plan.json').read_text())
    assert shared.digest(Path(__file__))==plan['adapter_sha256']
    shared.EXPERIMENT=EXPERIMENT
    shared.analyze=analyze
    shared.run()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','analyze','verify'])
    {'prepare':prepare,'run':run,'analyze':analyze,'verify':verify}[parser.parse_args().action]()
