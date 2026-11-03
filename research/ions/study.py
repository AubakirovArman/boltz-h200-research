"""Fixed, small ion/no-ion cofolding pilot for Boltz issue #711.

All predictions are included. Reference model 1, resolved protein C-alpha RMSD,
and metal coordination metrics are fixed before inference. No MSA/templates or
force potentials: this measures a restricted single-sequence inference setting.
"""
import argparse,datetime,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path

import numpy as np
import gemmi
import yaml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUNS=HERE/'runs'
CASES=[('1ZNF','ZN'),('1CLL','CA')]
GPU=os.environ.get('BOLTZ_RESEARCH_GPU','0')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def atom_rows(path):
    block=gemmi.cif.read(str(path)).sole_block()
    tags=['label_asym_id','label_seq_id','auth_seq_id','label_comp_id','label_atom_id','type_symbol',
          'Cartn_x','Cartn_y','Cartn_z','pdbx_PDB_model_num','label_alt_id','occupancy']
    rows=[]
    for row in block.find('_atom_site.',tags):
        data={k:gemmi.cif.as_string(v) for k,v in zip(tags,row)}
        if data['pdbx_PDB_model_num']!='1' or data['label_alt_id'] not in {'.','?','','A'}:continue
        data['xyz']=[float(data[k]) for k in ['Cartn_x','Cartn_y','Cartn_z']]
        rows.append(data)
    return rows


def prepare():
    assert not (HERE/'plan.json').exists(),'Preserve existing frozen plan'
    assert subprocess.check_output(['git','status','--short'],cwd=ROOT/'upstream',text=True)==''
    refs={};inputs={}
    for pdb,metal in CASES:
        path=HERE/'raw'/(pdb+'.cif');block=gemmi.cif.read(str(path)).sole_block()
        poly=list(block.find('_entity_poly.',['entity_id','pdbx_seq_one_letter_code_can']))
        assert len(poly)==1
        full=''.join(gemmi.cif.as_string(poly[0][1]).split())
        # 1ZNF has ACE/NH2 caps represented as X in the deposited polymer sequence.
        if pdb=='1ZNF':
            assert full[0]==full[-1]=='X' and 'X' not in full[1:-1]
            mapping=list(range(2,len(full)));seq=full[1:-1]
        else:mapping=list(range(1,len(full)+1));seq=full
        assert set(seq)<=set('ACDEFGHIKLMNPQRSTVWY')
        rows=atom_rows(path);atoms={}
        ref_to_input={str(old):new+1 for new,old in enumerate(mapping)}
        for row in rows:
            if row['label_asym_id']=='A' and row['label_seq_id'] in ref_to_input:
                key=str(ref_to_input[row['label_seq_id']])+':'+row['label_atom_id']
                atoms.setdefault(key,{'xyz':row['xyz'],'element':row['type_symbol']})
        metals=[r for r in rows if r['label_comp_id']==metal]
        assert len(metals)==(1 if pdb=='1ZNF' else 4)
        donors=[]
        for ion in metals:
            near=[key for key,a in atoms.items() if a['element'] in {'N','O','S'} and np.linalg.norm(np.array(a['xyz'])-ion['xyz'])<=3.]
            assert len(near)>=3,(pdb,near)
            donors.append(near)
        refs[pdb]={'pdb':pdb,'source':'https://www.rcsb.org/structure/'+pdb,'source_sha256':digest(path),
            'sequence':seq,'input_to_reference_label_seq_id':mapping,'protein_atoms':atoms,
            'metal':metal,'metals_xyz':[r['xyz'] for r in metals],'reference_donors':donors,
            'reference_model':1,'omissions':'ACE/NH2 caps for 1ZNF; waters/ethanol for 1CLL. Same omissions in both conditions.'}
        for with_ions in [False,True]:
            name=pdb.lower()+('_ions' if with_ions else '_noions')
            sequences=[{'protein':{'id':'A','sequence':seq,'msa':'empty'}}]
            if with_ions:
                for i in range(len(metals)):sequences.append({'ligand':{'id':'Z'+str(i+1),'ccd':metal}})
            payload={'version':1,'sequences':sequences}
            target=HERE/'inputs'/(name+'.yaml');target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(yaml.safe_dump(payload,sort_keys=False))
            inputs[name]={'pdb':pdb,'with_ions':with_ions,'sha256':digest(target),'input':str(target.relative_to(ROOT))}
    save(HERE/'references.json',refs)
    jobs=[]
    for seed in [0,1,2]:
        name='seed'+str(seed);work=RUNS/name;directory=work/'inputs'/name;directory.mkdir(parents=True)
        for source in sorted((HERE/'inputs').glob('*.yaml')):shutil.copy2(source,directory/source.name)
        argv=[sys.executable,'-m','boltz.main','predict',str(directory),'--model','boltz2','--accelerator','gpu','--devices','1',
            '--cache',str(ROOT/'cache'),'--out_dir',str(work/'output'),'--recycling_steps','3','--sampling_steps','200',
            '--diffusion_samples','3','--max_parallel_samples','1','--seed',str(seed),'--num_workers','2','--preprocessing-threads','1']
        job={'job':name,'seed':seed,'command':argv,'output':str(work/'output'/('boltz_results_'+name))}
        save(work/'config.json',job);jobs.append(job)
    save(HERE/'plan.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'question':'In two fixed metal-binding protein constructs, how do explicitly supplied native ions change cofolded protein geometry?',
        'inputs':inputs,'seeds':[0,1,2],'samples_per_input_seed':3,'predicted_structures':36,
        'structure_settings':{'recycles':3,'diffusion_steps':200,'samples':3,'max_parallel_samples':1},
        'msa':'empty / single sequence','templates':False,'use_potentials':False,
        'all_samples_included':True,'metrics':['all resolved protein CA RMSD to reference model1, after proper Kabsch alignment',
            'with ions only: permutation-matched ion position RMSD after protein alignment',
            'with ions only: mean absolute deviation of reference protein-donor/ion distances'],
        'reference_donor_cutoff_A':3.,'metal_assignment':'Hungarian matching of identical ions after protein CA alignment',
        'limitations':['Two old PDB constructs, potentially in training; no held-out or generalization claim.',
            'Only single-sequence mode, no MSA/templates/force guidance.',
            'Same seed does not imply matched noise after changing ion/tensor shapes.',
            'Global calmodulin RMSD is sensitive to inter-domain motion; missing reference residues are excluded.',
            'All observations descriptive; no independent-sample p-value from correlated samples.'],
        'jobs':jobs,'runner_sha256':digest(Path(__file__)),'references_sha256':digest(HERE/'references.json'),
        'upstream_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'upstream',text=True).strip()})
    print(json.dumps({'prepared':True,'cases':{p:{'residues':len(r['sequence']),'ions':len(r['metals_xyz'])} for p,r in refs.items()},'new_structure_samples':36},indent=2))


def run():
    import pynvml
    plan=json.loads((HERE/'plan.json').read_text());assert digest(Path(__file__))==plan['runner_sha256']
    assert digest(HERE/'references.json')==plan['references_sha256']
    assert not (RUNS/'supervisor.json').exists()
    for item in plan['inputs'].values():assert digest(ROOT/item['input'])==item['sha256']
    env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
        'OPENBLAS_NUM_THREADS':'8','NUMBA_NUM_THREADS':'8','PYTHONUNBUFFERED':'1','BOLTZ_CACHE':str(ROOT/'cache'),
        'CUEQ_DEFAULT_CONFIG':'1','CUEQ_DISABLE_AOT_TUNING':'1'})
    pynvml.nvmlInit();gpu=pynvml.nvmlDeviceGetHandleByUUID(GPU) if GPU.startswith('GPU-') else pynvml.nvmlDeviceGetHandleByIndex(int(GPU))
    todo=list(plan['jobs']);active=[];done=[];start=time.monotonic();last=0
    save(RUNS/'supervisor.json',{'pid':os.getpid(),'gpu':GPU,'children':{}})
    try:
        while todo or active:
            while todo and len(active)<2:
                job=todo.pop(0);work=RUNS/job['job'];assert not list(Path(job['output']).glob('predictions/*/*_model_*.cif'))
                with (work/'stdout.log').open('wb') as log:
                    p=subprocess.Popen(job['command'],cwd=ROOT/'upstream',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active.append({'job':job,'p':p,'start':time.monotonic(),'peak':0})
                receipt=json.loads((RUNS/'supervisor.json').read_text());receipt['children'][job['job']]=p.pid;save(RUNS/'supervisor.json',receipt)
            memory={p.pid:p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu)}
            for item in list(active):
                p=item['p'];job=item['job'];item['peak']=max(item['peak'],memory.get(p.pid,0))
                if p.poll() is None:continue
                cif=list(Path(job['output']).glob('predictions/*/*_model_*.cif'))
                result={'job':job['job'],'returncode':p.returncode,'structures':len(cif),'elapsed_s':time.monotonic()-item['start'],'peak_sampled_mib':item['peak']/2**20}
                save(RUNS/job['job']/'result.json',result);done.append(result);active.remove(item);print(json.dumps(result),flush=True)
                assert p.returncode==0 and len(cif)==12,result
            if time.monotonic()-last>15:
                state={'elapsed_s':time.monotonic()-start,'completed':done,'active':[{'job':i['job']['job'],'pid':i['p'].pid,'structures':len(list(Path(i['job']['output']).glob('predictions/*/*_model_*.cif')))} for i in active], 'queued':[j['job'] for j in todo]}
                save(RUNS/'progress.json',state);print(json.dumps(state),flush=True);last=time.monotonic()
            assert time.monotonic()-start<3600,'One-hour bound'
            time.sleep(.5)
        save(RUNS/'result.json',{'completed':True,'jobs':done,'elapsed_s':time.monotonic()-start})
        analyze()
    except BaseException as error:
        save(RUNS/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
    finally:
        for item in active:
            if item['p'].poll() is None:os.killpg(item['p'].pid,signal.SIGTERM)
        pynvml.nvmlShutdown()


def analyze():
    from scipy.optimize import linear_sum_assignment
    plan=json.loads((HERE/'plan.json').read_text());refs=json.loads((HERE/'references.json').read_text());table=[]
    for job in plan['jobs']:
        out=Path(job['output'])
        for name,definition in plan['inputs'].items():
            ref=refs[definition['pdb']];atoms=ref['protein_atoms']
            ca=sorted([k for k in atoms if k.endswith(':CA')],key=lambda k:int(k.split(':')[0]))
            target=np.array([atoms[k]['xyz'] for k in ca])
            paths=sorted((out/'predictions'/name).glob(name+'_model_*.cif'));assert len(paths)==3
            for path in paths:
                rows=atom_rows(path);pred={};metals=[]
                for row in rows:
                    if row['label_asym_id']=='A':pred.setdefault(row['auth_seq_id']+':'+row['label_atom_id'],row['xyz'])
                    if row['label_comp_id']==ref['metal']:metals.append(row['xyz'])
                assert set(ca)<=set(pred)
                mobile=np.array([pred[k] for k in ca]);u,_,vt=np.linalg.svd((mobile-mobile.mean(0)).T@(target-target.mean(0)))
                rotation=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
                aligned=(mobile-mobile.mean(0))@rotation+target.mean(0)
                entry={'pdb':ref['pdb'],'case':name,'with_ions':definition['with_ions'],'seed':job['seed'],
                    'rank':int(path.stem.rsplit('_',1)[1]),'ca_count':len(ca),'ca_rmsd_A':float(np.sqrt(np.mean(np.sum((aligned-target)**2,axis=1)))),
                    'prediction_file':str(path.relative_to(ROOT))}
                if definition['with_ions']:
                    m=np.array(metals);assert len(m)==len(ref['metals_xyz'])
                    aligned_m=(m-mobile.mean(0))@rotation+target.mean(0);reference=np.array(ref['metals_xyz'])
                    a,b=linear_sum_assignment(np.sum((aligned_m[:,None,:]-reference[None,:,:])**2,axis=-1));order={int(j):int(i) for i,j in zip(a,b)}
                    entry['metal_rmsd_after_protein_alignment_A']=float(np.sqrt(np.mean(np.sum((aligned_m[a]-reference[b])**2,axis=1))))
                    diffs=[]
                    for j,donors in enumerate(ref['reference_donors']):
                        for key in donors:
                            observed=float(np.linalg.norm(np.array(pred[key])-m[order[j]]))
                            expected=float(np.linalg.norm(np.array(atoms[key]['xyz'])-reference[j]))
                            diffs.append(abs(observed-expected))
                    entry['reference_donor_distance_mae_A']=float(np.mean(diffs))
                else:assert not metals
                assert np.isfinite(entry['ca_rmsd_A']);table.append(entry)
    assert len(table)==36
    summaries={};per_seed=[]
    for pdb,_ in CASES:
        summaries[pdb]={}
        for flag,label in [(False,'without_ions'),(True,'with_ions')]:
            g=[r for r in table if r['pdb']==pdb and r['with_ions']==flag]
            d={'samples':len(g),'ca_rmsd_mean_A':float(np.mean([r['ca_rmsd_A'] for r in g])),'ca_rmsd_median_A':float(np.median([r['ca_rmsd_A'] for r in g]))}
            if flag:d['donor_distance_mae_mean_A']=float(np.mean([r['reference_donor_distance_mae_A'] for r in g]))
            summaries[pdb][label]=d
        for seed in [0,1,2]:
            mean=lambda f:float(np.mean([r['ca_rmsd_A'] for r in table if r['pdb']==pdb and r['seed']==seed and r['with_ions']==f]))
            per_seed.append({'pdb':pdb,'seed':seed,'without_ions_ca_rmsd_A':mean(False),'with_ions_ca_rmsd_A':mean(True),'delta_with_minus_without_A':mean(True)-mean(False)})
    result={'structures':36,'summary':summaries,'per_seed':per_seed,'rows':table,'limitations':plan['limitations']}
    save(HERE/'analysis.json',result)
    resources=json.loads((RUNS/'result.json').read_text())
    save(RUNS/'progress.json',{'status':'completed','structures':36,'active':[],'elapsed_s':resources['elapsed_s']})
    print(json.dumps({'summary':summaries,'per_seed':per_seed},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','analyze']);{'prepare':prepare,'run':run,'analyze':analyze}[p.parse_args().action]()
