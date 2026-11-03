"""Portable preparation, conditional affinity rerun, and dependency-free scoring.

No GPU is used by prepare, score, or published-score. Inference is explicit via
run, uses the pinned Boltz source, and preserves all 140 cases and case order.
"""
import argparse,csv,hashlib,json,math,os,re,shutil,signal,statistics,subprocess,sys,tarfile,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'data/casp16'
PUBLISHED=ROOT/'research/casp16/experiments/context300-seed1-20261103'
PIN='b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*2**20),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def prepare(args):
    work=args.workdir.resolve()
    if work.exists():raise FileExistsError('Use a fresh workdir; existing results are preserved: '+str(work))
    manifest=json.loads((ASSETS/'context_inputs_manifest.json').read_text())
    archive=ROOT/manifest['archive']
    assert digest(archive)==manifest['archive_sha256'],'Input archive checksum mismatch'
    work.mkdir(parents=True)
    inputs=work/'frozen_inputs';inputs.mkdir()
    with tarfile.open(archive) as tar:tar.extractall(inputs,filter='data')
    for name,entry in manifest['files'].items():
        assert digest(inputs/name)==entry['sha256'],name
    for job in manifest['jobs']:
        source=inputs/job['target'];dest=work/job['job']
        output=dest/'output'/('boltz_results_'+job['job'])
        shutil.copytree(source/'processed',output/'processed',ignore=shutil.ignore_patterns('records'))
        records=output/'processed/records';records.mkdir()
        for name in job['record_order']:
            shutil.copy2(source/'processed/records'/(name+'.json'),records/(name+'.json'))
        save(dest/'case-order.json',job['record_order'])
        original=json.loads((source/'processed/manifest.json').read_text())['records']
        selected=[r for r in original if r['id'] in job['ids']]
        assert [r['id'] for r in selected]==job['record_order']
        (output/'processed/manifest.json').write_text(json.dumps({'records':selected}))
        yamls=dest/'inputs'/job['job'];yamls.mkdir(parents=True)
        for name in job['ids']:
            pred=output/'predictions'/name;pred.mkdir(parents=True)
            shutil.copy2(source/'predictions'/name/('pre_affinity_'+name+'.npz'),pred)
            text=(ROOT/'research/casp16/inputs'/job['target']/(name+'.yaml')).read_text()
            msa=ROOT/'data/msa'/(job['target']+'.a3m')
            text,n=re.subn(r'(?m)^(\s*msa:)\s*.*$',lambda m:m[1]+' '+json.dumps(str(msa)),text)
            assert n==1,name
            (yamls/(name+'.yaml')).write_text(text)
    # Historical filenames/order are preserved; machine-specific paths are new.
    save(work/'plan.json',{'context':args.context,'seed':args.seed,'jobs':manifest['jobs'],
        'upstream_commit':PIN,'archive_sha256':manifest['archive_sha256'],
        'scope':'Conditional affinity on the exact published rank0 structures; no structure resampling.'})
    print(json.dumps({'prepared':str(work),'pairs':140,'context':args.context,'verified_input_files':len(manifest['files'])}))


def command(work,job,plan):
    prefix=[sys.executable,str(ROOT/'scripts/portable_affinity_cli.py')]
    dest=work/job['job']
    return prefix+['predict',str(dest/'inputs'/job['job']),'--model','boltz2','--accelerator','gpu','--devices','1',
        '--cache',str(ROOT/'cache'),'--out_dir',str(dest/'output'),'--recycling_steps','10',
        '--sampling_steps','200','--diffusion_samples','5','--sampling_steps_affinity','200',
        '--diffusion_samples_affinity','5','--seed',str(plan['seed']),'--num_workers','2','--preprocessing-threads','1']


def run(args):
    work=args.workdir.resolve();plan=json.loads((work/'plan.json').read_text())
    upstream=ROOT/'upstream'
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()
    assert actual==PIN,'Check out the documented upstream commit'
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=upstream,text=True)
    assert not list(work.glob('*/output/*/predictions/*/affinity_*.json')),'Use a fresh workdir for a new RNG repeat'
    for entry in json.loads((ROOT/'manifests/model_checksums.json').read_text()):
        if entry['file'].endswith('.ckpt'):
            assert digest(ROOT/entry['file'])==entry['sha256'],entry['file']
    assert (ROOT/'cache/mols').is_dir(),'Prepare checkpoint/CCD assets first'
    env=os.environ.copy();env.setdefault('CUDA_VISIBLE_DEVICES','0')
    assert env['CUDA_VISIBLE_DEVICES'] and ',' not in env['CUDA_VISIBLE_DEVICES'],'Select exactly one physical GPU'
    env.update({'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8',
        'NUMBA_NUM_THREADS':'8','PYTHONUNBUFFERED':'1','BOLTZ_CACHE':str(ROOT/'cache'),
        'CUEQ_DEFAULT_CONFIG':'1','CUEQ_DISABLE_AOT_TUNING':'1'})
    todo=list(plan['jobs']);active=[];done=[];started=time.monotonic()
    try:
        while todo or active:
            while todo and len(active)<args.concurrency:
                job=todo.pop(0);dest=work/job['job'];argv=command(work,job,plan)
                save(dest/'command.json',{'argv':argv,'gpu':env['CUDA_VISIBLE_DEVICES']})
                with (dest/'stdout.log').open('wb') as log:
                    child_env=dict(env,BOLTZ_REPRO_CONTEXT=str(plan['context']),BOLTZ_REPRO_ORDER=str(dest/'case-order.json'))
                    process=subprocess.Popen(argv,cwd=upstream,env=child_env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active.append((job,process,time.monotonic()))
                print('Started '+job['job'],flush=True)
            for item in list(active):
                job,process,begin=item
                if process.poll() is None:continue
                result={'job':job['job'],'returncode':process.returncode,'elapsed_s':time.monotonic()-begin}
                save(work/job['job']/'result.json',result);active.remove(item)
                if process.returncode:raise RuntimeError('CLI failed; see '+str(work/job['job']/'stdout.log'))
                text=(work/job['job']/'stdout.log').read_text(errors='replace')
                assert 'Running structure prediction for' not in text,'Expected conditional affinity only'
                if plan['context']==300:assert 'EXPERIMENTAL_AFFINITY_CONTEXT protein=300' in text
                done.append(result);print(json.dumps(result),flush=True)
            time.sleep(.5)
        save(work/'result.json',{'completed':True,'elapsed_s':time.monotonic()-started,'jobs':done})
        score(work,plan['jobs'])
    finally:
        for _,process,_ in active:
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)


def read_predictions(work,jobs):
    fresh={}
    for job in jobs:
        directory=work/job['job']/'output'/('boltz_results_'+job['job'])/'predictions'
        paths=list(directory.glob('*/affinity_*.json'))
        assert {p.stem.removeprefix('affinity_') for p in paths}==set(job['ids']),job['job']
        for path in paths:
            name=path.stem.removeprefix('affinity_');assert name not in fresh,name
            d=json.loads(path.read_text());assert all(math.isfinite(float(v)) for v in d.values())
            assert abs(d['affinity_pred_value']-.5*(d['affinity_pred_value1']+d['affinity_pred_value2']))<1e-5
            fresh[name]=float(d['affinity_pred_value'])
    assert len(fresh)==140
    return fresh


def score(work,jobs,published=False):
    fresh=read_predictions(work,jobs)
    rows=list(csv.DictReader((PUBLISHED/'all-predictions.csv').open()))
    assert len(rows)==len({r['id'] for r in rows})==140 and {r['id'] for r in rows}==set(fresh)
    target={}
    for name in ['L1000','L3000']:
        group=[r for r in rows if r['target']==name]
        truth=[math.log10(float(r['true_ic50_um'])) for r in group]
        variants={'raw':[fresh[r['id']] for r in group]}
        for label,mass in [('processed','processed_mw'),('csv','mw')]:
            variants[label]=[1.03525938*fresh[r['id']]-.59992683*float(r[mass])**.3+2.83288489 for r in group]
        target[name]={'n':len(group),'r':{key:statistics.correlation(p,truth) for key,p in variants.items()}}
    aggregate={key:{'weighted':sum(g['n']*g['r'][key] for g in target.values())/140,
                    'uniform':sum(g['r'][key] for g in target.values())/2} for key in ['raw','processed','csv']}
    if published:
        reference=json.loads((PUBLISHED/'analysis.json').read_text())
        for key in aggregate:
            for metric in ['uniform','weighted']:
                assert abs(aggregate[key][metric]-reference['aggregate']['context300_'+key][metric])<1e-12
        assert all(fresh[r['id']]==float(r['context300']) for r in rows)
    result={'pairs':140,'per_target':target,'aggregate':aggregate,
        'scope':'Research arithmetic checks; no official upstream test suite is claimed.'}
    if not published:save(work/'score.json',result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','score','published-score'])
    parser.add_argument('--workdir',type=Path,default=ROOT/'local-context300')
    parser.add_argument('--context',type=int,choices=[200,300],default=300)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--concurrency',type=int,choices=[1,2],default=2)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='run':run(args)
    elif args.action=='published-score':
        jobs=json.loads((ASSETS/'context_inputs_manifest.json').read_text())['jobs'];score(PUBLISHED,jobs,True)
    else:
        work=args.workdir.resolve();score(work,json.loads((work/'plan.json').read_text())['jobs'])
