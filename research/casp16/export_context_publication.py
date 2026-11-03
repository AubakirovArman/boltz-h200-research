"""Export a whitelisted, checksummed context experiment to the public checkout."""
import argparse,datetime,gzip,hashlib,io,json,shutil,subprocess,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'research/casp16'
BASE=HERE/'runs/issue-settings-csv-seed0'
EXPERIMENT=HERE/'experiments/context300-seed1-20261103'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('destination',type=Path)
    dest=parser.parse_args().destination.resolve()
    assert (dest/'.git').exists() and dest!=ROOT
    paths=[str(p.relative_to(ROOT)) for p in HERE.glob('*.py')]
    paths+=subprocess.check_output(['git','ls-files','research/casp16/experiments'],cwd=ROOT,text=True).splitlines()
    paths+=[str(p.relative_to(ROOT)) for p in (HERE/'experiments').glob('*/**/output/**/affinity_*.json')]
    paths=sorted(set(paths))
    for name in paths:
        source=ROOT/name;target=dest/name
        assert source.is_file() and source.suffix in {'.py','.json','.csv','.md','.yaml'}
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    assets=dest/'data/casp16';assets.mkdir(parents=True,exist_ok=True)
    archive=assets/'context_inputs.tar.gz';entries={};stored={};logical_bytes=0
    with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0,compresslevel=6) as zipped,tarfile.open(fileobj=zipped,mode='w|') as tar:
        for target in ['L1000','L3000']:
            source=BASE/target/'output'/('boltz_results_'+target)
            files=list((source/'processed').rglob('*'))+list(source.glob('predictions/*/pre_affinity_*.npz'))
            for path in sorted(p for p in files if p.is_file()):
                name=target+'/'+str(path.relative_to(source));sha=digest(path)
                entry=tarfile.TarInfo(name);entry.mode=0o644;entry.mtime=0;entry.uid=entry.gid=0
                entries[name]={'sha256':sha,'bytes':path.stat().st_size};logical_bytes+=path.stat().st_size
                if sha in stored:
                    entry.type=tarfile.LNKTYPE;entry.linkname=stored[sha];tar.addfile(entry)
                else:
                    stored[sha]=name;entry.size=path.stat().st_size
                    with path.open('rb') as content:tar.addfile(entry,content)
    plan=json.loads((EXPERIMENT/'plan.json').read_text());jobs=[]
    for job in plan['jobs']:
        config=json.loads((EXPERIMENT/job['job']/'config.json').read_text())
        jobs.append({**job,'record_order':config['control_input_order']})
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'archive':'data/casp16/context_inputs.tar.gz','archive_sha256':digest(archive),
        'archive_bytes':archive.stat().st_size,'logical_bytes':logical_bytes,'unique_file_contents':len(stored),
        'scope':'Original processed inputs and all 140 rank0 pre_affinity structures. No model weights. Duplicate input bytes use in-archive hardlinks.',
        'source_research_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'upstream_commit':plan['upstream_commit'],'jobs':jobs,'files':entries}
    (assets/'context_inputs_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'exported_text_files':len(paths),'input_files':len(entries),'archive_bytes':archive.stat().st_size,'unique_contents':len(stored),'jobs':len(jobs)},indent=2))


if __name__=='__main__':main()
