"""Restore recorded case order on any filesystem, then run the pinned model.

The optional 300-residue crop is the original published experiment class.
Ordering only controls which case consumes each part of the seeded RNG stream.
"""
import json,os,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/casp16'))


def restore_order(out_dir,order):
    from boltz.data.types import Manifest
    path=Path(out_dir)/'processed/manifest.json'
    manifest=Manifest.load(path)
    records={r.id:r for r in manifest.records}
    assert len(records)==len(manifest.records)==len(order) and set(records)==set(order)
    manifest=Manifest(records=[records[name] for name in order])
    manifest.dump(path)
    return manifest


def install(context,order):
    import boltz.main as main
    if context==300:
        from expanded_context_cli import ExpandedAffinityCropper
        import boltz.data.module.inferencev2 as inference
        inference.AffinityCropper=ExpandedAffinityCropper
        print('EXPERIMENTAL_AFFINITY_CONTEXT protein=300 tokens=384 atoms=3072',flush=True)
    else:
        assert context==200
    original=main.process_inputs
    def process_inputs(*args,**kwargs):
        result=original(*args,**kwargs)
        restore_order(kwargs['out_dir'],order)
        print('Restored recorded order for '+str(len(order))+' cases',flush=True)
        return result
    main.process_inputs=process_inputs
    return main


if __name__=='__main__':
    context=int(os.environ['BOLTZ_REPRO_CONTEXT'])
    order=json.loads(Path(os.environ['BOLTZ_REPRO_ORDER']).read_text())
    install(context,order).cli()
