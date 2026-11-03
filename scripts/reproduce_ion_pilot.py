"""Run or independently rescore the small Boltz ion cofolding pilot."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / 'research' / 'ions'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['published-score', 'prepare', 'run', 'score'])
    parser.add_argument('--workdir', default='local-ion-pilot')
    args = parser.parse_args()
    if args.action == 'published-score':
        import json
        print(json.dumps(load('ion_verify', PUBLISHED / 'verify.py').verify(PUBLISHED), indent=2))
        return
    work = (ROOT / args.workdir).resolve()
    if not work.is_relative_to(ROOT) or work == PUBLISHED:
        parser.error('Use a separate work directory inside this repository.')
    study = load('ion_study', PUBLISHED / 'study.py')
    study.HERE, study.RUNS = work, work / 'runs'
    if args.action == 'prepare':
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / 'upstream', text=True).strip()
        assert commit == 'b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc', commit
        assert not (work / 'plan.json').exists(), 'Preserve existing plan and outputs.'
        (work / 'raw').mkdir(parents=True, exist_ok=True)
        for pdb, _ in study.CASES:
            shutil.copy2(PUBLISHED / 'raw' / (pdb + '.cif'), work / 'raw' / (pdb + '.cif'))
        study.prepare()
    elif args.action == 'run':
        study.run()
    else:
        study.analyze()


if __name__ == '__main__':
    main()
