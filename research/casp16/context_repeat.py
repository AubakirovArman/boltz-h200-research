"""Preregister and evaluate larger affinity context on the fixed 140 pairs.

The original structures and seed-1 controls are read-only. Exactly one new
context and equal-weight ensemble rules are fixed before fresh inference.
Resource measurements are descriptive, not an end-to-end speed benchmark.
"""

import argparse
import csv
import datetime
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import ensemble_repeat as shared

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE/'runs/issue-settings-csv-seed0'
CONTROL = HERE/'experiments/affinity-seed1-fixed-structures-20261103'
EXPERIMENT = HERE/'experiments/context300-seed1-20261103'
LAUNCHER = HERE/'expanded_context_cli.py'
GPU = shared.GPU
MODES = ['original', 'control', 'context300', 'matched_ensemble', 'original_ensemble']
MAX_CONCURRENT = 2


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def audit_crops():
    """Verify all 140 fixed structures before launching model inference."""
    import numpy as np
    from boltz.data.crop.affinity import AffinityCropper
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import Input, Record, StructureV2
    from expanded_context_cli import ExpandedAffinityCropper

    tokenizer = Boltz2Tokenizer()
    stock = AffinityCropper()
    enlarged = ExpandedAffinityCropper()
    rows = []
    for job in json.loads((CONTROL/'plan.json').read_text())['jobs']:
        config = json.loads((CONTROL/job['job']/'config.json').read_text())
        source = Path(config['output'])
        for name in job['ids']:
            record = Record.load(source/'processed/records'/(name+'.json'))
            structure = StructureV2.load(source/'predictions'/name/('pre_affinity_'+name+'.npz'))
            data = tokenizer.tokenize(Input(structure=structure, msa={}, record=record))
            before = stock.crop(data, max_tokens=256, max_atoms=2048)
            after = enlarged.crop(data, max_tokens=256, max_atoms=2048)
            a, b = set(before.tokens['token_idx']), set(after.tokens['token_idx'])
            assert a <= b, (name, 'Lost original context')
            ligand = set(data.tokens[data.tokens['affinity_mask']]['token_idx'])
            assert ligand and ligand <= a and ligand <= b, (name, 'Lost ligand tokens')
            # Selecting a larger prefix must preserve every existing token field.
            for token in before.tokens:
                matched = after.tokens[after.tokens['token_idx'] == token['token_idx']]
                assert len(matched) == 1 and matched[0].tobytes() == token.tobytes(), name
            old_n = int(np.count_nonzero(before.tokens['mol_type'] == 0))
            new_n = int(np.count_nonzero(after.tokens['mol_type'] == 0))
            full_n = int(np.count_nonzero(data.tokens['mol_type'] == 0))
            assert old_n < new_n <= min(300, full_n), (name, old_n, new_n, full_n)
            rows.append({'id': name, 'target': job['target'], 'full_protein_tokens': full_n,
                         'stock_protein_tokens': old_n, 'expanded_protein_tokens': new_n,
                         'stock_total_tokens': len(before.tokens), 'expanded_total_tokens': len(after.tokens),
                         'stock_atoms': int(before.tokens['atom_num'].sum()),
                         'expanded_atoms': int(after.tokens['atom_num'].sum()),
                         'ligand_tokens': len(ligand), 'all_ligand_tokens_preserved': True,
                         'stock_context_subset': True,
                         'expanded_protein_residues': [[int(t['asym_id']), int(t['res_idx'])]
                             for t in after.tokens if t['mol_type'] == 0]})
    assert len(rows) == 140 and len({r['id'] for r in rows}) == 140
    summary = {}
    for target in ['L1000', 'L3000']:
        group = [r for r in rows if r['target'] == target]
        summary[target] = {'n': len(group),
            'stock_protein_range': [min(r['stock_protein_tokens'] for r in group), max(r['stock_protein_tokens'] for r in group)],
            'expanded_protein_range': [min(r['expanded_protein_tokens'] for r in group), max(r['expanded_protein_tokens'] for r in group)],
            'expanded_total_token_range': [min(r['expanded_total_tokens'] for r in group), max(r['expanded_total_tokens'] for r in group)],
            'expanded_atom_range': [min(r['expanded_atoms'] for r in group), max(r['expanded_atoms'] for r in group)]}
    return {'scope': 'CPU-only input audit; no affinity outputs or labels used',
            'stock_limits': {'protein': 200, 'total_tokens': 256, 'atoms': 2048},
            'experimental_limits': {'protein': 300, 'total_tokens': 384, 'atoms': 3072},
            'all_140_crops_expand': True, 'all_140_ligands_preserved': True,
            'by_target': summary, 'pairs': rows}


def prepare():
    assert not EXPERIMENT.exists(), 'Preserve existing experiment'
    assert subprocess.check_output(['git','status','--short'], cwd=ROOT/'upstream', text=True) == ''
    audit = audit_crops()
    EXPERIMENT.mkdir(parents=True)
    shared.save(EXPERIMENT/'crop-audit.json', audit)
    jobs = json.loads((CONTROL/'plan.json').read_text())['jobs']
    checksums = {str(p.relative_to(ROOT)): shared.digest(p) for p in shared.baseline_files()}
    for p in CONTROL.glob('*/output/*/predictions/*/affinity_*.json'):
        checksums[str(p.relative_to(ROOT))] = shared.digest(p)
    for job in jobs:
        oldwork, work = CONTROL/job['job'], EXPERIMENT/job['job']
        old = json.loads((oldwork/'config.json').read_text())
        oldout = Path(old['output'])
        inputs = work/'inputs'/job['job']
        shutil.copytree(oldwork/'inputs'/job['job'], inputs)
        output = work/'output'/('boltz_results_'+job['job'])
        shutil.copytree(oldout/'processed', output/'processed')
        order = [r['id'] for r in json.loads((oldout/'processed/manifest.json').read_text())['records']]
        assert [p.stem for p in (output/'processed/records').glob('*.json')] == order
        for name in job['ids']:
            dest = output/'predictions'/name
            dest.mkdir(parents=True)
            source = oldout/'predictions'/name/('pre_affinity_'+name+'.npz')
            shutil.copy2(source, dest/source.name)
            assert shared.digest(source) == old['source_structure_sha256'][name]
        original_command = old['command']
        assert original_command[1:3] == ['-m','boltz.main']
        command = [sys.executable, str(LAUNCHER)] + [s.replace(str(oldwork), str(work)) for s in original_command[3:]]
        config = {**old, 'command': command, 'output': str(output),
            'phase': 'affinity_context300_on_unchanged_rank0_structures',
            'control_input_order': order, 'control_config_sha256': shared.digest(oldwork/'config.json'),
            'source_structure_sha256': {name: shared.digest(output/'predictions'/name/('pre_affinity_'+name+'.npz')) for name in job['ids']},
            'context_limits': {'protein': 300, 'total_tokens': 384, 'atoms': 3072},
            'note': 'Stock affinity model with a scoped cropper replacement; structure inputs are byte-identical to the seed1 control. Same seed/order, but changed shapes can change random-number consumption. No original output modified.'}
        shared.save(work/'config.json', config)
    shared.save(EXPERIMENT/'plan.json', {
        'created_utc': utc(), 'question': 'Does larger protein context improve frozen-model affinity on all 140 fixed complexes?',
        'jobs': jobs, 'seed': 1, 'gpu_uuid': GPU, 'max_concurrent_jobs': MAX_CONCURRENT,
        'structure_inputs': 'Byte-identical saved rank0 pre_affinity inputs used by the complete seed1 control',
        'change': 'Affinity crop only: 200/256/2048 -> 300/384/3072 protein tokens/total tokens/atoms',
        'primary_comparison': 'New context300 versus completed seed1 context200 control on all 140 pairs',
        'secondary_comparison': 'Historical original; fixed equal control/new and original/new ensembles',
        'ensemble_weights': [0.5, 0.5], 'bootstrap_replicates': 2500, 'bootstrap_seed': 20261103,
        'primary_mw': 'Exact processed manifest affinity.mw, unchanged from control',
        'secondary_mw': 'RDKit MolWt of original CSV SMILES for historical comparison',
        'no_fitting_or_label_based_selection': True, 'all_140_required_before_primary_score': True,
        'rng_limitation': 'The same CLI seed and case order do not guarantee identical noise when crop/tensor shapes change.',
        'generalization_limitation': 'Exploratory repeated use of the same benchmark; any promising configuration needs independent validation.',
        'performance_comparison': False, 'baseline_checksums': checksums,
        'runner_sha256': shared.digest(Path(__file__)), 'launcher_sha256': shared.digest(LAUNCHER),
        'helper_sha256': shared.digest(Path(shared.__file__)), 'upstream_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT/'upstream', text=True).strip(),
    })
    print(json.dumps({'prepared': str(EXPERIMENT), 'jobs': len(jobs), 'pairs': 140,
                     'crop_audit': audit['by_target'], 'concurrency': MAX_CONCURRENT}, indent=2))


def assert_frozen(plan):
    assert shared.digest(Path(__file__)) == plan['runner_sha256']
    assert shared.digest(LAUNCHER) == plan['launcher_sha256']
    assert shared.digest(Path(shared.__file__)) == plan['helper_sha256']
    assert subprocess.check_output(['git','status','--short'], cwd=ROOT/'upstream', text=True) == ''


def run():
    import pynvml
    plan = json.loads((EXPERIMENT/'plan.json').read_text())
    assert_frozen(plan)
    assert not (EXPERIMENT/'supervisor.json').exists(), 'Do not duplicate a running experiment'
    pending = list(plan['jobs'])
    active, results = [], []
    started = time.monotonic()
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByUUID(GPU)
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': GPU, 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
                'OPENBLAS_NUM_THREADS': '8', 'NUMBA_NUM_THREADS': '8', 'PYTHONUNBUFFERED': '1',
                'BOLTZ_CACHE': str(ROOT/'cache'), 'CUEQ_DEFAULT_CONFIG': '1', 'CUEQ_DISABLE_AOT_TUNING': '1'})
    supervisor = {'pid': os.getpid(), 'started_utc': utc(), 'gpu_uuid': GPU, 'children': {},
                  'phase': 'affinity_context300', 'max_concurrent_jobs': MAX_CONCURRENT}
    shared.save(EXPERIMENT/'supervisor.json', supervisor)
    last_write = 0
    try:
        while pending or active:
            while pending and len(active) < MAX_CONCURRENT:
                item = pending.pop(0)
                work = EXPERIMENT/item['job']
                config = json.loads((work/'config.json').read_text())
                assert not list(Path(config['output']).rglob('affinity_*.json'))
                with (work/'stdout.log').open('wb') as log:
                    process = subprocess.Popen(config['command'], cwd=ROOT/'upstream', env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active.append({'process': process, 'config': config, 'work': work, 'started': time.monotonic(), 'peak': 0})
                supervisor['children'][config['job']] = process.pid
                shared.save(EXPERIMENT/'supervisor.json', supervisor)
            memory = {p.pid: p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu)}
            for job in list(active):
                process, config, work = job['process'], job['config'], job['work']
                job['peak'] = max(job['peak'], memory.get(process.pid, 0))
                if process.poll() is not None:
                    found = list(Path(config['output']).rglob('affinity_*.json'))
                    ids = sorted(p.stem.removeprefix('affinity_') for p in found)
                    log = (work/'stdout.log').read_text(errors='replace')
                    result = {'job': config['job'], 'returncode': process.returncode, 'pairs': len(found),
                        'expected': len(config['ids']), 'ids_match': ids == sorted(config['ids']),
                        'elapsed_s': time.monotonic()-job['started'], 'gpu_peak_sampled_mib': job['peak']/2**20,
                        'structure_prediction_launched': 'Running structure prediction for' in log,
                        'expanded_crop_launcher_logged': 'EXPERIMENTAL_AFFINITY_CONTEXT protein=300 tokens=384 atoms=3072' in log,
                        'phase': 'affinity_context300', 'monitor_interval_s': 0.5}
                    shared.save(work/'result.json', result)
                    results.append(result)
                    active.remove(job)
                    print(json.dumps(result), flush=True)
                    assert result['returncode'] == 0 and result['ids_match'] and result['expanded_crop_launcher_logged']
                    assert not result['structure_prediction_launched']
            if time.monotonic()-last_write >= 15:
                progress = {'status': 'running', 'elapsed_s': time.monotonic()-started, 'completed_jobs': results,
                    'queued': [j['job'] for j in pending],
                    'active': [{'job': j['config']['job'], 'pid': j['process'].pid,
                        'pairs': len(list(Path(j['config']['output']).rglob('affinity_*.json'))),
                        'expected': len(j['config']['ids']), 'gpu_peak_mib': j['peak']/2**20} for j in active]}
                shared.save(EXPERIMENT/'progress.json', progress)
                print(json.dumps(progress), flush=True)
                last_write = time.monotonic()
            if time.monotonic()-started > 10800:
                raise TimeoutError('Three-hour safety bound for context experiment')
            time.sleep(0.5)
        assert_frozen(plan)
        for name, sha in plan['baseline_checksums'].items():
            assert shared.digest(ROOT/name) == sha, name
        shared.save(EXPERIMENT/'result.json', {'completed': True, 'jobs': results,
            'elapsed_s': time.monotonic()-started, 'original_baseline_unchanged': True,
            'performance_comparison': False, 'phase': 'affinity_context300'})
        analyze()
    except BaseException as error:
        shared.save(EXPERIMENT/'failure.json', {'type': type(error).__name__, 'message': str(error), 'utc': utc()})
        raise
    finally:
        for job in active:
            if job['process'].poll() is None:
                os.killpg(job['process'].pid, signal.SIGTERM)
        pynvml.nvmlShutdown()


def predictions():
    values = {}
    for job in json.loads((EXPERIMENT/'plan.json').read_text())['jobs']:
        config = json.loads((EXPERIMENT/job['job']/'config.json').read_text())
        output = Path(config['output'])
        order = [r['id'] for r in json.loads((output/'processed/manifest.json').read_text())['records']]
        assert order == config['control_input_order']
        found = list(output.rglob('affinity_*.json'))
        assert {p.stem.removeprefix('affinity_') for p in found} == set(job['ids'])
        for path in found:
            name = path.stem.removeprefix('affinity_')
            assert name not in values, name
            data = json.loads(path.read_text())
            assert all(math.isfinite(float(v)) for v in data.values())
            raw = float(data['affinity_pred_value'])
            assert abs(raw-.5*(data['affinity_pred_value1']+data['affinity_pred_value2'])) < 1e-5
            values[name] = raw
    assert len(values) == 140
    return values


def analyze():
    import numpy as np
    from scipy.stats import pearsonr
    fresh = predictions()
    original = list(csv.DictReader((ROOT/'reports/casp16-issue636/predictions-and-labels.csv').open()))
    control = {r['id']: r for r in csv.DictReader((CONTROL/'all-predictions.csv').open())}
    assert len(original) == 140 and set(fresh) == set(control) == {r['id'] for r in original}
    table = []
    for row in original:
        name = row['id']
        old, ctl, new = float(row['raw']), float(control[name]['seed1_raw']), fresh[name]
        table.append({**row, 'processed_mw': float(control[name]['processed_mw']),
            'original': old, 'control': ctl, 'context300': new,
            'matched_ensemble': .5*(ctl+new), 'original_ensemble': .5*(old+new)})
    arrays, per_target = {}, {}
    for target in ['L1000','L3000']:
        group = [r for r in table if r['target'] == target]
        y = np.array([float(r['true_log10_ic50_um']) for r in group])
        pred = {}
        for mode in MODES:
            pred[mode+'_raw'] = np.array([r[mode] for r in group])
            for label, key in [('processed','processed_mw'), ('csv','mw')]:
                pred[mode+'_'+label] = np.array([1.03525938*r[mode]-.59992683*float(r[key])**.3+2.83288489 for r in group])
        per_target[target] = {'n': len(group), 'r': {k: float(pearsonr(p,y).statistic) for k,p in pred.items()}}
        arrays[target] = y, pred
    aggregate = {k: {'weighted': sum(g['n']*g['r'][k] for g in per_target.values())/140,
        'uniform': sum(g['r'][k] for g in per_target.values())/2} for k in per_target['L1000']['r']}
    rng = np.random.default_rng(20261103)
    draws = []
    for _ in range(2500):
        totals = {m: 0. for m in MODES}
        for y, pred in arrays.values():
            idx = rng.integers(0,len(y),len(y))
            for m in MODES:
                totals[m] += len(y)/140*float(np.corrcoef(pred[m+'_processed'][idx],y[idx])[0,1])
        draws.append([totals[m] for m in MODES])
    draws = np.array(draws)
    def delta(a,b):
        return np.quantile(draws[:,MODES.index(a)]-draws[:,MODES.index(b)],[.025,.975]).tolist()
    summary = {'pairs': 140, 'phase': 'Affinity context300 on unchanged original structures',
        'per_target': per_target, 'aggregate': aggregate, 'bootstrap': {'resamples': 2500, 'seed': 20261103,
            'context300_vs_control_delta_ci95': delta('context300','control'),
            'context300_vs_original_delta_ci95': delta('context300','original'),
            'matched_ensemble_vs_control_delta_ci95': delta('matched_ensemble','control'),
            'original_ensemble_vs_original_delta_ci95': delta('original_ensemble','original')},
        'no_fitting_or_label_based_selection': True,
        'context300_exceeds_original': aggregate['context300_processed']['weighted'] > aggregate['original_processed']['weighted'],
        'context300_reaches_0_8': aggregate['context300_processed']['weighted'] >= .8}
    shared.save(EXPERIMENT/'analysis.json', summary)
    with (EXPERIMENT/'all-predictions.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    verify()
    resources = json.loads((EXPERIMENT/'result.json').read_text())
    lines = ['# Расширение контекста белка при расчёте affinity', '',
        'Все 140 пар. Сохранённые исходные структуры побитно те же. Меняется только ограничение affinity crop: белок 200→300 токенов, весь комплекс 256→384, атомы 2048→3072. Веса, химия, dtype и параметры диффузии прежние.', '',
        '| Вариант | Взвешенный R, MW CLI | Взвешенный R, MW CSV |', '|---|---:|---:|']
    for mode, label in zip(MODES,['Исторический исходный расчёт','Контроль context200, seed1','Контекст300, seed1','Среднее контроля и контекст300','Среднее исходного расчёта и контекст300']):
        lines.append(f"| {label} | {aggregate[mode+'_processed']['weighted']:.6f} | {aggregate[mode+'_csv']['weighted']:.6f} |")
    lines += ['', 'Парные 95% bootstrap-интервалы изменения (2500 повторных выборок на CPU):',
        json.dumps(summary['bootstrap'],ensure_ascii=False,indent=2), '',
        'Это исследовательский эксперимент на тех же двух белках, без обучения по меткам. Он не является независимой проверкой переносимости или замером полного ускорения. Одинаковые seed и порядок комплексов сохраняются, однако новые формы тензоров могут менять потребление случайных чисел.', '',
        f"Всего {len(resources['jobs'])} новых CLI-процесса, 140 новых affinity-расчётов. Общее время эксперимента {resources['elapsed_s']:.1f} с. Не более двух наших процессов одновременно на H200 №2.", '',
        '| Процесс | Пар | Время процесса, с | Пик процесса NVML, MiB |', '|---|---:|---:|---:|']
    for job in resources['jobs']:
        lines.append(f"| {job['job']} | {job['pairs']} | {job['elapsed_s']:.2f} | {job['gpu_peak_sampled_mib']:.0f} |")
    lines += ['', 'Пики получены опросом NVML каждые 0,5 с и могут пропускать короткие всплески. Чужие процессы не включены. Контроль входов и независимый пересчёт Pearson — исследовательские проверки, не официальный набор тестов upstream.']
    (EXPERIMENT/'REPORT_RU.md').write_text('\n'.join(lines)+'\n')
    shared.save(EXPERIMENT/'progress.json', {'status': 'completed', 'predictions': 140, 'active': [],
        'queued': [], 'completed_jobs': resources['jobs'], 'elapsed_s': resources['elapsed_s'], 'analysis_complete': True})
    print(json.dumps(summary,indent=2),flush=True)


def verify():
    reference = json.loads((EXPERIMENT/'analysis.json').read_text())
    plan = json.loads((EXPERIMENT/'plan.json').read_text())
    assert_frozen(plan)
    fresh = predictions()
    rows = list(csv.DictReader((EXPERIMENT/'all-predictions.csv').open()))
    assert len(rows) == len({r['id'] for r in rows}) == 140
    assert all(float(r['context300']) == fresh[r['id']] for r in rows)
    independent = {}
    for target in ['L1000','L3000']:
        group = [r for r in rows if r['target'] == target]
        truth = [math.log10(float(r['true_ic50_um'])) for r in group]
        manifest = json.loads((BASE/target/'output'/('boltz_results_'+target)/'processed/manifest.json').read_text())
        mw = {r['id']:r['affinity']['mw'] for r in manifest['records']}
        assert all(float(r['processed_mw']) == mw[r['id']] for r in group)
        values = {}
        for mode in MODES:
            raw = [float(r[mode]) for r in group]
            values[mode+'_raw'] = raw
            for kind,key in [('processed','processed_mw'),('csv','mw')]:
                values[mode+'_'+kind] = [1.03525938*float(r[mode])-.59992683*float(r[key])**.3+2.83288489 for r in group]
        independent[target] = {'n':len(group), 'r': {k:statistics.correlation(v,truth) for k,v in values.items()}}
    for key in independent['L1000']['r']:
        r = sum(g['n']*g['r'][key] for g in independent.values())/140
        assert abs(r-reference['aggregate'][key]['weighted']) < 1e-12, key
    preserved = 0
    for name, sha in plan['baseline_checksums'].items():
        assert shared.digest(ROOT/name) == sha, name
        preserved += 1
    for job in plan['jobs']:
        config = json.loads((EXPERIMENT/job['job']/'config.json').read_text())
        output = Path(config['output'])
        old = json.loads((CONTROL/job['job']/'config.json').read_text())
        assert shared.digest(output/'processed/manifest.json') == shared.digest(Path(old['output'])/'processed/manifest.json')
        for name, sha in config['source_structure_sha256'].items():
            assert shared.digest(output/'predictions'/name/('pre_affinity_'+name+'.npz')) == sha
        result = json.loads((EXPERIMENT/job['job']/'result.json').read_text())
        assert result['returncode'] == 0 and result['ids_match'] and not result['structure_prediction_launched']
        assert result['expanded_crop_launcher_logged']
    shared.save(EXPERIMENT/'independent-verification.json', {'passed': True, 'unique_pairs': 140,
        'finite_new_predictions': 140, 'preserved_reference_files': preserved,
        'byte_identical_structure_inputs': 140, 'cli_jobs': len(plan['jobs']),
        'independent_pearson_tolerance': 1e-12, 'independent_calculations': independent,
        'scope': 'Research input/ID/arithmetic checks, not official upstream tests.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','run','analyze','verify'])
    {'prepare':prepare, 'run':run, 'analyze':analyze, 'verify':verify}[parser.parse_args().action]()
