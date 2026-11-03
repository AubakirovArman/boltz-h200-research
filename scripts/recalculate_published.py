"""Recalculate published headline results from the CSVs, without third-party packages."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / 'reports/casp16-issue636/predictions-and-labels.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 140 or len({r['id'] for r in rows}) != 140:
        raise ValueError('Expected all 140 unique prediction IDs')
    groups = defaultdict(list)
    for row in rows:
        groups[row['target']].append(row)
    if {target: len(items) for target, items in groups.items()} != {'L1000': 17, 'L3000': 123}:
        raise ValueError('Unexpected per-target coverage')
    per_target = {}
    for target, items in groups.items():
        labels = [float(r['true_log10_ic50_um']) for r in items]
        raw = [float(r['raw']) for r in items]
        corrected = [1.03525938 * float(r['raw']) - 0.59992683 * float(r['mw']) ** 0.3 + 2.83288489 for r in items]
        per_target[target] = {'n': len(items), 'raw': statistics.correlation(raw, labels),
                              'mw_corrected': statistics.correlation(corrected, labels)}
    aggregate = {}
    for field in ['raw', 'mw_corrected']:
        aggregate[field] = {
            'uniform_target_mean': statistics.mean(r[field] for r in per_target.values()),
            'compound_weighted_target_mean': sum(r['n'] * r[field] for r in per_target.values()) / 140,
        }
    with (ROOT / 'reports/benchmark_results.csv').open() as stream:
        times = {r['label']: float(r['end_to_end_s']) for r in csv.DictReader(stream)}
    stock = statistics.median(times[k] for k in ['b1-l3001-repeat1', 'b1-l3001-repeat2', 'b1-l3001-control', 'b1-l3001-final-control'])
    optimized = statistics.median(times[k] for k in ['combined-l3001-first', 'combined-l3001-repeat1', 'combined-l3001-repeat2'])
    print(json.dumps({'pairs': len(rows), 'per_target': per_target, 'aggregate': aggregate,
                      'performance': {'stock_repeat_median_s': stock, 'optimized_median_s': optimized,
                                      'speedup': stock / optimized,
                                      'time_reduction_percent': 100 * (1 - optimized / stock)}}, indent=2))


if __name__ == '__main__':
    main()
