"""Small analytic checks for the independent correlation scorer, not Boltz tests."""

import tempfile
from pathlib import Path

import numpy as np

from score import correlation, mean_correlation


def main():
    x = np.arange(7, dtype=float)
    y = np.array([0, 1, 4, 2, 3, 6, 5], dtype=float)
    r = correlation(x, y)
    assert abs(correlation(1.364 * x - 8.184, 1.364 * y - 8.184) - r) < 1e-14
    assert abs(correlation(x + 19, y) - r) < 1e-14
    groups = {
        "small": [{"raw": a, "true_log10_ic50_um": b} for a, b in zip(x, y)],
        "large": [{"raw": a, "true_log10_ic50_um": b} for _ in range(3) for a, b in zip(x, x)],
    }
    assert abs(mean_correlation(groups, "raw", False) - (r + 1) / 2) < 1e-14
    assert abs(mean_correlation(groups, "raw", True) - (7 * r + 21) / 28) < 1e-14
    for values in groups.values():
        values.reverse()
    assert abs(mean_correlation(groups, "raw", True) - (7 * r + 21) / 28) < 1e-14
    assert correlation([1, 1, 1], [1, 2, 3]) is None
    assert abs((17 * .44 + 123 * .59) / 140 - .5717857142857142) < 1e-14
    print("Independent scorer analytic checks passed: aggregation, ordering, affine invariance and degenerate inputs.")


if __name__ == "__main__":
    main()
