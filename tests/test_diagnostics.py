"""Tests for scale-free item diagnostics.

The claims being checked:
  1. the item-total correlation does not move when theta is rescaled, which is
     exactly the failure that motivated replacing `a_i`;
  2. planted zero-discrimination questions land at the bottom of the ranking;
  3. flagging on the interval's upper bound does not fire on questions that are
     merely under-answered.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livebench_irt.diagnostics import (  # noqa: E402
    bootstrap_item_diagnostics,
    item_total_correlation,
)
from livebench_irt.irt import fit_2pl  # noqa: E402


def simulate(n_models=120, n_items=200, seed=0, n_dead=10):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n_models)
    a = np.exp(rng.normal(0.0, 0.35, n_items))
    b = rng.normal(0, 1.2, n_items)
    a[:n_dead] = 0.02
    p = expit(a[None, :] * (theta[:, None] - b[None, :]))
    Y = (rng.random(p.shape) < p).astype(float)
    return Y, theta, a, b


def test_invariant_to_theta_scale():
    # The whole point: rescaling ability must not change the diagnostic.
    Y, theta, a, b = simulate()
    mask = np.ones_like(Y, bool)
    r1 = item_total_correlation(Y, mask, theta)
    r2 = item_total_correlation(Y, mask, 7.3 * theta - 4.1)
    d = np.nanmax(np.abs(r1 - r2))
    print(f"max change under affine rescaling of theta: {d:.2e}")
    assert d < 1e-9


def test_ranks_dead_items_last():
    Y, theta, a, b = simulate(n_dead=10)
    mask = np.ones_like(Y, bool)
    fit = fit_2pl(Y)
    r = item_total_correlation(Y, mask, fit.theta)
    bottom = set(np.argsort(np.nan_to_num(r, nan=np.inf))[:10])
    hits = len(bottom & set(range(10)))
    print(f"planted dead items in bottom 10: {hits}/10")
    assert hits >= 8


def test_upper_bound_protects_thin_items():
    # A question answered by few models has a wide interval, so it must not be
    # flagged however low its point estimate happens to land. Flagging on the
    # point estimate alone would call it a bad question on 25 observations.
    Y, theta, a, b = simulate()
    mask = np.ones_like(Y, bool)
    thin = 26  # a question with a balanced pass rate, so r is estimable
    mask[25:, thin] = False  # answered by 25 of 120 models
    fit = fit_2pl(np.where(mask, Y, np.nan))
    diag = bootstrap_item_diagnostics(Y, mask, fit.theta, n_boot=200)

    flagged = {int(row[0]) for row in diag.uninformative(threshold=0.2)}
    width_thin = diag.hi[thin] - diag.lo[thin]
    width_typical = np.nanmedian(diag.hi - diag.lo)
    print(f"thin question: r={diag.r[thin]:.3f} "
          f"interval [{diag.lo[thin]:.3f}, {diag.hi[thin]:.3f}] width {width_thin:.3f}")
    print(f"typical interval width: {width_typical:.3f}")
    assert width_thin > 1.5 * width_typical
    assert thin not in flagged

    dead_flagged = len(flagged & set(range(10)))
    print(f"planted dead items flagged: {dead_flagged}/10")
    assert dead_flagged >= 6


if __name__ == "__main__":
    test_invariant_to_theta_scale()
    test_ranks_dead_items_last()
    test_upper_bound_protects_thin_items()
    print("all passed")
