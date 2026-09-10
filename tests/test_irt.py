"""Recovery test: simulate from a known 2PL, check the fitter finds it back.

This is the only test that matters at v0. If the fitter cannot recover
parameters it generated itself, nothing it says about LiveBench means anything.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livebench_irt.irt import fit_2pl  # noqa: E402


def simulate(n_models=60, n_items=400, seed=0, n_dead=10):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n_models)
    a = np.exp(rng.normal(0.0, 0.35, n_items))
    b = rng.normal(0, 1.2, n_items)
    a[:n_dead] = 0.02  # planted "dead" items: no discrimination
    p = expit(a[None, :] * (theta[:, None] - b[None, :]))
    Y = (rng.random(p.shape) < p).astype(float)
    return Y, theta, a, b


def test_recovers_ability_ranking():
    Y, theta, a, b = simulate()
    fit = fit_2pl(Y)
    r = np.corrcoef(fit.theta, (theta - theta.mean()) / theta.std())[0, 1]
    print(f"ability correlation: {r:.3f}")
    assert r > 0.95


def test_recovers_difficulty():
    # b_i is only identified where a_i is bounded away from 0: if an item
    # discriminates nothing, its location is not estimable. So the planted dead
    # items are excluded from this check by construction, not by convenience.
    #
    # Item parameters are estimated from one column at a time, so their
    # accuracy is governed by the NUMBER OF MODELS, not the number of items.
    # Joint MLE here is the classic incidental-parameters setting: measured
    # recovery of b (live items only) is
    #     60 models -> r = 0.81,  120 -> 0.94,  300 -> 0.96
    # This is a known limitation of joint MLE, not a bug, and it is the first
    # thing to fix after v0 (marginal MLE, integrating theta out).
    Y, theta, a, b = simulate(n_models=150)
    fit = fit_2pl(Y)
    live = a > 0.2
    r = np.corrcoef(fit.b[live], b[live])[0, 1]
    print(f"difficulty correlation (live items only): {r:.3f}")
    assert r > 0.90


def test_finds_planted_dead_items():
    # A fixed cutoff on `a` is used here and nowhere else. Inside a simulation
    # the true parameters are known, so the cutoff is checking that the fitter
    # recovers them. On real data the same cutoff is not usable -- `a` is not
    # comparable across fits -- which is why the shipped diagnostic is the
    # item-total correlation in diagnostics.py instead.
    Y, theta, a, b = simulate(n_dead=10)
    fit = fit_2pl(Y)
    flagged = set(np.where(fit.a < 0.15)[0])
    hits = len(flagged & set(range(10)))
    print(f"planted dead items recovered: {hits}/10, false positives: {len(flagged) - hits}")
    assert hits >= 8


def test_handles_missing_entries():
    Y, theta, a, b = simulate()
    rng = np.random.default_rng(1)
    Y[rng.random(Y.shape) < 0.3] = np.nan  # 30% of models never saw the item
    fit = fit_2pl(Y)
    r = np.corrcoef(fit.theta, (theta - theta.mean()) / theta.std())[0, 1]
    print(f"ability correlation with 30% missing: {r:.3f}")
    assert r > 0.90


def test_flags_separated_models():
    # A model that gets everything wrong has no finite MLE for ability. It must
    # be flagged, kept finite, and kept out of the standardisation so that it
    # does not rescale every other model.
    Y, theta, a, b = simulate()
    Y[0, :] = 0.0  # answered everything, got everything wrong
    Y[1, 50:] = np.nan  # only attempted 50 questions
    Y[1, :50] = 0.0  # and got all of those wrong too
    Y[2, :] = 1.0  # got everything right
    fit = fit_2pl(Y)

    flagged = set(np.where(fit.separated_models)[0])
    print(f"separated models flagged: {sorted(flagged)}")
    assert {0, 1, 2} <= flagged
    assert np.isfinite(fit.theta).all()

    ok = fit.theta[~fit.separated_models]
    print(f"non-separated theta: mean {ok.mean():.3f}, sd {ok.std():.3f}")
    assert abs(ok.mean()) < 1e-6 and abs(ok.std() - 1) < 1e-6
    assert len(fit.leaderboard()) == len(fit.theta) - len(flagged)


if __name__ == "__main__":
    test_recovers_ability_ranking()
    test_recovers_difficulty()
    test_finds_planted_dead_items()
    test_handles_missing_entries()
    test_flags_separated_models()
    print("all passed")
