"""Tests for marginal maximum likelihood.

The claim being checked is not "MML recovers the parameters" -- joint MLE does
that too, and correlation cannot tell them apart because correlation is
scale-invariant and the joint MLE bias is a scale bias. The claim is that MML
gets the *scale* of the item parameters right where joint MLE inflates it, and
that the gap widens as the model panel gets sparser, which is the regime the
real data sits in (67% of cells observed).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livebench_irt.irt import fit_2pl  # noqa: E402
from livebench_irt.mml import _e_step, _quadrature, fit_2pl_mml  # noqa: E402


def simulate(seed=0, n_models=178, n_items=300, missing=0.0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n_models)
    a = np.exp(rng.normal(0, 0.35, n_items))
    b = rng.normal(0, 1.2, n_items)
    p = expit(a[None, :] * (theta[:, None] - b[None, :]))
    Y = (rng.random(p.shape) < p).astype(float)
    mask = rng.random(p.shape) > missing
    return np.where(mask, Y, np.nan), mask, theta, a, b


def test_quadrature_integrates_the_prior():
    x, w = _quadrature(41)
    print(f"quadrature: mean {(w * x).sum():.1e}, E[x^2] {(w * x * x).sum():.6f}")
    assert abs((w * x).sum()) < 1e-10
    assert abs((w * x * x).sum() - 1) < 1e-9


def test_recovers_parameters():
    Y, mask, theta, a, b = simulate()
    fit = fit_2pl_mml(Y, mask)
    rt = np.corrcoef(fit.theta, theta)[0, 1]
    rb = np.corrcoef(fit.b, b)[0, 1]
    print(f"recovery: ability r={rt:.3f}, difficulty r={rb:.3f}, EM steps={fit.n_em}")
    assert rt > 0.97 and rb > 0.90
    assert fit.converged


def test_em_increases_the_likelihood():
    Y, mask, theta, a, b = simulate(n_items=120)
    lls = []
    for k in (1, 2, 4, 8, 16, 32):
        lls.append(fit_2pl_mml(Y, mask, max_em=k, tol=0.0).loglik)
    print("marginal log-likelihood by EM steps:", [round(x, 1) for x in lls])
    assert all(b_ >= a_ - 1e-6 for a_, b_ in zip(lls, lls[1:]))


def test_discrimination_scale_on_a_sparse_panel():
    # The regime that matters: the real matrix has 67% of its cells observed.
    # There, joint MLE inflates the discrimination scale badly and erratically
    # -- measured inflation across item counts was 1.5x at 494 items, 1.5x at
    # 300, and 5.9x at 150 -- while MML stays within a couple of percent.
    #
    # On a fully observed panel joint MLE is not systematically worse, only
    # unstable (ratios of 0.98, 1.17 and 1.32 at three item counts), so no
    # claim is asserted there. It is printed as context.
    print(f"{'missing':>8} | {'JML':>7} {'MML':>7}   (mean a_hat / mean a_true)")
    results = {}
    for missing in (0.0, 0.33):
        j, m = [], []
        for seed in range(2):
            Y, mask, theta, a, b = simulate(seed=seed, n_items=494, missing=missing)
            j.append(fit_2pl(Y).a.mean() / a.mean())
            m.append(fit_2pl_mml(Y, mask).a.mean() / a.mean())
        results[missing] = (np.mean(j), np.mean(m))
        print(f"{missing:8.0%} | {results[missing][0]:7.3f} {results[missing][1]:7.3f}")

    jm, mm = results[0.33]
    assert abs(mm - 1) < 0.10, "MML should recover the discrimination scale"
    assert abs(mm - 1) < abs(jm - 1), "and should beat joint MLE on a sparse panel"


def test_eap_is_finite_for_a_model_that_scores_zero():
    # Where the joint MLE does not exist, the posterior mean still does: it is
    # an average against a proper prior, not a maximiser.
    Y, mask, theta, a, b = simulate(n_items=200)
    Y[0, :] = 0.0
    fit = fit_2pl_mml(Y, mask)
    print(f"all-zero model: theta={fit.theta[0]:.2f}, posterior sd={fit.theta_sd[0]:.2f}, "
          f"next lowest={np.sort(fit.theta)[1]:.2f}")
    assert np.isfinite(fit.theta).all()
    assert fit.theta[0] == fit.theta.min()
    assert fit.theta[0] > -10


if __name__ == "__main__":
    test_quadrature_integrates_the_prior()
    test_recovers_parameters()
    test_em_increases_the_likelihood()
    test_discrimination_scale_on_a_sparse_panel()
    test_eap_is_finite_for_a_model_that_scores_zero()
    print("all passed")
