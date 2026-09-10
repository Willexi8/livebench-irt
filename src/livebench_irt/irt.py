"""Two-parameter logistic (2PL) IRT model for benchmark score matrices.

Model
-----
For model j and item i:

    P(correct) = sigmoid( a_i * (theta_j - b_i) )

    theta_j : ability of model j        (what a leaderboard *should* report)
    b_i     : difficulty of item i
    a_i     : discrimination of item i  (a_i ~ 0  =>  the item tells us nothing)

Fitting is joint maximum likelihood with a small ridge penalty, optimised with
L-BFGS. Identification: theta is standardised (mean 0, sd 1) after fitting, and
a, b are rescaled so the fitted probabilities are unchanged.

Scores in LiveBench are floats in [0, 1] (some tasks give partial credit). We
use the Bernoulli log-likelihood evaluated at fractional y, which is the usual
quasi-likelihood treatment and is what `binary_cross_entropy` does anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass
class IRTFit:
    theta: np.ndarray  # (n_models,)  ability
    a: np.ndarray  # (n_items,)   discrimination
    b: np.ndarray  # (n_items,)   difficulty
    models: np.ndarray  # (n_models,)  model names
    items: np.ndarray  # (n_items,)   question ids
    loglik: float

    def leaderboard(self):
        """Models ordered by fitted ability."""
        order = np.argsort(-self.theta)
        return list(zip(self.models[order], self.theta[order]))


def _unpack(params, n_models, n_items):
    theta = params[:n_models]
    log_a = params[n_models : n_models + n_items]
    b = params[n_models + n_items :]
    return theta, np.exp(log_a), b


def _neg_loglik(params, Y, mask, n_models, n_items, ridge):
    theta, a, b = _unpack(params, n_models, n_items)
    z = a[None, :] * (theta[:, None] - b[None, :])
    # stable log-sigmoid: log p = -log(1 + exp(-z)), log(1-p) = -z - log(1+exp(-z))
    log_p = -np.logaddexp(0.0, -z)
    log_q = -z - np.logaddexp(0.0, -z)
    ll = np.where(mask, Y * log_p + (1.0 - Y) * log_q, 0.0).sum()

    p = expit(z)
    resid = np.where(mask, p - Y, 0.0)
    g_theta = (resid * a[None, :]).sum(axis=1)
    g_b = -(resid * a[None, :]).sum(axis=0)
    g_log_a = (resid * (theta[:, None] - b[None, :]) * a[None, :]).sum(axis=0)

    # ridge on log_a and b keeps the joint MLE from running off on sparse items
    log_a = np.log(a)
    ll -= ridge * (np.sum(log_a**2) + np.sum(b**2))
    g_log_a += 2 * ridge * log_a
    g_b += 2 * ridge * b

    grad = np.concatenate([g_theta, g_log_a, g_b])
    return -ll, grad


def fit_2pl(Y, mask=None, ridge=1e-2, models=None, items=None, maxiter=2000, init=None):
    """Fit a 2PL model to a (n_models x n_items) score matrix.

    Y     : array of scores in [0, 1]. NaNs are treated as not-attempted.
    mask  : optional boolean array, True where the score is observed.
    init  : optional (theta, a, b) warm start -- used by the bootstrap, where
            every refit is a small perturbation of the full-data fit.
    """
    Y = np.asarray(Y, dtype=float)
    if mask is None:
        mask = ~np.isnan(Y)
    Y = np.where(mask, np.clip(Y, 1e-6, 1 - 1e-6), 0.0)
    n_models, n_items = Y.shape

    if init is not None:
        theta0, a0, b0 = init
        x0 = np.concatenate([theta0, np.log(np.maximum(a0, 1e-6)), b0])
    else:
        # sensible starting point: standardised model means / item means
        with np.errstate(invalid="ignore"):
            model_rate = np.nansum(Y * mask, axis=1) / np.maximum(mask.sum(axis=1), 1)
            item_rate = np.nansum(Y * mask, axis=0) / np.maximum(mask.sum(axis=0), 1)
        theta0 = _std(_logit(model_rate))
        b0 = _logit(1 - item_rate)
        x0 = np.concatenate([theta0, np.zeros(n_items), b0])

    res = minimize(
        _neg_loglik,
        x0,
        args=(Y, mask, n_models, n_items, ridge),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "maxfun": maxiter * 2},
    )
    theta, a, b = _unpack(res.x, n_models, n_items)

    # identification: theta ~ mean 0, sd 1; rescale a, b to keep z unchanged
    mu, sd = theta.mean(), theta.std()
    sd = sd if sd > 1e-8 else 1.0
    theta = (theta - mu) / sd
    b = (b - mu) / sd
    a = a * sd

    return IRTFit(
        theta=theta,
        a=a,
        b=b,
        models=np.asarray(models if models is not None else np.arange(n_models)),
        items=np.asarray(items if items is not None else np.arange(n_items)),
        loglik=-res.fun,
    )


def bootstrap_theta(Y, mask=None, n_boot=200, seed=0, **kwargs):
    """Resample ITEMS (not cells) to get error bars on ability.

    Items are the sampling unit: LiveBench draws a fresh set of questions each
    month, so "what if we had drawn different questions" is the question the
    error bars should answer.
    """
    Y = np.asarray(Y, dtype=float)
    if mask is None:
        mask = ~np.isnan(Y)
    # theta stabilises far earlier than the item parameters do, and theta is
    # all the bootstrap uses -- capping iterations here costs nothing measurable
    # and turns a 12-minute run into a 4-minute one.
    kwargs.setdefault("maxiter", 300)
    rng = np.random.default_rng(seed)
    n_items = Y.shape[1]
    full = fit_2pl(Y, mask, **kwargs)  # warm start: each refit is a perturbation
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        fit = fit_2pl(
            Y[:, idx],
            mask[:, idx],
            init=(full.theta, full.a[idx], full.b[idx]),
            **kwargs,
        )
        draws.append(fit.theta)
    return np.vstack(draws)  # (n_boot, n_models)


def rank_confidence_sets(boot_theta, level=0.95):
    """For each model, the range of ranks it plausibly occupies.

    Two models whose rank intervals overlap are not distinguishable by this
    benchmark, however far apart they sit on the published leaderboard.
    """
    ranks = (-boot_theta).argsort(axis=1).argsort(axis=1) + 1  # 1 = best
    lo = np.quantile(ranks, (1 - level) / 2, axis=0)
    hi = np.quantile(ranks, 1 - (1 - level) / 2, axis=0)
    return lo, hi


def flag_bad_items(fit, a_threshold=0.15):
    """Items that carry (almost) no information about model ability."""
    idx = np.where(fit.a < a_threshold)[0]
    order = idx[np.argsort(fit.a[idx])]
    return [(fit.items[i], float(fit.a[i]), float(fit.b[i])) for i in order]


def _logit(p, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _std(x):
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-8 else 1.0)
