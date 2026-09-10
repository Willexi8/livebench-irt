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

Separation
----------
A model that scores 0 on every question it attempted (or 1 on every question)
has no finite maximum likelihood estimate of ability: the likelihood increases
monotonically as theta runs to -inf (or +inf). Left alone, the optimiser walks
off and returns whatever it happened to reach, which looks like an estimate but
is not one. Two things guard against this: a ridge penalty on theta, which is
equivalent to a N(0, tau) prior and gives a finite posterior mode; and an
explicit separation flag, so downstream code can report those models as
"below/above the range this benchmark can measure" rather than as a number.
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
    separated_models: np.ndarray  # (n_models,) bool: all-0 or all-1, theta not identified
    separated_items: np.ndarray  # (n_items,)  bool: all-0 or all-1, b not identified

    def leaderboard(self, drop_separated=True):
        """Models ordered by fitted ability.

        Separated models are dropped by default: their theta is set by the
        prior, not by the data, so ranking them against the rest is meaningless.
        """
        keep = ~self.separated_models if drop_separated else np.ones_like(self.theta, bool)
        idx = np.where(keep)[0]
        order = idx[np.argsort(-self.theta[idx])]
        return list(zip(self.models[order], self.theta[order]))


def _unpack(params, n_models, n_items):
    theta = params[:n_models]
    log_a = params[n_models : n_models + n_items]
    b = params[n_models + n_items :]
    return theta, np.exp(log_a), b


def _neg_loglik(params, Y, mask, n_models, n_items, ridge, theta_ridge):
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

    # ridge on log_a and b keeps the joint MLE from running off on sparse items;
    # ridge on theta does the same for models with no finite MLE (see Separation)
    log_a = np.log(a)
    ll -= ridge * (np.sum(log_a**2) + np.sum(b**2))
    ll -= theta_ridge * np.sum(theta**2)
    g_log_a += 2 * ridge * log_a
    g_b += 2 * ridge * b
    g_theta += 2 * theta_ridge * theta

    grad = np.concatenate([g_theta, g_log_a, g_b])
    return -ll, grad


def fit_2pl(
    Y,
    mask=None,
    ridge=1e-2,
    theta_ridge=0.5,
    models=None,
    items=None,
    maxiter=2000,
    init=None,
):
    """Fit a 2PL model to a (n_models x n_items) score matrix.

    Y     : array of scores in [0, 1]. NaNs are treated as not-attempted.
    mask  : optional boolean array, True where the score is observed.
    init  : optional (theta, a, b) warm start -- used by the bootstrap, where
            every refit is a small perturbation of the full-data fit.
    theta_ridge : penalty on ability. 0.5 corresponds to a N(0, 1) prior on
            theta, the usual IRT identification assumption; it keeps separated
            models finite and, on simulated data with 400 items, costs nothing
            in recovery (r = 0.9920 at both 0 and 0.5). Set to 0 for the
            unpenalised joint MLE, which diverges on separated models.
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
        args=(Y, mask, n_models, n_items, ridge, theta_ridge),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "maxfun": maxiter * 2},
    )
    theta, a, b = _unpack(res.x, n_models, n_items)
    sep_models, sep_items = _find_separation(Y, mask)

    # identification: theta ~ mean 0, sd 1; rescale a, b to keep z unchanged.
    # Separated models are excluded from the standardisation -- their theta is
    # set by the prior, and letting it into the mean and sd would rescale every
    # other model's ability around a number the data never estimated.
    ref = theta[~sep_models] if (~sep_models).sum() > 1 else theta
    mu, sd = ref.mean(), ref.std()
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
        separated_models=sep_models,
        separated_items=sep_items,
    )


def _find_separation(Y, mask, eps=1e-3):
    """Rows and columns whose observed scores are all at one extreme.

    Uses the observed cells only: a model that answered 50 questions and got
    every one wrong is separated, regardless of the 444 it never attempted.
    """
    n_obs_m = mask.sum(axis=1)
    n_obs_i = mask.sum(axis=0)
    tot_m = np.where(mask, Y, 0.0).sum(axis=1)
    tot_i = np.where(mask, Y, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_m = np.divide(tot_m, n_obs_m, out=np.zeros_like(tot_m), where=n_obs_m > 0)
        mean_i = np.divide(tot_i, n_obs_i, out=np.zeros_like(tot_i), where=n_obs_i > 0)
    sep_m = (n_obs_m > 0) & ((mean_m <= eps) | (mean_m >= 1 - eps))
    sep_i = (n_obs_i > 0) & ((mean_i <= eps) | (mean_i >= 1 - eps))
    return sep_m, sep_i


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


def _logit(p, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _std(x):
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-8 else 1.0)
