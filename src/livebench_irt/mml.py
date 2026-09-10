"""Marginal maximum likelihood for the 2PL.

Why this exists
---------------
Joint MLE estimates one ability parameter per model alongside the item
parameters. The number of ability parameters grows with the sample, so the item
parameters are not consistently estimated -- the incidental parameters problem
(Neyman and Scott, 1948). Measured on simulated data in `tests/test_mml.py`,
joint MLE recovers difficulty at r = 0.81 with 60 models, 0.94 with 120, 0.96
with 300. The item parameters are the ones this project uses for diagnostics,
so this matters.

Marginal MLE treats ability as a random effect drawn from N(0, 1) and
integrates it out, leaving only two parameters per item however many models
there are. Ability is then recovered afterwards as a posterior mean (EAP),
which is a shrinkage estimate rather than a maximiser -- and, usefully, is
finite for a model that scored zero throughout, where the joint MLE does not
exist at all.

Method
------
The integral over ability is a Gaussian integral, so Gauss-Hermite quadrature
handles it in a fixed set of nodes. Then EM:

    E-step   posterior weight of each model at each ability node
    M-step   for each item, a weighted logistic regression against the nodes,
             using the expected counts the E-step produced

Both steps are vectorised over models and items; the M-step is a single L-BFGS
solve over all item parameters at once, which is valid because the expected
complete-data likelihood separates across items.

Fractional scores are handled the same way as elsewhere in this repository: the
expected counts carry fractional successes, which is what a quasi-likelihood
treatment amounts to. See the README for why that is an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass
class MMLFit:
    theta: np.ndarray  # (n_models,) posterior mean ability (EAP)
    theta_sd: np.ndarray  # (n_models,) posterior standard deviation
    a: np.ndarray  # (n_items,) discrimination
    b: np.ndarray  # (n_items,) difficulty
    models: np.ndarray
    items: np.ndarray
    loglik: float  # marginal log-likelihood
    n_em: int
    converged: bool

    def leaderboard(self):
        order = np.argsort(-self.theta)
        return list(zip(self.models[order], self.theta[order]))


def _quadrature(n_quad):
    """Nodes and weights for integrating against a standard normal density."""
    x, w = np.polynomial.hermite_e.hermegauss(n_quad)
    return x, w / w.sum()


def _log_p(a, b, nodes):
    """(n_items, n_quad) log P(correct) and log P(wrong) at each node."""
    z = a[:, None] * (nodes[None, :] - b[:, None])
    log_p = -np.logaddexp(0.0, -z)
    log_q = -z - np.logaddexp(0.0, -z)
    return log_p, log_q


def _e_step(Y, mask, a, b, nodes, log_prior):
    """Posterior weights over ability nodes, and the marginal log-likelihood."""
    log_p, log_q = _log_p(a, b, nodes)
    # (n_models, n_quad): log-likelihood of each model's responses at each node
    ll = (Y * mask) @ log_p + ((1.0 - Y) * mask) @ log_q
    ll = ll + log_prior[None, :]
    m = ll.max(axis=1, keepdims=True)
    unnorm = np.exp(ll - m)
    total = unnorm.sum(axis=1, keepdims=True)
    W = unnorm / total
    marginal = float((np.log(total) + m).sum())
    return W, marginal


def _m_step(Y, mask, W, nodes, a0, b0, ridge, maxiter):
    """Weighted logistic regression per item on the expected counts."""
    nbar = mask.T @ W  # (n_items, n_quad) expected attempts
    rbar = (Y * mask).T @ W  # (n_items, n_quad) expected successes
    n_items = nbar.shape[0]

    def objective(params):
        log_a = params[:n_items]
        b = params[n_items:]
        a = np.exp(log_a)
        z = a[:, None] * (nodes[None, :] - b[:, None])
        log_p = -np.logaddexp(0.0, -z)
        log_q = -z - np.logaddexp(0.0, -z)
        ll = (rbar * log_p + (nbar - rbar) * log_q).sum()
        ll -= ridge * (np.sum(log_a**2) + np.sum(b**2))

        p = expit(z)
        resid = nbar * p - rbar  # (n_items, n_quad)
        g_b = -(resid * a[:, None]).sum(axis=1)
        g_log_a = (resid * (nodes[None, :] - b[:, None]) * a[:, None]).sum(axis=1)
        g_log_a += 2 * ridge * log_a
        g_b += 2 * ridge * b
        return -ll, np.concatenate([g_log_a, g_b])

    x0 = np.concatenate([np.log(np.maximum(a0, 1e-6)), b0])
    res = minimize(
        objective, x0, jac=True, method="L-BFGS-B",
        options={"maxiter": maxiter, "maxfun": maxiter * 2},
    )
    return np.exp(res.x[:n_items]), res.x[n_items:]


def fit_2pl_mml(
    Y,
    mask=None,
    models=None,
    items=None,
    n_quad=41,
    max_em=300,
    tol=1e-6,
    ridge=1e-3,
    m_maxiter=200,
):
    """Fit a 2PL by marginal maximum likelihood.

    Returns item parameters estimated with ability integrated out, and EAP
    ability estimates with their posterior standard deviations.

    n_quad : quadrature nodes. 41 is generous; accuracy is flat above ~21.
    tol    : stop when the marginal log-likelihood improves by less than this.
    """
    Y = np.asarray(Y, dtype=float)
    if mask is None:
        mask = ~np.isnan(Y)
    mask = np.asarray(mask, dtype=bool)
    Y = np.where(mask, np.clip(Y, 1e-9, 1 - 1e-9), 0.0)
    n_models, n_items = Y.shape

    nodes, weights = _quadrature(n_quad)
    log_prior = np.log(weights)

    # start from item pass rates; discrimination starts flat
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.divide(
            (Y * mask).sum(axis=0), np.maximum(mask.sum(axis=0), 1),
            out=np.full(n_items, 0.5), where=mask.sum(axis=0) > 0,
        )
    a = np.ones(n_items)
    b = np.clip(np.log((1 - rate + 1e-3) / (rate + 1e-3)), -4, 4)

    prev, converged, it = -np.inf, False, 0
    for it in range(1, max_em + 1):
        W, marginal = _e_step(Y, mask, a, b, nodes, log_prior)
        a, b = _m_step(Y, mask, W, nodes, a, b, ridge, m_maxiter)
        if marginal - prev < tol * max(1.0, abs(prev)):
            converged = True
            break
        prev = marginal

    W, marginal = _e_step(Y, mask, a, b, nodes, log_prior)
    theta = W @ nodes
    theta_sd = np.sqrt(np.maximum(W @ (nodes**2) - theta**2, 0.0))

    return MMLFit(
        theta=theta,
        theta_sd=theta_sd,
        a=a,
        b=b,
        models=np.asarray(models if models is not None else np.arange(n_models)),
        items=np.asarray(items if items is not None else np.arange(n_items)),
        loglik=marginal,
        n_em=it,
        converged=converged,
    )
