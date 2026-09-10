"""Scale-free item diagnostics.

Why not just use the 2PL discrimination `a_i`?

Because it is not comparable across fits. `a_i` lives on whatever scale theta
was standardised to, so changing an unrelated setting moves it. Measured on
livebench/model_judgment, switching the ability prior from theta_ridge=0 to
theta_ridge=0.5 moved the median `a` by 1.8% but moved the smallest value by
31% (0.051 -> 0.035) and reordered the bottom of the list. A fixed cutoff on
`a` therefore reports a different set of "bad questions" depending on a knob
that has nothing to do with the questions: 5 items at one setting, 8 at the
other.

The item-total correlation does not have this problem. It is the correlation
between the scores on one question and the ability of the models that answered
it, so it lives in [-1, 1] by construction and does not move when theta is
rescaled. It is the standard discrimination index in classical test theory.

The second half of the fix is to stop reporting a list and start reporting an
interval. A point estimate near zero and a point estimate near zero with a
confidence interval reaching 0.4 mean very different things, and only one of
them is evidence about the question.

Bootstrap unit: models. The question being answered is "if a different set of
models had been evaluated, would this question still look uninformative?"
Ability is held fixed at the full-data estimate rather than refitted inside
each draw, so these intervals capture sampling in the model panel but not the
uncertainty in theta itself. They are therefore mildly optimistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ItemDiagnostics:
    items: np.ndarray  # (n_items,) question ids
    r: np.ndarray  # (n_items,) item-total correlation, NaN if not estimable
    lo: np.ndarray  # (n_items,) lower bootstrap bound
    hi: np.ndarray  # (n_items,) upper bootstrap bound
    n_models: np.ndarray  # (n_items,) models that answered
    level: float

    def uninformative(self, threshold=0.2):
        """Questions whose discrimination is confidently below `threshold`.

        The test is on the UPPER bound, not the point estimate: a question is
        only called uninformative if even the optimistic end of its interval
        fails to clear the bar. Questions with a low point estimate and a wide
        interval are not evidence of anything and are deliberately left out.
        """
        flag = np.isfinite(self.hi) & (self.hi < threshold)
        order = np.where(flag)[0][np.argsort(self.r[flag])]
        return [
            (self.items[i], float(self.r[i]), float(self.lo[i]), float(self.hi[i]),
             int(self.n_models[i]))
            for i in order
        ]

    def inconclusive(self, threshold=0.2):
        """Questions that look weak but whose interval cannot rule anything out."""
        flag = np.isfinite(self.r) & (self.r < threshold) & (self.hi >= threshold)
        order = np.where(flag)[0][np.argsort(self.r[flag])]
        return [
            (self.items[i], float(self.r[i]), float(self.lo[i]), float(self.hi[i]),
             int(self.n_models[i]))
            for i in order
        ]


def item_total_correlation(Y, mask, theta, min_models=10):
    """Correlation between each question's scores and model ability.

    Returns NaN for questions answered by fewer than `min_models` models, and
    for questions where every model scored the same (no variance to correlate).
    """
    Y = np.asarray(Y, dtype=float)
    theta = np.asarray(theta, dtype=float)
    n_items = Y.shape[1]
    out = np.full(n_items, np.nan)
    for i in range(n_items):
        sel = mask[:, i]
        if sel.sum() < min_models:
            continue
        y = Y[sel, i]
        t = theta[sel]
        if y.std() < 1e-12 or t.std() < 1e-12:
            continue
        out[i] = float(np.corrcoef(y, t)[0, 1])
    return out


def bootstrap_item_diagnostics(
    Y, mask, theta, items=None, n_boot=400, level=0.95, seed=0, min_models=10
):
    """Item-total correlations with bootstrap intervals over the model panel."""
    Y = np.asarray(Y, dtype=float)
    theta = np.asarray(theta, dtype=float)
    n_models, n_items = Y.shape

    r = item_total_correlation(Y, mask, theta, min_models=min_models)
    n_answered = mask.sum(axis=0)

    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, n_items))
    for k in range(n_boot):
        idx = rng.integers(0, n_models, n_models)
        draws[k] = item_total_correlation(
            Y[idx], mask[idx], theta[idx], min_models=min_models
        )

    import warnings

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns are expected
        lo = np.nanquantile(draws, (1 - level) / 2, axis=0)
        hi = np.nanquantile(draws, 1 - (1 - level) / 2, axis=0)
    # a question whose bootstrap draws were almost all unestimable has no interval
    enough = np.isfinite(draws).sum(axis=0) >= 0.5 * n_boot
    lo = np.where(enough, lo, np.nan)
    hi = np.where(enough, hi, np.nan)

    return ItemDiagnostics(
        items=np.asarray(items if items is not None else np.arange(n_items)),
        r=r,
        lo=lo,
        hi=hi,
        n_models=n_answered,
        level=level,
    )
