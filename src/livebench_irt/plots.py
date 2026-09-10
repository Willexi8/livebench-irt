"""The two figures that make the point. Everything else is supporting material."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def leaderboard_with_error_bars(fit, boot_theta, top_n=30, ax=None, level=0.95):
    """Figure 1. The leaderboard, with the uncertainty it never shows.

    Bars that overlap are models this benchmark cannot tell apart.
    """
    lo = np.quantile(boot_theta, (1 - level) / 2, axis=0)
    hi = np.quantile(boot_theta, 1 - (1 - level) / 2, axis=0)
    order = np.argsort(-fit.theta)[:top_n][::-1]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.28 * len(order) + 1.5))
    y = np.arange(len(order))
    ax.errorbar(
        fit.theta[order],
        y,
        xerr=[fit.theta[order] - lo[order], hi[order] - fit.theta[order]],
        fmt="o",
        markersize=3.5,
        linewidth=1,
        capsize=2,
        color="#1a1a1a",
        ecolor="#999999",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([str(m) for m in fit.models[order]], fontsize=7)
    ax.set_xlabel("estimated ability (θ)")
    ax.set_title(f"LiveBench ability with {int(level * 100)}% bootstrap intervals")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return ax


def item_diagnostics(diag, n_show=40, threshold=0.2, ax=None):
    """Figure 2. The weakest questions, each with its confidence interval.

    This replaces an earlier plot of the 2PL discrimination `a_i` against
    difficulty, with a horizontal line at a fixed cutoff. That plot had two
    problems and both of them are the subject of findings in this repository:
    `a_i` is not comparable across fits, and a point estimate with no interval
    cannot distinguish a weak question from an under-answered one. Drawing it
    would have contradicted the analysis it was illustrating.

    Questions whose whole interval sits below the threshold are marked; the
    rest are drawn in grey, because a low point estimate on its own is not
    evidence.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 0.16 * n_show + 1.5))

    ok = np.isfinite(diag.r) & np.isfinite(diag.lo)
    idx = np.where(ok)[0]
    idx = idx[np.argsort(diag.r[idx])][:n_show][::-1]
    y = np.arange(len(idx))
    confident = diag.hi[idx] < threshold

    for mark, colour, label in ((confident, "#c1440e", f"CI entirely below {threshold}"),
                                (~confident, "#9a9a9a", "interval reaches above")):
        if not mark.any():
            continue
        sel = np.where(mark)[0]
        ax.errorbar(
            diag.r[idx][sel], y[sel],
            xerr=[diag.r[idx][sel] - diag.lo[idx][sel], diag.hi[idx][sel] - diag.r[idx][sel]],
            fmt="o", markersize=3, linewidth=1, capsize=2, color=colour,
            ecolor=colour, alpha=0.9, label=label,
        )

    ax.axvline(threshold, linestyle="--", linewidth=0.8, color="#c1440e", alpha=0.6)
    ax.axvline(0.0, linewidth=0.6, color="#333333", alpha=0.5)
    ax.set_yticks([])
    ax.set_xlabel("item-total correlation (95% bootstrap interval)")
    ax.set_ylabel(f"{len(idx)} lowest-scoring questions")
    ax.set_title(f"{int(confident.sum())} of these {len(idx)} are confidently uninformative")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    return ax
