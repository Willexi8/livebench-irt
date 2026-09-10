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


def difficulty_discrimination(fit, a_threshold=0.15, ax=None):
    """Figure 2. Every question as a point. The flat band along the bottom is
    the part of the benchmark that is not measuring anything."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5))
    dead = fit.a < a_threshold
    ax.scatter(fit.b[~dead], fit.a[~dead], s=7, alpha=0.45, color="#1a1a1a", label="informative")
    ax.scatter(fit.b[dead], fit.a[dead], s=10, alpha=0.8, color="#c1440e", label="a < threshold")
    ax.axhline(a_threshold, linestyle="--", linewidth=0.8, color="#c1440e", alpha=0.6)
    ax.set_xlabel("difficulty (b)")
    ax.set_ylabel("discrimination (a)")
    ax.set_title(f"{dead.sum()} of {len(fit.a)} questions carry almost no signal")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax
