"""Is the discrimination gap between tasks a few bad questions, or the whole distribution?

The earlier framing -- "23% of `typos` questions are uninformative against 2.6%
of `LCB_generation`" -- depends on where the flagging threshold sits, and that
turned out to be a real problem: moving an unrelated regularisation parameter
moved the count from 5 to 8. This script asks the threshold-free version of the
question instead: how does the whole distribution of item discrimination differ
between tasks?

Attenuation control. Point-biserial correlation is attenuated on binary
outcomes, and the further a question's pass rate sits from 0.5 the worse the
attenuation. So the headline comparison is restricted to binary-scored tasks,
and the pass rate of each is reported alongside, so the reader can check that
the ordering is not simply the attenuation ordering. Graded and continuous
tasks are drawn on the figure for context but kept out of the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livebench_irt.diagnostics import item_total_correlation  # noqa: E402
from livebench_irt.irt import fit_2pl  # noqa: E402
from livebench_irt.load import build_matrix, load_judgments  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
MIN_QUESTIONS = 20  # tasks smaller than this are too thin to compare
BINARY_LEVELS = 2


def ecdf(x):
    x = np.sort(np.asarray(x))
    return x, np.arange(1, len(x) + 1) / len(x)


def main():
    df = load_judgments()
    Y, mask, models, items = build_matrix(df)
    fit = fit_2pl(Y, mask, models=models, items=items)
    r = item_total_correlation(Y, mask, fit.theta)

    per_q = df.groupby("question_id").agg(
        task=("task", "first"),
        levels=("score", "nunique"),
        pass_rate=("score", "mean"),
    )
    d = pd.DataFrame({"question_id": items, "r": r}).join(per_q, on="question_id")
    d = d.dropna(subset=["r"])

    # a task is binary-scored if its questions only ever take two score values
    task_levels = d.groupby("task").levels.max()
    binary_tasks = set(task_levels[task_levels <= BINARY_LEVELS].index)

    summary = (
        d.groupby("task")
        .agg(
            n=("r", "size"),
            mean_r=("r", "mean"),
            sd_r=("r", "std"),
            median_r=("r", "median"),
            pass_rate=("pass_rate", "mean"),
        )
        .sort_values("mean_r")
    )
    summary["binary"] = summary.index.isin(binary_tasks)
    summary = summary[summary.n >= MIN_QUESTIONS]
    summary.round(3).to_csv(ROOT / "task_discrimination.csv")
    print(summary.round(3).to_string())

    # ---- the test, on binary-scored tasks only --------------------------
    from scipy.stats import kruskal, mannwhitneyu

    groups = {t: d.loc[d.task == t, "r"].values for t in summary.index[summary.binary]}
    if len(groups) >= 2:
        h, p = kruskal(*groups.values())
        print(f"\nKruskal-Wallis across {len(groups)} binary-scored tasks: H={h:.1f}, p={p:.2e}")

        names = list(groups)
        pairs = [(a, b) for k, a in enumerate(names) for b in names[k + 1 :]]
        raw = np.array([mannwhitneyu(groups[a], groups[b])[1] for a, b in pairs])
        order = np.argsort(raw)  # Holm-Bonferroni
        adj = np.minimum(1.0, raw[order] * (len(raw) - np.arange(len(raw))))
        adj = np.maximum.accumulate(adj)
        holm = np.empty_like(adj)
        holm[order] = adj
        print("\npairwise (Holm-adjusted):")
        for (a, b), p_adj in zip(pairs, holm):
            gap = groups[a].mean() - groups[b].mean()
            print(f"  {a:20s} vs {b:20s}  mean gap {gap:+.3f}  p={p_adj:.4f}")

    # ---- figure ---------------------------------------------------------
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = plt.cm.viridis(np.linspace(0.1, 0.85, len(summary)))
    for color, task in zip(palette, summary.index):
        vals = d.loc[d.task == task, "r"].values
        x, y = ecdf(vals)
        style = "-" if task in binary_tasks else "--"
        ax.step(
            x, y, style, where="post", color=color, linewidth=1.6,
            label=f"{task} (n={len(vals)}, mean {vals.mean():.2f})",
        )
    ax.set_xlabel("item-total correlation")
    ax.set_ylabel("cumulative fraction of questions")
    ax.set_title("Item discrimination by task")  # descriptive: let the curves argue
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.92, edgecolor="none")
    ax.grid(alpha=0.2, linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(
        0.02, 0.97, "solid = binary-scored\ndashed = graded or continuous",
        transform=ax.transAxes, fontsize=7.5, va="top", color="#555555",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "task_discrimination.png", dpi=200)
    print(f"\nfigure -> {FIGURES / 'task_discrimination.png'}")


if __name__ == "__main__":
    main()
