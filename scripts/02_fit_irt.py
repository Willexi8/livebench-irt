"""Fit the 2PL, write the two figures and the two tables.

    python scripts/02_fit_irt.py            # all categories pooled
    python scripts/02_fit_irt.py reasoning  # one category
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livebench_irt.diagnostics import bootstrap_item_diagnostics  # noqa: E402
from livebench_irt.irt import (  # noqa: E402
    bootstrap_theta,
    fit_2pl,
    rank_confidence_sets,
)
from livebench_irt.load import build_matrix, load_judgments, raw_leaderboard  # noqa: E402
from livebench_irt.plots import item_diagnostics, leaderboard_with_error_bars  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
N_BOOT = 200          # bootstrap draws for ability (resamples questions)
N_BOOT_ITEMS = 400    # bootstrap draws for item diagnostics (resamples models)
R_THRESHOLD = 0.2     # item-total correlation below which a question is weak

if __name__ == "__main__":
    category = sys.argv[1] if len(sys.argv) > 1 else None

    df = load_judgments()
    Y, mask, models, items = build_matrix(df, category=category)

    fit = fit_2pl(Y, mask, models=models, items=items)
    print(f"log-likelihood: {fit.loglik:.1f}")

    boot = bootstrap_theta(Y, mask, n_boot=N_BOOT)
    lo_rank, hi_rank = rank_confidence_sets(boot)

    FIGURES.mkdir(exist_ok=True)
    suffix = f"_{category}" if category else ""

    ax = leaderboard_with_error_bars(fit, boot)
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES / f"leaderboard{suffix}.png", dpi=200)

    # ---- the tables that go in the email --------------------------------
    mean_score = raw_leaderboard(df, category=category)
    # separated models have no identified ability -- report them, rank them last
    order = np.argsort(-np.where(fit.separated_models, -np.inf, fit.theta))
    table = pd.DataFrame(
        {
            "irt_rank": np.arange(1, len(order) + 1),
            "model": fit.models[order],
            "theta": fit.theta[order].round(3),
            "rank_lo": lo_rank[order].astype(int),
            "rank_hi": hi_rank[order].astype(int),
            "mean_score": [mean_score.get(m, np.nan) for m in fit.models[order]],
            "separated": fit.separated_models[order],
        }
    )
    table.to_csv(ROOT / f"leaderboard{suffix}.csv", index=False)

    # Item diagnostics: item-total correlation with bootstrap intervals over the
    # model panel. This replaces a fixed cutoff on the 2PL discrimination, which
    # was not comparable across fits -- see diagnostics.py for the numbers.
    diag = bootstrap_item_diagnostics(Y, mask, fit.theta, items=items, n_boot=N_BOOT_ITEMS)
    cols = ["question_id", "r", "ci_lo", "ci_hi", "n_models"]
    weak = pd.DataFrame(diag.uninformative(threshold=R_THRESHOLD), columns=cols)
    unsure = pd.DataFrame(diag.inconclusive(threshold=R_THRESHOLD), columns=cols)
    weak.to_csv(ROOT / f"weak_questions{suffix}.csv", index=False)
    unsure.to_csv(ROOT / f"inconclusive_questions{suffix}.csv", index=False)

    ax = item_diagnostics(diag, threshold=R_THRESHOLD)
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES / f"items{suffix}.png", dpi=200)

    # how many published adjacent pairs are actually indistinguishable?
    ranked = table[~table.separated].reset_index(drop=True)
    overlapping = sum(
        1
        for i in range(len(ranked) - 1)
        if ranked.rank_hi.iloc[i] >= ranked.rank_lo.iloc[i + 1]
    )
    n_sep = int(table.separated.sum())
    if n_sep:
        print(f"\n{n_sep} models scored all-0 or all-1: ability not identified, excluded from ranking")
        for m in table.model[table.separated]:
            print(f"    {m}")
    print(
        f"\n{len(weak)} questions confidently uninformative "
        f"(95% CI upper bound below {R_THRESHOLD})"
    )
    print(
        f"{len(unsure)} more look weak but their intervals cannot rule it out "
        f"-- not evidence either way"
    )
    print(
        f"{overlapping} of {len(ranked) - 1} adjacent leaderboard pairs have "
        f"overlapping rank intervals"
    )
    print(f"\nfigures -> {FIGURES}/   tables -> {ROOT}/")
