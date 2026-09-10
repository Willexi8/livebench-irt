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

from livebench_irt.irt import (  # noqa: E402
    bootstrap_theta,
    fit_2pl,
    flag_bad_items,
    rank_confidence_sets,
)
from livebench_irt.load import build_matrix, load_judgments, raw_leaderboard  # noqa: E402
from livebench_irt.plots import (  # noqa: E402
    difficulty_discrimination,
    leaderboard_with_error_bars,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
N_BOOT = 200

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

    ax = difficulty_discrimination(fit)
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES / f"items{suffix}.png", dpi=200)

    # ---- the tables that go in the email --------------------------------
    mean_score = raw_leaderboard(df, category=category)
    order = np.argsort(-fit.theta)
    table = pd.DataFrame(
        {
            "irt_rank": np.arange(1, len(order) + 1),
            "model": fit.models[order],
            "theta": fit.theta[order].round(3),
            "rank_lo": lo_rank[order].astype(int),
            "rank_hi": hi_rank[order].astype(int),
            "mean_score": [mean_score.get(m, np.nan) for m in fit.models[order]],
        }
    )
    table.to_csv(ROOT / f"leaderboard{suffix}.csv", index=False)

    bad = pd.DataFrame(
        flag_bad_items(fit), columns=["question_id", "discrimination", "difficulty"]
    )
    bad.to_csv(ROOT / f"bad_questions{suffix}.csv", index=False)

    # how many published adjacent pairs are actually indistinguishable?
    overlapping = sum(
        1
        for i in range(len(table) - 1)
        if table.rank_hi.iloc[i] >= table.rank_lo.iloc[i + 1]
    )
    print(f"\n{len(bad)} questions with discrimination < 0.15")
    print(
        f"{overlapping} of {len(table) - 1} adjacent leaderboard pairs have "
        f"overlapping rank intervals"
    )
    print(f"\nfigures -> {FIGURES}/   tables -> {ROOT}/")
