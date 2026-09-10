"""Joint vs marginal MLE on the real data.

Joint MLE estimates an ability parameter for every model alongside the item
parameters, so the number of parameters grows with the sample and the item
estimates are not consistent. Marginal MLE integrates ability out against a
N(0, 1) prior, leaving two parameters per item however many models there are,
and recovers ability afterwards as a posterior mean.

Simulation at the real dimensions (178 models, 494 questions, a third of cells
missing) says the difference shows up in the *scale* of the item parameters,
not in their ordering: joint MLE inflated the mean discrimination by about 40%
while marginal MLE stayed within 5%. Correlation cannot see this, because
correlation is scale-invariant.

This script checks what that means on the actual data: whether the two
estimators agree about the leaderboard, and how far apart their item
parameters sit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livebench_irt.diagnostics import item_total_correlation  # noqa: E402
from livebench_irt.irt import fit_2pl  # noqa: E402
from livebench_irt.load import build_matrix, load_judgments  # noqa: E402
from livebench_irt.mml import fit_2pl_mml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    df = load_judgments()
    Y, mask, models, items = build_matrix(df)

    jml = fit_2pl(Y, mask, models=models, items=items)
    mml = fit_2pl_mml(Y, mask, models=models, items=items)
    print(f"MML converged in {mml.n_em} EM steps, marginal log-likelihood {mml.loglik:.1f}")

    # ---- do they agree about the ranking? -------------------------------
    rank_j = (-jml.theta).argsort().argsort() + 1
    rank_m = (-mml.theta).argsort().argsort() + 1
    shift = np.abs(rank_j - rank_m)
    print()
    print(f"ability: pearson {np.corrcoef(jml.theta, mml.theta)[0, 1]:.4f}, "
          f"spearman {pd.Series(jml.theta).corr(pd.Series(mml.theta), method='spearman'):.4f}")
    print(f"rank shift: median {np.median(shift):.0f}, mean {shift.mean():.1f}, max {shift.max()}")

    # ---- where they disagree: the item parameter scale -------------------
    print()
    print(f"{'':22s} {'JML':>8} {'MML':>8}")
    print(f"{'mean discrimination':22s} {jml.a.mean():8.3f} {mml.a.mean():8.3f}")
    print(f"{'sd of discrimination':22s} {jml.a.std():8.3f} {mml.a.std():8.3f}")
    print(f"{'sd of difficulty':22s} {jml.b.std():8.3f} {mml.b.std():8.3f}")
    print(f"discrimination correlation between the two: "
          f"{np.corrcoef(jml.a, mml.a)[0, 1]:.4f}")

    # ---- does the choice of estimator change the diagnostics? ------------
    r_j = item_total_correlation(Y, mask, jml.theta)
    r_m = item_total_correlation(Y, mask, mml.theta)
    ok = np.isfinite(r_j) & np.isfinite(r_m)
    print()
    print("item-total correlation computed against each ability estimate:")
    print(f"  correlation between the two: {np.corrcoef(r_j[ok], r_m[ok])[0, 1]:.4f}")
    print(f"  max absolute difference:     {np.abs(r_j[ok] - r_m[ok]).max():.4f}")

    out = pd.DataFrame(
        {
            "model": models,
            "theta_jml": jml.theta,
            "theta_mml": mml.theta,
            "theta_mml_sd": mml.theta_sd,
            "rank_jml": rank_j,
            "rank_mml": rank_m,
        }
    ).sort_values("rank_mml")
    out.to_csv(ROOT / "estimator_comparison.csv", index=False)
    print(f"\nwrote {ROOT / 'estimator_comparison.csv'}")
