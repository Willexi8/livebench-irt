"""Check that the bootstrap intervals are not an artefact of capped iterations.

bootstrap_theta caps L-BFGS at 300 iterations for speed. If theta had not
converged by then, the spread across bootstrap draws would be inflated and the
error bars would be too wide. This compares the capped run against a run with
maxiter=2000.

Measured on livebench/model_judgment (178 models x 494 questions):
    theta sd across models : 1.000   (standardised by construction)
    boot sd, maxiter=300   : 0.167
    boot sd, maxiter=2000  : 0.172
A 3% difference, so the cap is safe. Measurement noise is roughly one sixth of
the spread between models.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livebench_irt.irt import bootstrap_theta, fit_2pl  # noqa: E402
from livebench_irt.load import build_matrix, load_judgments  # noqa: E402

N_BOOT = 20

if __name__ == "__main__":
    df = load_judgments()
    Y, mask, models, items = build_matrix(df)
    fit = fit_2pl(Y, mask, models=models, items=items)

    fast = bootstrap_theta(Y, mask, n_boot=N_BOOT, maxiter=300)
    slow = bootstrap_theta(Y, mask, n_boot=N_BOOT, maxiter=2000)

    sd_theta = fit.theta.std()
    sd_fast = fast.std(axis=0).mean()
    sd_slow = slow.std(axis=0).mean()

    print(f"theta sd across models : {sd_theta:.3f}")
    print(f"boot sd, maxiter=300   : {sd_fast:.3f}")
    print(f"boot sd, maxiter=2000  : {sd_slow:.3f}")
    print(f"relative difference    : {abs(sd_fast - sd_slow) / sd_slow:.1%}")
    print(f"noise-to-signal ratio  : {sd_slow / sd_theta:.2f}")
