"""Write a synthetic judgment table in LiveBench's schema to data/.

Run this before touching the real download. It lets you develop and debug the
whole pipeline against data whose true parameters you know -- including 25
planted "dead" questions that the analysis is supposed to find.

    python scripts/00_smoke_test.py
    python scripts/02_fit_irt.py

Delete data/model_judgment.parquet afterwards, or 01_download.py will happily
keep using the fake one.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
N_MODELS, N_ITEMS, N_DEAD = 90, 500, 25
CATEGORIES = ["math", "reasoning", "coding", "language", "data_analysis", "instruction_following"]

if __name__ == "__main__":
    out = ROOT / "data" / "model_judgment.parquet"
    if out.exists() and "--force" not in sys.argv:
        sys.exit(f"{out} already exists -- pass --force to overwrite")

    rng = np.random.default_rng(3)
    theta = rng.normal(0, 1, N_MODELS)
    a = np.exp(rng.normal(0, 0.35, N_ITEMS))
    b = rng.normal(0, 1.2, N_ITEMS)
    a[:N_DEAD] = 0.02  # planted dead questions: q0000 .. q0024
    category = rng.choice(CATEGORIES, N_ITEMS)

    p = expit(a[None, :] * (theta[:, None] - b[None, :]))
    Y = (rng.random(p.shape) < p).astype(float)
    seen = rng.random(p.shape) > 0.15  # not every model answers every question

    rows = [
        (f"q{i:04d}", f"task_{category[i]}", f"model-{j:02d}", Y[j, i], 1, 1.7e9, category[i])
        for j in range(N_MODELS)
        for i in range(N_ITEMS)
        if seen[j, i]
    ]
    df = pd.DataFrame(
        rows, columns=["question_id", "task", "model", "score", "turn", "tstamp", "category"]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"wrote {len(df):,} synthetic rows to {out}")
    print(f"planted dead questions: q0000 .. q{N_DEAD - 1:04d}")
