"""Load LiveBench per-question judgments and reshape into a score matrix.

Source: https://huggingface.co/datasets/livebench/model_judgment
Columns: question_id, task, model, score, turn, tstamp, category

NOTE: the exact parquet filename on the Hub changes as LiveBench re-uploads.
If the direct URL 404s, use `load_judgments(use_datasets=True)`, which lets the
`datasets` library resolve the current file for you.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HF_PARQUET_URL = (
    "https://huggingface.co/datasets/livebench/model_judgment/"
    "resolve/main/data/leaderboard-00000-of-00001.parquet"
)
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "model_judgment.parquet"


def load_judgments(cache: Path | str = DEFAULT_CACHE, use_datasets: bool = False) -> pd.DataFrame:
    """Return the raw judgment table, downloading and caching it once."""
    cache = Path(cache)
    if cache.exists():
        return pd.read_parquet(cache)

    if use_datasets:
        from datasets import load_dataset

        df = load_dataset("livebench/model_judgment", split="leaderboard").to_pandas()
    else:
        df = pd.read_parquet(HF_PARQUET_URL)

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def build_matrix(
    df: pd.DataFrame,
    category: str | None = None,
    min_items_per_model: int = 30,
    min_models_per_item: int = 10,
):
    """Pivot judgments into a (models x items) score matrix.

    Multiple turns of the same (model, question) are averaged. Rows and columns
    that are too thin to estimate anything from are dropped -- and the function
    tells you how many it dropped, because that number belongs in the writeup.

    Returns (Y, mask, models, items) where Y has NaN for unattempted cells.
    """
    if category is not None:
        df = df[df["category"] == category]

    wide = (
        df.groupby(["model", "question_id"])["score"]
        .mean()
        .unstack("question_id")
        .sort_index()
    )

    # LiveBench scores are already in [0, 1] for most tasks; guard anyway
    if wide.to_numpy(dtype=float, na_value=np.nan).max() > 1.0:
        wide = wide / 100.0

    before = wide.shape
    keep_items = wide.notna().sum(axis=0) >= min_models_per_item
    wide = wide.loc[:, keep_items]
    keep_models = wide.notna().sum(axis=1) >= min_items_per_model
    wide = wide.loc[keep_models]
    print(
        f"matrix {before[0]}x{before[1]} -> {wide.shape[0]}x{wide.shape[1]} "
        f"(dropped {before[0] - wide.shape[0]} models, {before[1] - wide.shape[1]} items)"
    )

    Y = wide.to_numpy(dtype=float)
    mask = ~np.isnan(Y)
    density = mask.mean()
    print(f"observed cells: {density:.1%}")
    return Y, mask, wide.index.to_numpy(), wide.columns.to_numpy()


def raw_leaderboard(df: pd.DataFrame, category: str | None = None) -> pd.Series:
    """The published-style leaderboard: plain mean score per model.

    This is the thing your IRT ability estimates get compared against.
    """
    if category is not None:
        df = df[df["category"] == category]
    return df.groupby("model")["score"].mean().sort_values(ascending=False)
