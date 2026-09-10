"""Download the LiveBench judgments once and cache them under data/.

    python scripts/01_download.py
    python scripts/01_download.py --datasets   # fallback via the datasets lib
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livebench_irt.load import load_judgments  # noqa: E402

if __name__ == "__main__":
    df = load_judgments(use_datasets="--datasets" in sys.argv)
    print(df.shape)
    print(df.head())
    print("\nmodels:", df["model"].nunique())
    print("questions:", df["question_id"].nunique())
    print("categories:", sorted(df["category"].unique()))
    print("\nscore distribution:")
    print(df["score"].describe())
