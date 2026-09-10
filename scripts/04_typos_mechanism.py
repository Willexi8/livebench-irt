"""Does correction ambiguity explain why `typos` questions carry no signal?

Hypothesis. The `typos` task plants misspellings and scores an exact match on
the whole restored text. If a planted misspelling has more than one plausible
correction -- `hten` could be `then` or `the` -- then models that pick the other
reasonable reading score zero for a reason unrelated to ability. Questions with
many such corrections would look uninformative however capable the models are.

Method. Align the corrupted text against the ground truth word by word, pull
out each correction, and score its ambiguity as the number of distinct
vocabulary words sitting at the same (minimal) edit distance from the corrupted
token. Ambiguity 1 means the correction is forced; 2 or more means a model
could reasonably have gone elsewhere.

Vocabulary is built from the ground-truth texts themselves, so this understates
ambiguity: a plausible alternative that never appears in any answer is invisible
here. Treat the resulting counts as a lower bound.
"""

from __future__ import annotations

import difflib
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"[A-Za-z]+")


def tokenize(text):
    return WORD.findall(text)


def edit_distance(a, b, cap=3):
    """Levenshtein distance, cut off at `cap` for speed."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for jj, cb in enumerate(b, 1):
            cur.append(min(prev[jj] + 1, cur[jj - 1] + 1, prev[jj - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def extract_corrections(corrupt_text, truth_text):
    """Word-level diff, split into two kinds of correction.

    A 1:1 replacement is a misspelt word with a single intended target, and its
    ambiguity is measurable. An n:m block is a word-boundary error -- a missing
    or spurious space, as in `introducehten` -> `introduce the` -- where the
    model has to re-segment the text rather than pick a spelling. These are a
    different failure mode and are counted separately rather than mixed in.
    """
    a, b = tokenize(corrupt_text), tokenize(truth_text)
    subs, boundary = [], []
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        src, dst = a[i1:i2], b[j1:j2]
        if not src or not dst:
            boundary.append((" ".join(src), " ".join(dst)))
        elif len(src) == len(dst):
            subs.extend(zip(src, dst))
        else:
            boundary.append((" ".join(src), " ".join(dst)))
    return subs, boundary


def ambiguity(token, vocab_by_len, cap=2):
    """How many vocabulary words tie for closest to this corrupted token."""
    token = token.lower()
    best, hits = cap + 1, 0
    for L in range(max(1, len(token) - cap), len(token) + cap + 1):
        for w in vocab_by_len.get(L, ()):
            if w == token:
                continue
            d = edit_distance(token, w, cap=cap)
            if d < best:
                best, hits = d, 1
            elif d == best and d <= cap:
                hits += 1
    return (best, hits if best <= cap else 0)


def _strip_instructions(prompt):
    """Keep only the corrupted passage, not the instruction that precedes it.

    Leaving the instruction in would diff as a spurious insertion at the start
    of every question and inflate the boundary-error count uniformly.
    """
    if "-------" in prompt:
        return prompt.split("-------")[-1]
    parts = [p for p in prompt.split("\n\n") if p.strip()]
    return parts[-1] if parts else prompt


def main():
    from datasets import load_dataset

    weak = pd.read_csv(ROOT / "weak_questions.csv")
    unsure = pd.read_csv(ROOT / "inconclusive_questions.csv")
    flagged = set(weak.question_id) | set(unsure.question_id)

    ds = load_dataset("livebench/language", split="test").to_pandas()
    t = ds[ds.task == "typos"].copy()
    print(f"typos questions with recoverable text: {len(t)}")

    vocab = Counter()
    for txt in t.ground_truth:
        vocab.update(w.lower() for w in tokenize(txt))
    vocab_by_len = {}
    for w in vocab:
        vocab_by_len.setdefault(len(w), []).append(w)
    print(f"vocabulary size: {len(vocab)}")

    rows = []
    for _, r in t.iterrows():
        prompt = r.turns[0] if isinstance(r.turns, np.ndarray) else r.turns
        corrupt = _strip_instructions(prompt)
        subs, boundary = extract_corrections(corrupt, r.ground_truth)
        amb = [ambiguity(src, vocab_by_len) for src, _ in subs]
        n_amb = sum(1 for best, hits in amb if hits >= 2)
        rows.append(
            {
                "question_id": r.question_id,
                "flagged": r.question_id in flagged,
                "n_subs": len(subs),
                "n_boundary": len(boundary),
                "n_corrections": len(subs) + len(boundary),
                "n_ambiguous": n_amb,
                "frac_ambiguous": n_amb / len(subs) if subs else np.nan,
                "chars": len(r.ground_truth),
            }
        )

    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "typos_corrections.csv", index=False)

    print()
    print(
        d.groupby("flagged")[
            ["n_corrections", "n_subs", "n_boundary", "n_ambiguous", "frac_ambiguous"]
        ]
        .agg(["size", "mean"])
        .round(3)
        .to_string()
    )

    from scipy.stats import mannwhitneyu

    print()
    for col in ("n_corrections", "n_subs", "n_boundary", "n_ambiguous", "frac_ambiguous"):
        x = d.loc[d.flagged, col].dropna()
        y = d.loc[~d.flagged, col].dropna()
        if len(x) > 2 and len(y) > 2:
            u, p = mannwhitneyu(x, y)
            print(f"{col:16s} flagged {x.mean():7.3f}  unflagged {y.mean():7.3f}  p={p:.4f}")


if __name__ == "__main__":
    main()
