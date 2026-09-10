# livebench-irt

A measurement model for [LiveBench](https://livebench.ai/).

LiveBench publishes a leaderboard of average scores. This repository fits an
item response theory (IRT) model to the same per-question judgments and asks
three questions the average score cannot answer:

1. **How much of the leaderboard ordering is real?** Fitting ability with
   bootstrap intervals over questions shows which adjacent ranks are
   distinguishable and which are an artefact of which questions happened to be
   drawn.
2. **Which questions are not measuring anything?** A question whose estimated
   discrimination is near zero — everyone gets it right, everyone gets it
   wrong, or right and wrong answers are unrelated to model ability —
   contributes noise and nothing else.
3. **(next) When the question set is replaced each month, are the scores still
   comparable?** LiveBench rotates questions to resist contamination. That
   design makes month-to-month score changes confounded: a drop can mean the
   model got worse or the questions got harder. Test equating is the standard
   fix and has not, as far as I can tell, been applied here.

## The model

For model $j$ and question $i$:

$$P(\text{correct}) = \sigma\big(a_i(\theta_j - b_i)\big)$$

- $\theta_j$ — model ability, the quantity a leaderboard is trying to report
- $b_i$ — question difficulty
- $a_i$ — question discrimination; $a_i \approx 0$ means the question is not
  measuring ability

Fitted by joint maximum likelihood with a ridge penalty (`scipy` L-BFGS),
identified by standardising $\theta$. Scores are floats in $[0,1]$, so the
Bernoulli log-likelihood is evaluated at fractional outcomes.

## Quickstart

```bash
git clone https://github.com/<you>/livebench-irt.git
cd livebench-irt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/00_smoke_test.py    # synthetic data with known parameters
python scripts/02_fit_irt.py       # ~4 min: fit + 200 bootstrap refits
```

Once that works, delete `data/model_judgment.parquet`, then:

```bash
python scripts/01_download.py      # pulls livebench/model_judgment from HF
python scripts/02_fit_irt.py       # all categories
python scripts/02_fit_irt.py math  # one category
```

Outputs:

| file | contents |
|---|---|
| `figures/leaderboard.png` | ability with 95% bootstrap intervals |
| `figures/items.png` | every question as a (difficulty, discrimination) point |
| `leaderboard.csv` | ability, rank interval, and published mean score per model |
| `bad_questions.csv` | questions with discrimination below threshold |

## Tests

```bash
python tests/test_irt.py
```

Simulates from a known 2PL and checks the fitter recovers it: ability
correlation > 0.95, difficulty > 0.90, and at least 8 of 10 planted zero-
discrimination questions flagged.

## Known limitations

- **Joint MLE is inconsistent in the item parameters.** Each question's
  parameters are estimated from as many observations as there are models, so
  this is the classic incidental-parameters setting. Measured recovery of
  difficulty on simulated data: 0.81 at 60 models, 0.94 at 120, 0.96 at 300.
  Ability is much better behaved. Marginal MLE — integrating $\theta$ out — is
  the first thing to replace here.
- **Unidimensionality is assumed and is probably wrong.** Coding and language
  ability are not the same latent trait. A per-category fit is a crude check;
  a multidimensional model is the honest version.
- **Local independence is assumed.** Questions generated from the same source
  article or dataset are not independent given ability.
- **`a_i ≈ 0` has several possible causes** — a broken question, a mislabelled
  ground truth, a question all current models saturate, or one none can do.
  Separating those requires reading the questions, not just the scores.

## Data

`livebench/model_judgment` on Hugging Face (Apache 2.0), one row per
(question, model, turn) with an objective 0–1 score. LiveBench paper:
[arXiv:2406.19314](https://arxiv.org/abs/2406.19314).

## Status

Work in progress. Steps 1 and 2 above are implemented; equating is not.
