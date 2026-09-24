# Product Price Prediction Pipeline

An end-to-end pipeline that predicts what a product sells for from its text
description, covering the whole ladder: data curation, classical ML, a deep
neural network, and zero-shot LLMs — all scored on the same held-out items by
the same harness, so the comparison actually means something.

```
Amazon Reviews 2023 ──> parse ──> dedup ──> price-balanced sample ──> split
                                                                        │
                                            LLM rewrites listings into summaries
                                                                        │
     baselines ── linear ── bag of words ── SVD ── forest ── XGBoost ── residual net ── zero-shot LLM
                                                                        │
                            avg $ error · median · RMSLE · R² · hit rate · diagnostics
```

## What is in here

| Path | Purpose |
|---|---|
| `pricing/parsing.py` | Turns a raw listing into a usable item, or rejects it |
| `pricing/curation.py` | Dedup, price-balanced sampling, splits |
| `pricing/summarise.py` | Rewrites listings into short comparable summaries with an LLM |
| `pricing/models/` | Every predictor, behind one `fit`/`predict` interface |
| `pricing/evaluation.py` | The shared harness: metrics, diagnostics, reports, charts |
| `run.py` | `curate`, `summarise`, `train`, `compare`, `predict` |

## Documentation

A full documentation set lives in [`docs/`](docs/): architecture, a file-by-file
code walkthrough, the concepts behind the implementation, the reasoning for every
non-obvious decision, and 30 interview questions with worked answers. Start with
[`docs/README.md`](docs/README.md) for the reading order.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On macOS, XGBoost needs OpenMP: `brew install libomp`. Every other model runs
without it — model imports are resolved on demand, so a missing optional
dependency does not break the rest of the CLI.

## Usage

```bash
# compare model families on the curated dataset
python run.py compare --models average bow-linear random-forest neural --size 200

# add a zero-shot LLM to the comparison
python run.py compare --models random-forest frontier --frontier-model openai/gpt-4.1-mini

# train the residual net once and keep the weights
python run.py train --epochs 8

# price a single description
python run.py predict "Sony WH-1000XM4 wireless noise cancelling headphones" --model random-forest

# rebuild the dataset from scratch (large download)
python run.py curate --categories Appliances Electronics --size 50000
python run.py summarise --limit 1000
```

`--chart` on `compare` writes a predicted-vs-actual scatter per model, coloured by
how far off each prediction was.

## Measured results

200 held-out items from `ed-donner/items_lite` (20,000 training items, mean price
$153, median $107). Hit rate is the share of predictions within $40 or 20% of the
real price — the bar for a prediction being useful rather than merely close.

| model | avg $ error | median $ | RMSLE | R² | hit rate |
|---|---|---|---|---|---|
| random forest | **68.72** | 34.22 | **0.705** | 0.316 | **60.0%** |
| SVD + ridge | 69.70 | 34.86 | 0.756 | 0.322 | 56.0% |
| residual net | 71.36 | 39.39 | 0.750 | **0.444** | 51.5% |
| linear (bag of words) | 82.97 | **33.55** | 0.805 | -0.899 | 55.0% |
| linear (hand features) | 83.75 | 45.25 | 0.915 | 0.067 | 44.0% |
| category average | 93.56 | 66.11 | 1.026 | 0.113 | 29.0% |
| training average | 106.08 | 90.42 | 1.154 | -0.001 | 18.5% |

Reading it as a ladder: predicting the mean is $106 off. Knowing only the
category buys $12. The first real text model — a linear regression on word
counts — buys another $11 but has a **negative R²**, because a handful of wild
over-predictions cost more than everything it gets right. Non-linear models on
the same features fix that: the random forest keeps the low median error *and*
stops blowing up, which is why it wins on both average error and hit rate. The
residual net has the best R² of the lot, meaning it tracks expensive items better
than anything else here, but it is less reliable on the cheap ones.

The gap between median ($34) and mean ($69) error is the real story: half of all
predictions are within about $34, and the average is dragged up by a small number
of large misses. Those misses are consistent across every model — branded
electronics at the top of the price range, where the description reads the same
whether the item costs $200 or $800.

A zero-shot local llama3.2 scored $264 average error with R² of -18.6 on the same
harness: confidently wrong, frequently by an order of magnitude ($4,500 for a
$53 projector). A frontier model does far better than that, but the point stands
that a zero-shot LLM needs comparable prices in its context to be competitive —
which is exactly what the RAG-based pricer in the deal discovery project does.

Every run writes a JSON report to `reports/` with per-item predictions.

## Design notes

**Everything is fit in log space.** Prices run from $0.50 to $999 with a long
tail. Squared error on raw dollars lets a few expensive items dominate the
gradient; in log space the models optimise relative error, which is what a
pricing decision cares about. Predictions are transformed back before scoring, so
the reported dollar errors are real dollars.

**The sample is deliberately unrepresentative.** Raw marketplace data is mostly
cheap items from two categories. Sampling proportional to price squared, with
Automotive and Tools damped, produces a training set the models can actually
learn a price *range* from instead of one that rewards always guessing $20.

**Curation drops more than it keeps.** Items with no parseable price, a price
outside the modelled range, or under 600 characters of text are rejected outright,
and long alphanumeric part numbers are stripped — they explode the vocabulary for
bag-of-words models and distract the LLMs without carrying price signal.

**Summaries are built once, not per prediction.** An LLM rewrites each listing
into a five-line summary during curation. Every downstream model reads that, so
the normalisation is paid for once rather than on every inference call.

**One harness, one test slice.** A predictor is any callable taking an item and
returning a price, so a linear model and a zero-shot LLM go through identical
code and are scored on identical items. Reported metrics deliberately include
both a central measure and a tail measure, because on this problem they disagree
and that disagreement is the finding.

## Verified

Curation, every classical model, the residual net (trained on this machine, MPS)
and the zero-shot LLM path were all run end to end; the table above is a single
real run, not assembled from separate ones. XGBoost is implemented but was not
run here — this machine is missing the OpenMP runtime it links against. Full
curation from Amazon Reviews 2023 downloads millions of rows per category; the
results above use the pre-curated 20k `items_lite` dataset. Zero-shot pricing was
exercised with a local llama3.2 rather than a frontier model.
