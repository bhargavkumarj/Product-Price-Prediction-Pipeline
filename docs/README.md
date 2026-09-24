# Product Price Prediction Pipeline — Complete Documentation

Everything about this project: the data
pipeline, every model family, the shared evaluation harness, the reasoning behind
each choice, and the interview questions it invites.

## Read in this order

| Document | What it covers |
|---|---|
| [01-architecture.md](01-architecture.md) | The pipeline end to end, and the lifecycle of one prediction |
| [02-repository-map.md](02-repository-map.md) | Every folder and file, with its role and dependencies |
| [03-code-walkthrough-data.md](03-code-walkthrough-data.md) | `config.py`, `items.py`, `parsing.py`, `loaders.py`, `curation.py`, `summarise.py`, `dataset.py` |
| [04-code-walkthrough-models.md](04-code-walkthrough-models.md) | `base.py`, the registry, baselines, classical models, the residual net, the zero-shot LLM |
| [05-code-walkthrough-evaluation.md](05-code-walkthrough-evaluation.md) | `evaluation.py` and `run.py` |
| [06-concepts.md](06-concepts.md) | Log-space regression, every metric with worked arithmetic, bag of words, TF-IDF, SVD, trees, residual networks, the hashing trick |
| [07-design-decisions.md](07-design-decisions.md) | Every non-obvious choice, rejected alternatives, and bugs found while running it |
| [08-interview-questions.md](08-interview-questions.md) | 30 questions with full answers |

## The project in one paragraph

An end-to-end pipeline that predicts what a product sells for from its text
description. It spans the whole ladder: curating a dataset from raw Amazon
listings (parse, clean, deduplicate, price-balance, split), rewriting listings
into normalised summaries with an LLM, then training and comparing six model
families — trivial baselines, hand-built features, bag of words, SVD, random
forest, XGBoost, a residual neural network, and a zero-shot LLM. Every model goes
through one evaluation harness on the same held-out items, so the comparison is
real.

## The headline result

200 held-out items from `ed-donner/items_lite` (20,000 training items, mean price
$153, median $107), measured in a single run on this machine:

| model | avg $ error | median $ | RMSLE | R² | hit rate |
|---|---|---|---|---|---|
| random forest | **68.72** | 34.22 | **0.705** | 0.316 | **60.0%** |
| SVD + ridge | 69.70 | 34.86 | 0.756 | 0.322 | 56.0% |
| residual net | 71.36 | 39.39 | 0.750 | **0.444** | 51.5% |
| linear (bag of words) | 82.97 | **33.55** | 0.805 | -0.899 | 55.0% |
| linear (hand features) | 83.75 | 45.25 | 0.915 | 0.067 | 44.0% |
| category average | 93.56 | 66.11 | 1.026 | 0.113 | 29.0% |
| training average | 106.08 | 90.42 | 1.154 | -0.001 | 18.5% |

Two things in this table are worth understanding properly, and both are covered
in detail in [06-concepts.md](06-concepts.md):

1. **Bag-of-words linear regression has the best median error and a negative
   R².** It is the single most instructive row in the project.
2. **Average error is double the median error for every model.** That gap is the
   real finding, not the headline number.
