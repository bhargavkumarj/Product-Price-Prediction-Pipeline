# Repository Map

Every folder and every file in this repository.

```
product-price-prediction/
├── README.md                     project-facing readme (setup, usage, results)
├── requirements.txt              12 dependencies
├── .env.example                  every environment variable, documented
├── .gitignore                    excludes .env, artifacts/, reports/, data/curated/, *.pt
│
├── run.py                        THE ENTRY POINT — five subcommands
│
├── pricing/
│   ├── __init__.py               re-exports Item and the settings singleton
│   ├── config.py                 paths, dataset names, model names, seed, CATEGORIES
│   ├── items.py                  the Item model and Hub round-tripping
│   ├── parsing.py                one raw Amazon row -> Item, or None
│   ├── loaders.py                streams a category off the Hub, parses in processes
│   ├── curation.py               deduplicate, price-balance, split, describe
│   ├── summarise.py              LLM rewriting of listings, with cost tracking
│   ├── dataset.py                local JSONL save/load, and the unified loader
│   ├── evaluation.py             THE SHARED HARNESS — metrics, table, diagnostics, report, chart
│   └── models/
│       ├── __init__.py           lazy registry: name -> (module, class)
│       ├── base.py               the Pricer contract
│       ├── baselines.py          random, training average, category average
│       ├── classical.py          features, bag of words, SVD+ridge, random forest, XGBoost
│       ├── neural.py             residual MLP in PyTorch
│       └── frontier.py           zero-shot LLM via LiteLLM
│
└── (generated, git-ignored)
    ├── data/curated/             train.jsonl, validation.jsonl, test.jsonl
    ├── artifacts/residual_net.pt trained weights + target scaler
    └── reports/                  timestamped JSON reports, predicted_vs_actual.html
```

## What each file is responsible for

### Entry point

**`run.py`** — Five subcommands, each a `cmd_*` function wired through
`set_defaults(func=...)`:

| Command | Does | Cost |
|---|---|---|
| `curate` | Builds the dataset from raw Amazon data | Hours, large download |
| `summarise` | Adds LLM-written summaries to a curated dataset | Money per item |
| `compare` | Fits and scores several models on the same items | Seconds to minutes |
| `train` | Trains the residual net and saves weights | Minutes |
| `predict` | Prices one description from the command line | Seconds after fit |

Heavy imports (`curation`, `summarise`, `scatter`) are deferred into the
functions that need them, so `run.py --help` does not import torch.

### Data layer

**`pricing/config.py`** — Paths derived from `ROOT`, dataset names, model names,
the seed, and the eight `CATEGORIES`. `__post_init__` creates `data/`,
`artifacts/` and `reports/` so no other module needs `mkdir` guards.

**`pricing/items.py`** — The `Item` pydantic model. Carries the raw text
(`full`), the LLM summary (`summary`), the price, the category, a weight and an
id. The `text` property is the single place that decides which text a model
sees. Also handles HuggingFace Hub round-tripping.

**`pricing/parsing.py`** — Pure functions turning one raw row into an `Item` or
`None`. All the rejection rules and text cleaning live here. No I/O, no state —
which is what lets it run inside worker processes.

**`pricing/loaders.py`** — `CategoryLoader` streams one Amazon category and farms
parsing out to a process pool in 1,000-row chunks.

**`pricing/curation.py`** — The dataset-shaping logic: `deduplicate`, `balance`
(the price-squared weighted sample), `split`, `describe`, and `curate` which
composes them.

**`pricing/summarise.py`** — `Summariser` wraps a LiteLLM call, tracks input
tokens, output tokens and cumulative cost, and applies itself across a list with
a thread pool. Skips items that already have a summary, so it is resumable.

**`pricing/dataset.py`** — `save_local`, `load_local` and `load`, which accepts
either a HuggingFace dataset name or a local folder. This is what makes the
project work with or without a Hub account.

### Model layer

**`pricing/models/base.py`** — The `Pricer` contract in 17 lines.

**`pricing/models/__init__.py`** — The registry. Maps a CLI name to
`(module_path, class_name)` strings and resolves them with `import_module` only
when asked. This is why a missing `libomp` breaks `xgboost` and nothing else.

**`pricing/models/baselines.py`** — `RandomPricer`, `AveragePricer`,
`CategoryAveragePricer`. The floor: anything that cannot beat these is not
learning from the text.

**`pricing/models/classical.py`** — Five models plus the shared `to_log`/
`from_log` helpers and `MAX_TRAIN_ROWS`.

**`pricing/models/neural.py`** — `ResidualBlock`, `PriceNet`, `NeuralPricer`, and
`best_device()`. Includes `save`/`load` so the trained net can be reused — the
deal discovery project loads exactly this artefact.

**`pricing/models/frontier.py`** — `FrontierPricer`. Zero training, one prompt,
any LiteLLM-supported provider.

### Evaluation layer

**`pricing/evaluation.py`** — The most reused file in the project. `to_price`
(normalising any predictor's output), `band` (the green/amber/red rule),
`Prediction` and `Result` (metrics as lazy properties), `evaluate` (the threaded
runner), `table`, `diagnostics`, `save` and `scatter`.

## Dependency direction

```
            config.py  ◄── everything
                ▲
             items.py
                ▲
    ┌───────────┼──────────────┬─────────────┐
 parsing     dataset        evaluation    models/base
    ▲                           ▲             ▲
 loaders                        │     ┌───────┼────────┬────────┐
    ▲                           │  baselines classical neural frontier
 curation                       │             ▲
    ▲                           │        models/__init__ (lazy)
    └──────────── run.py ───────┴─────────────┘
```

`evaluation.py` imports only `config` and `items` — it knows nothing about any
model, which is exactly why every model can go through it.

`models/__init__.py` imports **no** model module at import time. It holds
strings and resolves them on demand.

## External dependencies and why each is there

| Package | Why | Optional? |
|---|---|---|
| `datasets` | Load Amazon Reviews and the curated Hub dataset | No |
| `huggingface-hub` | Pushing a curated dataset | Only for `--push-to` |
| `pydantic` | The `Item` model, validation on load | No |
| `numpy` | Weighted sampling, array maths | No |
| `pandas` | `FeaturePricer` builds a DataFrame for sklearn | No |
| `scikit-learn` | Vectorisers, linear models, forest, SVD, `r2_score` | No |
| `xgboost` | Gradient-boosted trees | Yes — lazily imported |
| `torch` | The residual network | Yes — lazily imported |
| `litellm` | Zero-shot LLM pricing and summarisation | Yes — lazily imported |
| `plotly` | `--chart` scatter plots | Yes — imported inside `scatter` |
| `python-dotenv` | Loads `.env` | No |
| `tqdm` | Progress bars | No |

Four of the twelve are effectively optional because of the lazy registry. On this
machine `xgboost` cannot load — it links against OpenMP, which is missing — and
every other model still runs. That is not an accident; it is what the registry
design buys.
