# Architecture

## The problem

Given a product's text description, predict its price in dollars. Supervised
regression on text, with three properties that shape every design decision:

1. **The target is heavy-tailed.** Prices run $0.50 to $999 with most mass at the
   cheap end. Mean $153, median $107 on the curated set.
2. **The input is long and noisy.** Raw listings are up to 4,000 characters of
   marketing copy, spec tables, shipping notes and part numbers.
3. **The raw data is badly distributed.** Most listings are cheap items in two
   categories. A model trained on the raw distribution learns to always guess
   about $20.

Most of the engineering is in the data, not the models.

## The pipeline

```
McAuley-Lab/Amazon-Reviews-2023          millions of rows per category
        │
        ▼  CategoryLoader.load()          process pool, 1000-row chunks
   parse(row, category)                   reject: no price, out of range,
        │                                 under 600 chars of text
        ▼
   clean()                                drop catalogue noise, strip part
        │                                 numbers, cap at 4000 chars
        ▼
   deduplicate()                          unique titles, then unique bodies
        │
        ▼
   balance(size)                          sample ∝ price², Automotive ×0.05,
        │                                 Tools ×0.5  → flatter price curve
        ▼
   split()                                train / validation / test
        │
        ▼
   Summariser.apply()                     LLM rewrites each listing into a
        │                                 normalised 5-line summary
        ▼
   train / validation / test  ──────────────────────────────────┐
        │                                                       │
        ▼ fit()                                                 ▼ evaluate()
   ┌─────────────────────────────────────────┐        ┌──────────────────────┐
   │ random · average · category-average     │        │ avg $ error          │
   │ features · bag-of-words · SVD+ridge     │───────►│ median $ error       │
   │ random forest · XGBoost                 │        │ RMSLE                │
   │ residual net (PyTorch)                  │        │ R²                   │
   │ zero-shot LLM (LiteLLM)                 │        │ hit rate             │
   └─────────────────────────────────────────┘        │ worst-miss diagnostic│
                                                      └──────────────────────┘
                                                               │
                                                  table + JSON report + chart
```

## The three stages, and why they are separate

### Stage 1 — curation (`run.py curate`)

Expensive and run rarely. Downloads millions of rows, parses them in a process
pool, and reduces them to a balanced, deduplicated, split dataset written as
JSONL or pushed to the HuggingFace Hub.

Separate because it takes hours and its output is stable. Nobody should re-parse
Amazon to try a different learning rate.

### Stage 2 — summarisation (`run.py summarise`)

An LLM rewrites every listing into five normalised lines:

```
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features
```

Separate because it costs money per item and, crucially, is **paid once**. Every
downstream model reads `item.summary`. If summarisation happened at prediction
time, every model would pay an LLM call per prediction — which would make the
bag-of-words baseline more expensive than the zero-shot LLM it is meant to be a
cheap alternative to.

### Stage 3 — modelling and evaluation (`run.py compare` / `train` / `predict`)

Fast and run constantly. Loads the curated dataset, fits the requested models,
scores them all on the same held-out items, prints a table and saves a report.

## Lifecycle of one prediction

Take `random-forest` predicting on a held-out item.

**Fit time** (once, in `cmd_compare`):

1. `dataset.load()` pulls train/validation/test from the Hub or a local folder.
2. `create("random-forest")` resolves the registry entry and constructs the
   `RandomForestPricer`, which builds a `CountVectorizer` but fits nothing yet.
3. `model.fit(train, validation)` takes the first 20,000 items, fits the
   vectoriser on their text, transforms to a sparse matrix, and fits 100 trees
   against `log1p(price)`.

**Predict time** (once per item):

4. `evaluate(model, test, size=200)` takes the first 200 test items and maps
   `run` over them in a thread pool.
5. `run` calls `predictor(item)` — `Pricer.__call__` forwards to `predict`.
6. `predict` transforms `item.text` with the **already-fitted** vectoriser, calls
   `model.predict`, and inverts the log with `expm1`, clipped at $0.50.
7. `to_price` normalises the return — a float passes through; a string like
   `"$249.99"` has its symbol and commas stripped and a number regexed out.
8. A `Prediction(title, truth, guess)` is built; `error`, `squared_log_error` and
   `band` are computed lazily as properties.

**Aggregation:**

9. `Result` computes average error, median error, RMSLE, R² and hit rate from the
   prediction list.
10. `diagnostics` prints the green/amber/red split and the five worst misses.
11. `table` sorts all models by average error and prints an aligned comparison.
12. `save` writes a JSON report including every individual prediction.

## The key abstraction

```python
class Pricer:
    display_name = "pricer"
    needs_training = True

    def fit(self, train, validation=None) -> "Pricer": return self
    def predict(self, item: Item) -> float: raise NotImplementedError
    def __call__(self, item: Item) -> float: return self.predict(item)
```

That is the entire contract. A linear regression, a 13-million-parameter neural
network and an HTTP call to a frontier LLM all satisfy it, which is what makes
the comparison table meaningful — every row went through identical evaluation
code on identical items.

`__call__` matters: it means `evaluate()` takes any callable, so a bare function
works too and the harness has no idea what a `Pricer` is.

`needs_training` lets `cmd_compare` skip `fit` for `random` and `frontier`
without special-casing their names.

## Where the extension points are

| You want to | Change |
|---|---|
| Add a model | Subclass `Pricer`, add one line to `REGISTRY` |
| Add a metric | Add a property to `Result`, a column to `table()` |
| Use a different LLM | `--frontier-model anthropic/claude-...` (any LiteLLM string) |
| Use your own data | `run.py curate`, or point `--dataset` at a folder of JSONL |
| Change the target transform | `to_log`/`from_log` in `classical.py` |

## Concurrency model

Two different mechanisms, chosen for two different bottlenecks:

- **`CategoryLoader` uses processes.** Parsing millions of JSON rows is CPU
  bound, and the GIL would serialise threads. `ProcessPoolExecutor` with
  1,000-row chunks gives each worker meaningful work per IPC round trip.
- **`evaluate` and `Summariser` use threads.** Both are waiting on something —
  an HTTP call for the LLM paths, or native numpy/torch code that releases the
  GIL for the local models. Threads are cheap and share the fitted model, which
  processes could not do without pickling it per worker.
