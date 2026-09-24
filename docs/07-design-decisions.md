# Design Decisions

Every non-obvious choice, the alternatives rejected, and the bugs found by
running the code.

---

## Decision 1 — Every model fits in log space

**Choice.** `to_log` / `from_log` wrap every classical model, and the neural net
applies `log1p` plus standardisation.

**Rejected.** Fitting raw dollars.

**Why.** Prices span $0.50–$999 with a long right tail. Squared error on raw
dollars treats a $100 miss identically at $120 and at $900, and lets the
expensive tail dominate the gradient — the model spends its capacity on a few
hundred items and gets the bulk of the distribution wrong. In log space, equal
errors are equal ratios, which matches both how the data is distributed
(roughly log-normal) and what a pricing decision cares about.

**The detail that matters.** `log1p`/`expm1` rather than `log`/`exp` with a
manual `±1`. `log1p` is precise for small `x`, and most of this dataset is small
`x`.

---

## Decision 2 — Price-weighted sampling that is deliberately unrepresentative

**Choice.** Sample proportional to scaled-price squared, with Automotive at 5%
and Tools at 50%.

**Rejected.** A uniform random sample, which preserves the real distribution.

**Why.** Raw marketplace data is mostly cheap items in two categories. A model
trained on that learns the base rate — always guess about $20 — which minimises
training error and is useless as a pricer. Squaring the scaled price makes a $900
item roughly 320× more likely to be sampled than a $50 one, giving a flatter
target the model can learn a *range* from.

**The defensible part.** This is a training-distribution choice, not a test-set
choice. Train and test come from the same balanced pool, so the evaluation is
internally consistent. If the deployment distribution were the raw Amazon mix,
you would want to either re-weight at evaluation time or report per-price-bucket
metrics — and that is the honest caveat to state.

---

## Decision 3 — The final shuffle in `balance()`

**The bug this prevents.** `np.random.choice` returns indices in **ascending
order**. `split()` then slices the list into train/validation/test.

Without the final shuffle, the cheapest items would go to train and the most
expensive to test. Every model would appear unable to predict expensive items,
and the real cause — that it had never seen any — would be invisible in every
metric.

This is the most dangerous class of bug in an ML pipeline: it produces plausible,
consistently bad results with no error anywhere.

---

## Decision 4 — Shuffle before deduplicating

**Choice.** `deduplicate` shuffles with a seeded generator, then keeps the first
occurrence of each title and each body.

**Why.** Deduplication keeps the *first* occurrence. Without shuffling, "first"
means "first in catalogue order", which correlates with category and listing age
— so the survivor of every duplicate group is systematically biased.

**Two passes, not one.** Identical titles are the same product listed twice.
Identical bodies are the same product under different titles. Both matter,
because a duplicate spanning train and test is a label leak that inflates the
test score.

---

## Decision 5 — `parse()` returns `None` instead of raising

**Choice.** Unusable rows return `None`; the caller filters.

**Why.** Rejection is the *normal* outcome — most raw rows are rejected. Using
exceptions for the common path would mean try/except around every row and would
conflate "this row is not useful" with "something is broken".

It also keeps `parsing.py` free of side effects, which is a hard requirement: it
runs inside `ProcessPoolExecutor` workers.

---

## Decision 6 — Processes for parsing, threads for everything else

**Parsing uses processes.** JSON decoding and regex over millions of rows is CPU
bound; the GIL would serialise threads and give no speedup.

**Evaluation and summarisation use threads.** Both wait — on HTTP for the LLM
paths, or on native numpy/torch code that releases the GIL for the local models.
Threads also share the fitted model, which processes could not do without
pickling a 13-million-parameter network per worker.

**The chunk size follows from the choice.** 1,000 rows per task, because
dispatching single rows to a process pool spends all its time on IPC.

---

## Decision 7 — The `Pricer` contract, and `__call__` forwarding

**Choice.** A 17-line base class: `fit`, `predict`, `__call__`, plus
`display_name` and `needs_training`.

**Why `__call__` matters.** `evaluate()` takes any callable, so it never needs to
know what a `Pricer` is. A bare function or a third-party estimator works
unchanged. That is what makes the harness genuinely model-agnostic rather than
model-agnostic-in-principle.

**Why `needs_training` is a flag, not a name check.** `cmd_compare` writes
`if model.needs_training: model.fit(...)`. Adding an untrained model needs no
change to the CLI. Special-casing `"random"` and `"frontier"` by name would.

---

## Decision 8 — The lazy model registry

**The bug that forced it.** The first version imported every model class at the
top of `models/__init__.py`. Running `python run.py compare --models average`
produced:

```
xgboost.core.XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded.
Likely causes: OpenMP runtime is not installed
```

The entire CLI was dead — including `--help` — because one optional model's
native dependency was missing.

**The fix.** The registry holds `(module_path, class_name)` **strings** and
resolves them with `import_module` only when asked. `XGBoostPricer.fit` also
imports xgboost inside the method, as a second line of defence.

**What it bought.** The published results table was produced on a machine where
XGBoost genuinely cannot load. Nothing else was affected, and the README states
plainly that XGBoost was not run rather than filling in a plausible number.

**The general principle.** Optional dependencies should fail at the point of use,
not at import. A tool that dies on `--help` because of a model you did not ask
for is broken.

---

## Decision 9 — Five metrics, not one

**Rejected.** A single score.

**Why rejected.** The measured table is unreadable with one number:

```
linear (bag of words)   avg $82.97   median $33.55   RMSLE 0.805   R² -0.899   hit 55.0%
```

**Best median error in the comparison, and R² worse than predicting the mean.**
Both true. Pick either as *the* metric and you get a completely different, and
wrong, conclusion about this model.

The five answer different questions:

| Metric | Question | Sensitive to |
|---|---|---|
| average error | How wrong on average? | The tail |
| median error | How wrong typically? | The centre |
| RMSLE | How wrong proportionally? | Ratios, scale-free |
| R² | Better than guessing the mean? | The tail, heavily |
| hit rate | How often is it usable? | A business threshold |

The gap between average ($68.72) and median ($34.22) is itself the finding: half
of all predictions are within $34, and the average is inflated by a minority of
large misses.

---

## Decision 10 — Hit rate uses two criteria joined by `or`

```python
if error < 40 or error / truth < 0.2:
    return "green"
```

**Why both.** A $5 error on a $20 item is 25% — fails relative, passes absolute,
and rightly so, because nobody cares about $5. A $60 error on a $900 item is
6.7% — fails absolute, passes relative, and rightly so, because that is a good
estimate.

Either rule alone mislabels one end of a price range spanning three orders of
magnitude. The dual rule mirrors how a person actually judges an estimate: close
in dollars *or* close in proportion.

---

## Decision 11 — `items[:size]`, not a random sample

**Choice.** `evaluate` scores the **first** N test items, deterministically.

**Why.** Every model must be scored on exactly the same items. A random sample
per model would introduce between-row variance that has nothing to do with the
models, and a $3 difference in average error could be sampling noise.

It is unbiased because the list was already shuffled during curation. Determinism
here is what makes the table a controlled comparison.

---

## Decision 12 — `to_price` lives in the harness, not in each model

**Choice.** `FrontierPricer.predict` returns a `str`; every other `predict`
returns a `float`; `to_price` in the harness normalises both.

**Why.** LLM output parsing is fiddly — strip `$`, strip commas, regex a number,
handle the decimal branch first so `"249.99"` does not truncate to `"249"`. Put
that in one place and every future LLM-backed predictor inherits it.

The type inconsistency is deliberate and documented in the code, not an
oversight.

---

## Decision 13 — Summarise once at curation, not per prediction

**Choice.** `run.py summarise` is a separate stage that writes `item.summary`.

**Why.** If summarisation happened at prediction time, the bag-of-words baseline
would cost an LLM call per prediction — making the cheap alternative more
expensive than the expensive one it is meant to be compared against.

**The second reason is comparability.** Every model reads the same normalised
text through `Item.text`. Without that, a comparison between models would be
partly a comparison of preprocessing choices.

**Resumability follows.** `if not item.summary` means an interrupted 400,000-item
run resumes where it stopped, which is not optional at that scale.

---

## Decision 14 — `Item.text` as a single fallback chain

```python
@property
def text(self) -> str:
    return self.summary or self.full or self.title
```

**Why it is a property, not a parameter.** If each model chose its own input
text, the results table would be comparing input choices as much as algorithms.
One property, one decision, applied everywhere.

It also makes the pipeline degrade gracefully: on `items_lite`, `full` is `None`
for every row but `summary` is populated, and every model works unchanged.

---

## Decision 15 — The neural net saves its target scaler

```python
torch.save({"state": self.model.state_dict(), "mean": self.mean, "std": self.std}, path)
```

**Why.** The network predicts a *standardised log price*. Without `mean` and
`std` the weights are useless — you would have a number with no way to convert it
to dollars.

Saving only `state_dict()` is a common and painful mistake, and it is silent: the
model loads, predicts, and returns numbers around 0.3 that look like nothing in
particular.

This artefact is the bridge between projects: the deal discovery system's
`NeuralAgent` loads this exact file.

---

## Decision 16 — The hashing trick for the neural net

**Choice.** `HashingVectorizer` rather than `CountVectorizer`.

**Trade-off accepted.** Hash collisions (different words share a bucket) and no
inverse mapping (not interpretable at the feature level).

**What it buys.** No vocabulary to fit, nothing to persist beyond the weights,
and fixed memory. The "nothing to persist" property is the valuable one: the deal
discovery project loads this model with three lines and a single `.pt` file,
because the vectoriser is reconstructed from a constant.

For a 13-million-parameter network, collisions act like mild feature noise and it
was never interpretable anyway. The costs are close to free; the benefit is real.

---

## Decision 17 — `SmoothL1Loss` rather than MSE

**Why.** MSE lets outliers dominate the gradient — which is *precisely* the
failure mode that gives the bag-of-words model R² of -0.899. Pure L1 has constant
gradient magnitude and converges poorly near the optimum. Huber is quadratic
where fine convergence matters and linear where robustness matters.

It is the same reasoning as the log transform, applied at the loss level: the
model should not be allowed to obsess over a handful of extreme items.

---

## Decision 18 — Report the worst misses, not just the scores

```python
def worst(self, n: int = 5) -> list[Prediction]:
    return sorted(self.predictions, key=lambda p: p.error, reverse=True)[:n]
```

**Why.** The same items appear in every model's worst-miss list:

```
Canon EOS Rebel T7 Digital SLR Camera   truth $819.00   guess $183.26
Dell XPS 13 9343-2727SLV                truth $839.99   guess $392.70
Bam France 2002XL Contoured Hightech    truth $713.00   guess $143.81
```

All expensive branded electronics, all under-predicted, by every model.

That is a **data** finding, not a model finding: at the top of the price range
the description reads much the same whether the item costs $200 or $800, because
price there is driven by brand equity and market position. No aggregate metric
would surface it. Trying a sixth model would not fix it; better features — brand
as an explicit field, category-conditional models — might.

---

## Decision 19 — Store every prediction in the report

**Choice.** The JSON report includes each `(title, truth, guess)`, not just
summary statistics.

**Why.** It makes reports re-analysable. You can compute a metric that did not
exist when the run happened, plot the error distribution, or compare two models
item by item to see whether they fail on the same products — which is how you
find out that an ensemble would help.

---

## Decision 20 — The `curate` / `summarise` / `compare` split

**Why three commands and not one pipeline.** They have completely different cost
profiles:

| Stage | Cost | Frequency |
|---|---|---|
| `curate` | Hours, large download | Once |
| `summarise` | Money per item | Once |
| `compare` | Seconds to minutes | Constantly |

Fusing them would mean re-parsing Amazon to try a different learning rate. The
boundary is drawn where the artefacts are stable, which is the same reasoning
that makes a build cache useful.

---

## Bugs found by running the code

| Bug | Symptom | Root cause | Fix |
|---|---|---|---|
| Eager xgboost import | Whole CLI dead, including `--help` | Registry imported every model class at import time | Lazy registry of `(module, class)` strings |
| Neural scaler device mismatch | `Expected all tensors to be on the same device, mps:0 and cpu` | `mean`/`std` saved as tensors, `map_location` moved them to MPS while the prediction was on CPU | Store the scaler as plain Python floats |
| `--out` path coercion | `AttributeError: 'str' object has no attribute 'mkdir'` for `--out data/v2` | A lambda whose two branches returned different types | Named `out_folder()` returning `Path` on both branches |

The first two were found by running the code. The third was found by **writing
these docs** — I wrote a sentence claiming both branches worked, went to verify
it, and found they did not.

None of the three would have been caught by reading the code. The scaler bug is
the most instructive: it did not appear in the training path at all, because
there the scaler was already on CPU. It only surfaced when a *different project*
loaded the saved weights — which is the argument for the cross-project
integration being a real test rather than a nice-to-have.
