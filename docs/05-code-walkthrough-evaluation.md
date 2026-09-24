# Code Walkthrough — Evaluation and the CLI

Covers `pricing/evaluation.py` and `run.py`.

---

## `pricing/evaluation.py`

The file that makes the comparison honest. It imports only `config` and `items` —
it knows nothing about any model, which is exactly why every model can go
through it.

### `to_price()` — the adapter

```python
NUMBER = re.compile(r"[-+]?\d*\.\d+|\d+")

def to_price(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    match = NUMBER.search(str(value).replace("$", "").replace(",", ""))
    return float(match.group()) if match else 0.0
```

The shim that lets a scikit-learn model returning `68.7231` and an LLM returning
`"$1,249.99"` go through identical downstream code.

**The regex order matters.** `[-+]?\d*\.\d+|\d+` tries the decimal branch
*first*. Regex alternation is ordered, so with `\d+` first, `"249.99"` would
match just `"249"` and silently truncate the cents. The decimal branch must come
first.

`\d*\.\d+` allows `.99` with no leading digit. `[-+]?` accepts a sign, although a
negative price is floored later.

**Stripping `$` and `,` before matching** handles `"$1,249.99"`. Without removing
the comma, the regex would stop at it and return `1.0`.

**`return ... else 0.0`** — a model that replies "I cannot estimate this" scores
as 0, producing an error equal to the true price. That is a loud, visible failure
in the worst-misses diagnostic rather than a crash that loses the whole run.

### `band()` — what counts as a usable prediction

```python
def band(error: float, truth: float) -> str:
    if error < 40 or error / truth < 0.2:
        return "green"
    if error < 80 or error / truth < 0.4:
        return "amber"
    return "red"
```

**Two criteria joined by `or`, which is the design.** An absolute dollar
tolerance *and* a relative percentage tolerance, and passing either is enough.

Why both are needed:

- A **$5 error on a $20 item** is 25% — fails the relative test, passes the
  absolute one. And rightly: nobody cares about $5.
- A **$60 error on a $900 item** is 6.7% — fails the absolute test, passes the
  relative one. And rightly: that is a good estimate of an expensive item.

Either criterion alone would mislabel one end of the price range. This mirrors
how a person judges a price estimate: close in dollars *or* close in proportion.

### `Prediction`

```python
@dataclass
class Prediction:
    title: str
    truth: float
    guess: float

    @property
    def error(self) -> float:
        return abs(self.guess - self.truth)

    @property
    def squared_log_error(self) -> float:
        return (math.log(max(self.guess, 0) + 1) - math.log(self.truth + 1)) ** 2

    @property
    def band(self) -> str:
        return band(self.error, self.truth)
```

Stores three facts; derives everything else. Properties rather than stored fields
means the derived values cannot go stale, and a new metric is a new property
rather than a migration of stored data.

`max(self.guess, 0)` in `squared_log_error` guards `log` of a negative number.
The `+ 1` inside each log is the `log1p` convention, defined at zero.

### `Result` — the aggregate

```python
@property
def average_error(self) -> float:
    return statistics.fmean(p.error for p in self.predictions)

@property
def median_error(self) -> float:
    return statistics.median(p.error for p in self.predictions)
```

**Both, deliberately.** The measured gap between them is the finding of the whole
project:

```
random forest    avg $68.72   median $34.22
svd + ridge      avg $69.70   median $34.86
bow linear       avg $82.97   median $33.55
```

Average error is roughly double median error for every model. That is the
signature of a **skewed error distribution**: half of all predictions are within
about $34, and the average is dragged up by a small number of large misses.

Reporting only the average would say "the model is $69 off", which is misleading
about typical behaviour. Reporting only the median would say "$34 off" and hide
that the tail exists. Both together describe the actual shape.

```python
@property
def rmsle(self) -> float:
    return math.sqrt(statistics.fmean(p.squared_log_error for p in self.predictions))
```

Root Mean Squared Log Error — the scale-free metric. Because it works on log
ratios, being 2× off costs the same at $10 and $500. It is the fairest single
number for comparing models on a heavy-tailed target, and it is the metric that
ranks the models most consistently with intuition.

```python
@property
def r2(self) -> float:
    return r2_score([p.truth for p in self.predictions], [p.guess for p in self.predictions])
```

Fraction of variance explained, relative to predicting the mean. 1.0 is perfect,
0.0 is as good as the mean, negative is worse than the mean.

**Negative R² is not a bug.** The bag-of-words model scored -0.899 because its
handful of extreme over-predictions contribute enormous squared error. R² is the
metric most sensitive to the tail, which is exactly what makes it useful
alongside the median.

```python
@property
def hit_rate(self) -> float:
    """Share of predictions a buyer would accept: within $40 or 20%."""
    return sum(p.band == "green" for p in self.predictions) / len(self.predictions)
```

The business metric. Statistical metrics answer "how wrong on average?"; hit rate
answers "what fraction of the time is this usable?" — 60% for the random forest.
For a decision system, that framing is often the one that matters, and it is the
number to lead with when explaining results to a non-specialist.

```python
def worst(self, n: int = 5) -> list[Prediction]:
    return sorted(self.predictions, key=lambda p: p.error, reverse=True)[:n]
```

Error analysis, not just scoring. The worst misses are where you learn what the
model does not understand. Across every model in the measured run, the same items
appear:

```
Canon EOS Rebel T7 Digital SLR Camera   truth $819.00   guess $183.26
Dell XPS 13 9343-2727SLV                truth $839.99   guess $392.70
Bam France 2002XL Contoured Hightech    truth $713.00   guess $143.81
```

All expensive branded electronics. Every model systematically under-predicts
them, because at the top of the price range the description reads much the same
whether the item costs $200 or $800 — the price is driven by brand and market
position, which the text barely encodes.

That is a **data** finding, not a model finding, and no aggregate metric would
have produced it.

### `evaluate()`

```python
def evaluate(predictor, items, name=None, size=250, workers=8) -> Result:
    name = name or getattr(predictor, "display_name", predictor.__class__.__name__)
    sample = items[:size]
    started = datetime.now()

    def run(item: Item) -> Prediction:
        guess = to_price(predictor(item))
        return Prediction(title=item.title[:40], truth=item.price, guess=max(guess, 0.0))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        predictions = list(tqdm(pool.map(run, sample), total=len(sample), desc=name))
```

**`predictor` is just a callable.** No type annotation, no isinstance check. A
`Pricer`, a bare function, a lambda — anything callable works. That is the
`__call__` forwarding in `base.py` paying off.

**`getattr(predictor, "display_name", predictor.__class__.__name__)`** —
use the friendly name if the object has one, otherwise fall back to the class
name. So a plain function or a third-party estimator still gets a sensible label.

**`items[:size]`** — the first N, deterministically, **not a random sample**. That
is the critical detail for comparability: every model is scored on *exactly the
same items*. A random sample per model would introduce variance between rows that
has nothing to do with the models. The list was already shuffled during curation,
so the first N is an unbiased slice.

**`pool.map` preserves order**, so `predictions[i]` corresponds to `sample[i]` —
useful when reading the raw JSON report.

**`max(guess, 0.0)`** floors at zero. A negative price is not a prediction.

**`title[:40]`** truncates for display; the report is meant to be read in a
terminal.

**Timing covers scoring only, not fitting.** The `secs` column answers "how
expensive is this model to *use*", which is the number that matters at inference
time. Worth knowing when reading the table: the residual net shows a low `secs`
despite taking minutes to train.

### `table()`

```python
for r in sorted(results, key=lambda r: r.average_error)
```

Sorted best-first by average error, so the table has an obvious reading order.

```python
widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]

def line(cells):
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()
```

`zip(headers, *rows)` transposes so each `column` contains the header plus every
value beneath it; `max(len(...))` gives the required width. `.rstrip()` removes
trailing padding.

Plain text, no pandas. It pastes cleanly into a terminal, a README or a commit
message.

```python
f"{r.hit_rate:.1%}"
```

The `%` format multiplies by 100 and appends the sign, so `0.60` renders as
`60.0%`.

### `diagnostics()`

```python
GREEN, AMBER, RED, RESET = "\033[92m", "\033[93m", "\033[91m", "\033[0m"
```

ANSI colour codes. The green/amber/red counts are scannable at a glance in a
terminal, which is where this output is read.

### `save()`

```python
"results": [
    {
        **result.summary(),
        "predictions": [
            {"title": p.title, "truth": p.truth, "guess": p.guess}
            for p in result.predictions
        ],
    }
    for result in results
],
```

**Every individual prediction is stored**, not just the summary. That is what
makes a report re-analysable: you can compute a metric that did not exist when
the run happened, plot the error distribution, or compare two models item by item
to see whether they fail on the same products.

`settings.dataset` is recorded too, so a report states what it was run against.

Timestamped filenames in UTC, never overwritten.

### `scatter()`

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

Imported **inside the function**, so plotly is only needed when `--chart` is
passed.

```python
top = max(max(p.truth, p.guess) for p in result.predictions)
...
go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(dash="dash", ...))
```

Every panel gets a dashed **y = x** reference line. On a predicted-vs-actual plot
that line is perfection, so the visual story is immediate: points below the line
are under-predictions, points above are over-predictions.

The measured charts show a clear pattern — a dense cluster hugging the line at
the cheap end, and a fan of points well *below* the line at the expensive end.
That is the systematic under-prediction of expensive items, visible at a glance
in a way no table conveys.

Points are coloured by band, so the green/amber/red split is spatially readable
too.

---

## `run.py`

### Subcommand wiring

```python
sub = parser.add_subparsers(dest="command", required=True)
curate = sub.add_parser("curate", help="...")
curate.set_defaults(func=cmd_curate)
...
args = parser.parse_args()
args.func(args)
```

`set_defaults(func=...)` attaches the handler to the parsed namespace, so
dispatch is two lines with no if-chain. Adding a subcommand does not touch
`main`'s dispatch logic.

`required=True` means running `python run.py` with no subcommand prints usage
instead of failing with an unhelpful `AttributeError` on `args.func`.

### Deferred imports in handlers

```python
def cmd_curate(args) -> None:
    from pricing.curation import curate, describe
```

`curation` pulls in `loaders`, which pulls in `datasets` — heavy. Importing it
inside the handler keeps `run.py --help` and `run.py compare --models average`
fast.

The same pattern for `Summariser` (pulls litellm), `scatter` (pulls plotly) and
`Item` inside `cmd_predict`.

### `build()`

```python
def build(name: str, args):
    if name == "frontier":
        return create("frontier", model=args.frontier_model)
    return create(name)
```

The single place a model name becomes an object. Only `frontier` needs a
construction argument from the CLI, because only it has a provider to choose.

If a second model needed CLI arguments this would become a dict of builders —
but it does not, and three lines beats a framework.

### `cmd_compare()`

```python
prices = [item.price for item in test[: args.size]]
print(f"Test sample: {len(prices)} items, mean ${statistics.fmean(prices):.2f}, "
      f"median ${statistics.median(prices):.2f}\n")
```

Prints the **test set's own statistics first**, before any model runs. This is
the reference frame for everything that follows: mean $153.68, median $107.00.
A model with $106 average error is doing roughly as well as guessing the mean,
and you can only see that if the mean is on screen.

```python
for name in args.models:
    model = build(name, args)
    if model.needs_training:
        model.fit(train, validation)
    result = evaluate(model, test, name=model.display_name, size=args.size, workers=args.workers)
    results.append(result)
    print(f"\n{diagnostics(result)}\n")

print(table(results))
```

Per-model diagnostics as each finishes, then the comparison table at the end.
Diagnostics-as-you-go means a long run gives feedback continuously rather than
producing everything at the end.

Every model in the loop is fit on the same `train`/`validation` and scored on the
same `test[:size]`. The comparison is controlled by construction, which is the
entire reason for structuring it this way rather than running models separately
and collecting numbers.

```python
choices=sorted(REGISTRY)
```

The CLI's valid model names come from the registry, so they cannot drift out of
sync. `sorted` makes `--help` readable.

### `cmd_train()`

```python
model = create("neural", epochs=args.epochs, rows=args.rows)
model.fit(train, validation)
print(f"Saved weights to {model.save()}")

result = evaluate(model, test, name=model.display_name, size=args.size)
```

Exists because the neural net is the one model expensive enough to be worth
training once and reusing. `compare --models neural` retrains from scratch every
time; `train` persists the weights.

It evaluates immediately after saving, so you never have a saved artefact of
unknown quality.

The saved file is the artefact the deal discovery project's `NeuralAgent` loads.

### `cmd_predict()`

```python
item = Item(title=args.text[:60], category="unknown", price=0.0, full=args.text)
print(f"${model.predict(item):,.2f}")
```

Wraps free text in an `Item` so the normal `predict` path works unchanged.

`price=0.0` is a required field being satisfied with a placeholder — it is never
read during prediction. `category="unknown"` is what makes
`CategoryAveragePricer`'s `self.overall` fallback necessary.

`full=args.text` rather than `summary`, so `Item.text` falls through to it.

The cost of this command is honest and visible: it loads the dataset and fits the
model before predicting, because nothing except the neural net persists. For a
real service the fitted model would be cached — a fair thing to point out.

### `out_folder()` — and a bug found while writing these docs

```python
def out_folder(value: str) -> Path:
    """A bare name lands under data/; anything with a separator is used as given."""
    return Path(value) if "/" in value else settings.data / value
```

A bare name like `mydata` resolves under `data/`; anything containing a separator
is used as given, so `--out /tmp/scratch` works.

This started life as an inline lambda:

```python
type=lambda p: settings.data / p if "/" not in p else p     # the old version
```

Writing this documentation surfaced the bug. The `if` branch returns a `Path`
(because `settings.data / p` is path division) but the `else` branch returns the
raw `str`. `save_local` then calls `folder.mkdir(...)` on it:

```
AttributeError: 'str' object has no attribute 'mkdir'
```

So `run.py curate --out data/v2` — a completely reasonable invocation — would
crash, while `--out v2` worked. A named function with a return annotation makes
the inconsistency obvious; the lambda hid it. Fixed by wrapping both branches in
`Path`.
