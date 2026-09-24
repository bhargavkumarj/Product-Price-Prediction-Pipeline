# Code Walkthrough — The Data Pipeline

Covers `pricing/config.py`, `items.py`, `parsing.py`, `loaders.py`,
`curation.py`, `summarise.py` and `dataset.py`.

Most of the work in this project is here. The models are standard; the data
shaping is what makes them work.

---

## `pricing/config.py`

```python
@dataclass
class Settings:
    data: Path = ROOT / "data"
    artifacts: Path = ROOT / "artifacts"
    reports: Path = ROOT / "reports"

    dataset: str = os.getenv("PRICER_DATASET", "ed-donner/items_lite")
    source_dataset: str = "McAuley-Lab/Amazon-Reviews-2023"

    summariser_model: str = os.getenv("SUMMARISER_MODEL", "groq/openai/gpt-oss-20b")
    frontier_model: str = os.getenv("FRONTIER_MODEL", "openai/gpt-4.1-mini")

    seed: int = 42
```

**`dataset` is configurable, `source_dataset` is not.** The curated dataset is
something you choose — a Hub name, or a local folder. The raw source is a fixed
fact about where this pipeline gets data; making it configurable would imply the
parser works on arbitrary datasets, and it does not — `parse()` expects Amazon's
exact field names.

**Two model settings, deliberately separate.** The summariser runs once over
hundreds of thousands of items, so it should be cheap; the default is a small
open model on Groq. The frontier pricer runs over a few hundred test items and is
meant to represent a strong model. Different jobs, different economics.

**`seed: int = 42`** is a single shared seed used by deduplication shuffling,
balanced sampling, and the torch/numpy seeds in the neural model. One seed for
the whole project means "reproducible" is a property of the project, not
something you have to check module by module.

```python
def __post_init__(self):
    for folder in (self.data, self.artifacts, self.reports):
        folder.mkdir(parents=True, exist_ok=True)
```

Creates the three output folders once, at import. Every other module can write to
them without a `mkdir` guard. `exist_ok=True` makes it idempotent.

`CATEGORIES` is module-level rather than a `Settings` field because a mutable
list default in a dataclass needs `field(default_factory=...)`, and this is a
constant, not configuration.

---

## `pricing/items.py`

### The model

```python
class Item(BaseModel):
    title: str
    category: str
    price: float
    full: str | None = None
    weight: float | None = None
    summary: str | None = None
    prompt: str | None = None
    id: int | None = None
```

Three required fields, five optional ones filled in at different pipeline stages:

| Field | Set by | Used by |
|---|---|---|
| `title`, `category`, `price` | `parse()` | Everything |
| `full` | `parse()` | `Summariser`, fallback for `text` |
| `weight` | `parse()` | `FeaturePricer` |
| `summary` | `Summariser` | Every model, via `text` |
| `prompt` | `make_prompt()` | Fine-tuning formats |
| `id` | `split()` | Batch jobs that need to map results back |

`price: float` and the JSONL round trip interact usefully: the Hub stores price
as a string in some revisions, and pydantic coerces `"64.3"` to `64.3` on
`model_validate`. Without pydantic that becomes a silent string-vs-float bug
deep inside numpy.

### `text` — the single most important property

```python
@property
def text(self) -> str:
    return self.summary or self.full or self.title
```

Three lines that every model depends on. It defines the fallback chain: use the
LLM summary if there is one, otherwise the cleaned raw listing, otherwise at
least the title.

The value of centralising it: **every model sees the same text**. If each model
chose its own, a comparison between them would be partly a comparison of input
choices, not of algorithms. This property is why the results table means
something.

It also makes the pipeline degrade gracefully. On the `items_lite` dataset,
`full` is `None` for every row but `summary` is populated — the models work
unchanged.

### `make_prompt` and `test_prompt`

```python
def make_prompt(self) -> str:
    self.prompt = f"{QUESTION}\n\n{self.text}\n\n{PREFIX}{round(self.price)}.00"

def test_prompt(self) -> str:
    return f"{QUESTION}\n\n{self.text}\n\n{PREFIX}"
```

The training/inference pair for a causal-LM fine-tune. Training text ends with
the answer; test text ends at `"Price is $"` so the model must generate the
number.

`round(self.price)` and the literal `.00` mean the model always emits an integer
number of dollars, which makes both generation and parsing easier — the model
learns a fixed answer shape instead of having to produce arbitrary decimals.

These are not used by the models in this project. They exist because the same
`Item` is the interface to a fine-tuned pricer, which is what the deal discovery
project's Specialist agent calls.

### Hub round-tripping

```python
@classmethod
def from_hub(cls, name: str):
    data = load_dataset(name)
    return tuple(
        [cls.model_validate(row) for row in data[split]]
        for split in ("train", "validation", "test")
    )
```

`model_validate` per row, so a schema mismatch fails at load with a field-level
error rather than surfacing as a strange result later.

---

## `pricing/parsing.py`

Pure functions, no state, no I/O. That is a requirement, not a style preference:
these run inside `ProcessPoolExecutor` workers, so everything they touch must be
picklable and side-effect free.

### The rejection rules

```python
MIN_CHARS = 600
MIN_PRICE = 0.5
MAX_PRICE = 999.49
```

Each is a modelling decision:

**`MIN_PRICE = 0.5`** — below this the price is usually a data error or a
shipping-only listing.

**`MAX_PRICE = 999.49`** — the upper bound, and the odd value is deliberate:
`round(999.49)` is 999, so every price in the dataset renders as at most three
digits. That keeps the target range bounded and makes the fine-tuning prompt
format consistent. Items above $999 are genuinely different — they are priced by
brand and market position more than by description — and including them would add
a long tail that dominates squared-error training without being learnable from
text.

**`MIN_CHARS = 600`** — an item with 50 characters of description has nothing for
a text model to work with. Including such rows adds noise the model cannot
explain, which shows up as irreducible error.

These three rules discard a large fraction of raw Amazon data. That is correct:
curation is subtractive.

### `flatten()`

```python
text = str(value).replace("\n", " ").replace("\r", "").replace("\t", "")
while "  " in text:
    text = text.replace("  ", " ")
return text.strip()[:MAX_CHARS_PER_FIELD]
```

`str(value)` because Amazon's `description` and `features` fields are *lists*, and
`str(["a", "b"])` gives `"['a', 'b']"` — ugly but usable, and it preserves the
content.

The `while` loop collapses runs of spaces. A single `replace("  ", " ")` pass
turns four spaces into two, not one; looping until none remain handles arbitrary
runs. Whitespace runs matter because they tokenise and waste both vocabulary
slots and LLM context.

### `clean()`

```python
for key in DROP_DETAILS:
    details.pop(key, None)
```

Removes `Part Number`, `Best Sellers Rank`, `Batteries Included?`,
`Batteries Required?`, `Item model number`. None of these carry price signal, and
each adds tokens. `pop(key, None)` is the no-KeyError form.

```python
PART_NUMBER = re.compile(r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b")
```

Worth reading carefully. Three lookaheads at a word boundary:

- `(?=[A-Z0-9]{7,}\b)` — at least 7 uppercase-alphanumeric characters to the word
  end
- `(?=.*[A-Z])` — contains at least one letter
- `(?=.*\d)` — contains at least one digit

Then `[A-Z0-9]+\b` consumes it. So `WH1000XM4` and `B08N5WRWNW` match, while
`BLUETOOTH` (no digit), `2023` (no letter) and `USB` (too short) do not.

Why remove them: part numbers are near-unique strings. In a bag-of-words model
each one becomes its own vocabulary entry appearing once, which is pure noise
that crowds out real words. For an LLM they consume tokens and invite spurious
pattern matching.

```python
return PART_NUMBER.sub("", "\n".join(parts)).strip()[:MAX_CHARS_TOTAL]
```

Assemble, strip part numbers, truncate to 4,000 characters — a bound on both
vectoriser input and LLM cost.

### `weight_in_pounds()`

```python
WEIGHT_IN_POUNDS = {
    "pounds": 1.0, "ounces": 1/16, "grams": 1/453.592,
    "milligrams": 1/453592, "kilograms": 1/0.453592,
}
```

A conversion table rather than an if-chain. Adding a unit is one entry.

```python
try:
    amount, unit = float(parts[0]), parts[1].lower()
except (IndexError, ValueError):
    return 0.0
```

Catches both malformed shapes: a one-token string (`IndexError`) and a
non-numeric first token (`ValueError`). Real catalogue data contains both.

```python
if unit == "hundredths" and len(parts) > 2 and parts[2].lower() == "pounds":
    return amount / 100
return amount * WEIGHT_IN_POUNDS.get(unit, 0.0)
```

`"50 hundredths pounds"` is a real Amazon format and needs a three-token special
case. The `.get(unit, 0.0)` fallback means an unknown unit yields 0.0 rather than
raising — and `FeaturePricer` has a `weight_missing` flag precisely so the model
can distinguish "weighs nothing" from "we do not know".

### `parse()`

```python
try:
    price = float(row["price"])
except (TypeError, ValueError):
    return None
```

`TypeError` for `None`, `ValueError` for `""` or `"see price in cart"`. Both are
common.

**Returning `None` rather than raising** is the design. A rejected row is a normal
outcome, not an error — most rows are rejected. The caller filters:

```python
parsed = (parse(row, self.category) for row in chunk)
return [item for item in parsed if item is not None]
```

Exceptions would mean try/except around every row and would conflate "this row is
not useful" with "something is broken".

---

## `pricing/loaders.py`

### Why processes, not threads

```python
WORKERS = max((os.cpu_count() or 2) - 1, 1)
```

Parsing is CPU bound — JSON decoding, regex substitution, string manipulation on
millions of rows. The GIL serialises threads for that work, so threads would give
no speedup. Processes get real parallelism.

`cpu_count() - 1` leaves a core for the main process and the OS, so the machine
stays usable. `or 2` handles `cpu_count()` returning `None`, and `max(..., 1)`
guarantees at least one worker on a single-core machine.

### Chunking

```python
CHUNK_SIZE = 1000

def _chunks(self):
    size = len(self.dataset)
    for start in range(0, size, CHUNK_SIZE):
        yield self.dataset.select(range(start, min(start + CHUNK_SIZE, size)))
```

Row-by-row dispatch to a process pool would spend all its time on IPC — the
serialisation cost per task would dwarf the parse. 1,000 rows per task amortises
that.

A generator, not a list, so the whole dataset is never materialised as chunk
objects at once. `datasets.select` returns a lazy view.

### The pool

```python
with ProcessPoolExecutor(max_workers=workers) as pool:
    for batch in tqdm(pool.map(self._parse_chunk, self._chunks()), total=expected):
        items.extend(batch)
```

`pool.map` preserves input order, which keeps the output deterministic — worth
having, since the seed-based sampling downstream assumes a stable input order.

`_parse_chunk` is a **method**, so `self` is pickled to each worker. That works
because `CategoryLoader` holds only a category string and a dataset handle. If it
held an open file or a model, this would fail — a real constraint that the pure
`parsing.py` design respects.

The `tqdm(total=expected)` uses `(len // CHUNK_SIZE) + 1` so the bar is
meaningful from the start.

### Timing

```python
minutes = (datetime.now() - started).total_seconds() / 60
print(f"{self.category}: {len(items):,} usable items in {minutes:.1f} min", flush=True)
```

`flush=True` because this runs for minutes with a process pool attached, and
buffered output would appear in bursts or not at all. Printing the yield per
category also surfaces a broken parser immediately: a category returning 0 usable
items is obvious.

---

## `pricing/curation.py`

The most consequential file in the project.

### `deduplicate()`

```python
rng = random.Random(settings.seed)
items = items[:]
rng.shuffle(items)

for attribute in ("title", "full"):
    seen: set[str] = set()
    kept = []
    for item in items:
        value = getattr(item, attribute)
        if value not in seen:
            seen.add(value)
            kept.append(item)
    items = kept
```

**Shuffle first, then dedup.** Deduplication keeps the *first* occurrence. Without
shuffling, "first" means "first in catalogue order", which correlates with
category and listing age — so the survivor of each duplicate group would be
systematically biased. Shuffling makes the survivor random. Seeded, so it is
reproducible.

**Two passes, title then body.** Different duplicates. Identical titles are the
same product listed twice. Identical bodies are the same product under different
titles — variants, or re-listings. Both must go, because a duplicate that lands
in both train and test leaks the answer.

**A `set` for membership.** O(1) lookup; a list would make this O(n²) on millions
of items.

`items[:]` copies before shuffling, because `rng.shuffle` is in-place and
mutating the caller's list is a surprising side effect.

### `balance()` — the heart of the curation

```python
scaled = (prices - prices.min()) / (prices.max() - prices.min() + 1e-9)
weights = scaled**PRICE_POWER            # PRICE_POWER = 2
for category, factor in CATEGORY_WEIGHTS.items():
    weights[categories == category] *= factor
weights /= weights.sum()
rng = np.random.default_rng(settings.seed)
chosen = rng.choice(len(items), size=size, replace=False, p=weights)
```

**The problem.** Raw marketplace data is dominated by cheap items in two
categories. Train on that and the model learns the base rate: always guess about
$20, which minimises error on the training distribution and is useless.

**The fix — price-weighted sampling.** Scale prices to 0–1, square them, use that
as a sampling probability. A $900 item is `(0.9)² = 0.81`; a $50 item is
`(0.05)² = 0.0025`. The expensive item is ~320× more likely to be sampled. The
result is a flatter price distribution the model can learn a *range* from.

Why squared rather than linear: linear weighting is not aggressive enough given
how extreme the raw skew is. The exponent is a knob (`PRICE_POWER`), and the
right value depends on the source distribution.

**Category damping.**

```python
CATEGORY_WEIGHTS = {"Tools_and_Home_Improvement": 0.5, "Automotive": 0.05}
```

Automotive is cut to 5%. It is by far the largest category in Amazon's data and
its listings are mostly interchangeable parts whose price depends on the vehicle,
not the description. Without damping it would dominate the sample and the model
would specialise in something it cannot learn.

**`replace=False`** — no item sampled twice, which would be a self-inflicted
duplicate after all the work `deduplicate` just did.

**`+ 1e-9`** guards against every item having the same price, which would make the
denominator zero.

**`np.random.default_rng(seed)`** is the modern NumPy generator API, seeded. Like
`random.Random`, it is a local instance that neither reads nor writes global
random state.

```python
random.Random(settings.seed).shuffle(sample)
```

A final shuffle, because `rng.choice` returns indices in ascending order. Without
this, `split()` — which slices — would put the cheapest items in train and the
most expensive in test. That would be a catastrophic distribution shift, and it
would look like the model simply cannot predict expensive items.

### `split()`

```python
validation = min(validation, len(items) // 10)
test = min(test, len(items) // 10)
cut = len(items) - validation - test
train, val, held_out = items[:cut], items[cut:cut + validation], items[cut + validation:]
```

Slicing is a fair split **only because the list was just shuffled** — which is
why the shuffle is in `balance` and the docstring here says so explicitly.

The `min(..., len // 10)` clamps are what let the same function work for a 5,000-
item smoke test and an 800,000-item real run. Defaults of 10,000 validation and
2,000 test would consume the entire small dataset.

```python
for index, item in enumerate(train + val + held_out):
    item.id = index
```

Sequential ids across all splits, assigned after splitting so they are stable.
Used by batch APIs that return results keyed by `custom_id`.

### `describe()`

Returns count, mean, median, min, max and a category histogram sorted by
frequency. Printed after curation. Mean versus median immediately shows whether
the balancing worked: raw Amazon data has a median far below its mean, and a
successful balance narrows that gap.

---

## `pricing/summarise.py`

### Why summarise at all

Raw listings are up to 4,000 characters of marketing copy, spec tables and
shipping notes. Two products at the same price can have wildly different listing
styles, which is noise for every model.

The LLM rewrites each into a fixed five-line format. For bag-of-words models this
collapses the vocabulary onto words that matter. For the neural net it means the
hashed features are computed over consistent text. For the zero-shot LLM it means
a shorter, cleaner prompt.

### The format

```
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features
```

Fixed fields, one sentence each. `Brand` is explicitly extracted because brand is
one of the strongest price signals in consumer goods — the same specification
from two brands can differ 3× in price, and burying the brand in prose makes it
one token among hundreds.

"Do not include part numbers" repeats at the LLM level what the regex does at the
parsing level. Belt and braces, because the regex only catches uppercase
alphanumeric runs.

### Cost tracking

```python
usage = response.usage
self.input_tokens += usage.prompt_tokens
self.output_tokens += usage.completion_tokens
self.cost += response._hidden_params.get("response_cost") or 0.0
```

Accumulated per instance and printed at the end. Summarising 400,000 items is a
real spend; a pipeline that does it without reporting the cost is a pipeline that
surprises somebody.

`response._hidden_params` is a LiteLLM internal, hence the `.get(...) or 0.0` —
providers that do not report cost yield `None`, and `or 0.0` handles both a
missing key and a `None` value.

### `apply()`

```python
def work(item: Item) -> Item:
    if not item.summary:
        item.summary = self.summarise(item.full or item.title)
    return item

with futures.ThreadPoolExecutor(max_workers=workers) as pool:
    list(tqdm(pool.map(work, items), total=len(items), desc="summarising"))
```

**`if not item.summary`** makes the whole operation resumable. Interrupt a
400,000-item run, restart it, and it picks up where it left off. That is not
optional at this scale — something will always fail partway.

**Threads, not processes.** Every call is waiting on HTTP.

**Mutation in place.** `work` mutates the `Item` and the results of `pool.map`
are discarded — `list(...)` exists only to drive the iterator and feed `tqdm`.
That is why `cmd_summarise` can call `apply` on each split and then save all
three: the objects were updated in place.

---

## `pricing/dataset.py`

### JSONL rather than CSV or pickle

```python
for item in items:
    handle.write(json.dumps(item.model_dump()) + "\n")
```

One JSON object per line. Streamable (read one line at a time, never load the
whole file), diffable, appendable, and unlike pickle it is safe to load and
readable in five years. Unlike CSV it handles the embedded newlines and commas
that product descriptions are full of.

### `load()` — the unified entry

```python
folder = Path(source)
if folder.exists() and folder.is_dir():
    train, validation, test = load_local(folder)
else:
    train, validation, test = Item.from_hub(source)
```

One function that takes either a Hub name or a local path, disambiguated by
whether the string happens to be an existing directory.

This is a small thing with a large effect: `run.py compare --dataset
ed-donner/items_lite` works for someone who has never run curation, and
`--dataset data/curated` works for someone who has. The same code path serves
both, so there is no "demo mode" that behaves differently from the real thing.
