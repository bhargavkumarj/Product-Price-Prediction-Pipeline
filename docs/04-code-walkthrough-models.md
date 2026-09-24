# Code Walkthrough — The Models

Covers `pricing/models/base.py`, `__init__.py`, `baselines.py`, `classical.py`,
`neural.py` and `frontier.py`.

---

## `pricing/models/base.py` — the contract

```python
class Pricer:
    display_name = "pricer"
    needs_training = True

    def fit(self, train, validation=None) -> "Pricer":
        return self

    def predict(self, item: Item) -> float:
        raise NotImplementedError

    def __call__(self, item: Item) -> float:
        return self.predict(item)
```

Seventeen lines that make the entire comparison possible.

**`fit` returns `self`** so `create("neural").fit(train, val)` chains. It also has
a working default — a model that needs no training inherits a no-op rather than
having to define an empty method.

**`predict` raises `NotImplementedError`** rather than returning 0. A subclass
that forgets to implement it fails loudly at the first prediction instead of
silently scoring as the worst model in the table.

**`__call__` forwarding to `predict`** is the important one. It means
`evaluate()` never needs to know about `Pricer` — it just calls
`predictor(item)`. A bare function works. A lambda works. This is what makes the
harness genuinely model-agnostic.

**`needs_training`** lets `cmd_compare` write:

```python
if model.needs_training:
    model.fit(train, validation)
```

instead of special-casing `random` and `frontier` by name. Adding an untrained
model needs no change to the CLI.

**`display_name`** separates the CLI key from the human label. `bow-linear` is
what you type; `linear (bag of words)` is what appears in the table. It is a
class attribute for most models and an instance attribute for `FrontierPricer`,
which builds it from the model name at construction.

---

## `pricing/models/__init__.py` — the lazy registry

```python
REGISTRY = {
    "random": ("pricing.models.baselines", "RandomPricer"),
    ...
    "xgboost": ("pricing.models.classical", "XGBoostPricer"),
    "neural": ("pricing.models.neural", "NeuralPricer"),
    "frontier": ("pricing.models.frontier", "FrontierPricer"),
}

def get(name: str):
    module, attribute = REGISTRY[name]
    return getattr(import_module(module), attribute)
```

The registry holds **strings**, not classes. Nothing is imported until `get` is
called.

### Why this is not over-engineering

The first version imported every class at the top of `__init__.py`. Running
`python run.py compare --models average` produced:

```
xgboost.core.XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded.
Likely causes: OpenMP runtime is not installed
  Mac OSX users: Run `brew install libomp`
```

The *entire CLI* was dead — including `--help` — because one optional model's
native dependency was missing on this machine.

With the lazy registry, `xgboost` is imported only if you ask for it. Every other
model runs. That is what let the published results table be produced on a machine
where XGBoost genuinely cannot load.

The same applies to `torch` (heavy, slow to import) and `litellm` (pulls in many
provider SDKs). `python run.py --help` imports none of them.

### The residual eager import

`XGBoostPricer.fit` also does `from xgboost import XGBRegressor` *inside* the
method. Belt and braces: even if someone imports `classical` directly, only
constructing and fitting an `XGBoostPricer` touches xgboost. The class can be
listed in `--help` on a machine that cannot run it.

### `create()`

```python
def create(name: str, **kwargs):
    return get(name)(**kwargs)
```

One line, and it is what `run.py` calls. `create("frontier", model="openai/...")`
and `create("neural", epochs=8)` both work, so per-model construction arguments
flow through without the CLI knowing any model's signature.

---

## `pricing/models/baselines.py`

### Why baselines matter

A model that cannot beat "always guess the average" is not learning anything from
the text, no matter how sophisticated it looks. These three establish the floor,
and the measured table is only interpretable because they are in it.

### `RandomPricer`

```python
class RandomPricer(Pricer):
    display_name = "random guess"
    needs_training = False

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def predict(self, item: Item) -> float:
        return self.rng.randrange(1, 1000)
```

The absolute floor. Seeded so two runs agree; a random baseline that varies run
to run is not a baseline.

Note this is the one model with predictable thread-safety weirdness — 8 threads
sharing one `random.Random` produce an interleaved but still valid sequence. It
does not matter for a random baseline, and it is worth knowing it is there.

### `AveragePricer`

```python
def fit(self, train, validation=None):
    self.average = statistics.fmean(item.price for item in train)
    return self
```

Predicts the training mean for everything. Measured: **$106.08 average error,
R² of -0.001.**

That R² is worth understanding. R² measures variance explained *relative to
predicting the mean*. A model that always predicts the mean explains exactly zero
variance, so R² ≈ 0 by construction. It is a sanity check on the whole metric
setup: if `AveragePricer` scored R² of 0.3, something would be wrong.

### `CategoryAveragePricer`

```python
prices: dict[str, list[float]] = {}
for item in train:
    prices.setdefault(item.category, []).append(item.price)
self.by_category = {name: statistics.fmean(values) for name, values in prices.items()}
self.overall = statistics.fmean(item.price for item in train)
```

Predicts the mean for the item's category. Measured: **$93.56** — $12.52 better
than the global mean.

That $12.52 is the value of *one categorical feature with eight levels*. It is
the yardstick for everything else: the best text model gains $37 over the global
mean, so roughly two-thirds of the achievable gain comes from reading the text
rather than knowing the category.

`self.overall` is the fallback for an unseen category at prediction time — which
happens in `cmd_predict`, where an item is constructed with
`category="unknown"`. Without the fallback that would be a `KeyError` on a
perfectly reasonable use of the CLI.

---

## `pricing/models/classical.py`

### The log transform — the most important 8 lines in the file

```python
def to_log(prices) -> np.ndarray:
    return np.log1p(np.asarray(prices, dtype=float))

def from_log(values) -> np.ndarray:
    return np.clip(np.expm1(np.asarray(values, dtype=float)), 0.5, None)
```

**Every model here fits `log1p(price)` and inverts with `expm1`.**

Why. Squared error on raw dollars treats a $100 error the same whether the item
costs $120 or $900. It is catastrophic in the first case and reasonable in the
second. Worse, with prices spanning $0.50 to $999, the gradient is dominated by
the expensive items — the model spends its capacity on the tail and gets the bulk
of the distribution wrong.

In log space, equal errors are equal *ratios*. Being 2× off is the same penalty
at $10 and at $500. That matches what a pricing decision actually cares about,
and it matches how the price distribution is shaped — roughly log-normal.

`log1p(x)` is `log(1 + x)`, which is defined at 0 and numerically stable for
small x. `expm1` is its exact inverse. Using `log`/`exp` with a manual `+1`/`-1`
loses precision for small values.

**`np.clip(..., 0.5, None)`** floors predictions at the dataset's minimum price.
`expm1` of a sufficiently negative number approaches -1, and a negative price is
meaningless. Clipping to the known floor is more honest than clipping to 0.

### `FeaturePricer` — how far do the obvious signals get you?

```python
@staticmethod
def features(item: Item) -> dict:
    text = item.text
    return {
        "weight": item.weight or 0.0,
        "weight_missing": int(not item.weight),
        "text_length": len(text),
        "word_count": text.count(" ") + 1,
    }
```

Four numbers, no text understanding at all. The hypothesis: heavier things and
more-described things cost more.

**`weight_missing` is the interesting feature.** Without it, a missing weight is
encoded as 0.0, which the model reads as "weighs nothing" — indistinguishable
from a genuinely weightless item. The flag lets the model learn a separate
intercept for "unknown", which is the standard fix for missing numeric data.

Measured: **$83.75**, R² 0.067. Better than the category average on error, and it
proves the features carry *some* signal. But 0.067 R² means it explains almost no
variance — it gets the middle of the distribution roughly right and nothing else.

`text.count(" ") + 1` is a deliberately cheap word count. `len(text.split())` is
more correct and allocates a list per call; at this scale the approximation is
fine.

### `BagOfWordsLinearPricer` — the most instructive result in the project

```python
self.vectorizer = CountVectorizer(max_features=20_000, stop_words="english", min_df=2)
...
matrix = self.vectorizer.fit_transform(item.text for item in train)
self.model = Ridge(alpha=1.0).fit(matrix, to_log([item.price for item in train]))
```

Count how many times each word appears; fit a linear model on those counts.

**`min_df=2`** drops words appearing in only one document. Those are almost
always typos, part-number fragments or one-off brand names — a vocabulary entry
that can never generalise, and with one observation the coefficient is fit to
noise.

**`stop_words="english"`** removes "the", "and", "with". No price signal, and
they are the highest-frequency terms so they would dominate the count matrix.

**`Ridge` rather than `LinearRegression`.** With 20,000 features and 20,000
training rows, ordinary least squares is at the edge of being underdetermined and
will fit huge coefficients to rare words. Ridge's L2 penalty shrinks them. This
matters enormously here and is discussed under the R² result below.

**The measured result:**

```
linear (bag of words)   avg $82.97   median $33.55   RMSLE 0.805   R² -0.899   hit 55.0%
```

**Best median error of any model, and a strongly negative R².**

Both are true and the combination is the point. Half its predictions are within
$33.55 — better than the random forest's $34.22. But R² of -0.899 means it is
*worse than predicting the mean* in variance-explained terms.

The mechanism: linear models extrapolate without bound. A rare word with a large
positive coefficient, appearing several times in one listing, produces an
arbitrarily large prediction. The measured worst miss:

```
For Volvo C70 S70 V70 V40 S40 AC Compres   truth $280.41   guess $2,595.82
```

A $2,300 error on one item contributes more squared error than dozens of good
predictions contribute correctly. R² is a squared-error metric, so a handful of
blowups sink it while the median — which ignores the tail entirely — stays
excellent.

**This is why the project reports five metrics.** Any single one tells a false
story about this model.

### `WordVectorPricer` (SVD + ridge)

```python
self.vectorizer = TfidfVectorizer(max_features=40_000, stop_words="english", min_df=2)
matrix = self.vectorizer.fit_transform(item.text for item in train)
self.svd = TruncatedSVD(n_components=min(self.components, matrix.shape[1] - 1), random_state=42)
reduced = self.svd.fit_transform(matrix)
self.model = Ridge(alpha=1.0).fit(reduced, to_log([...]))
```

Three stages: TF-IDF, then reduce 40,000 sparse dimensions to 300 dense ones with
truncated SVD, then ridge on the dense vectors.

**TF-IDF over raw counts.** Term frequency × inverse document frequency
down-weights words that appear everywhere and up-weights distinctive ones. For
price prediction, "wireless" appearing in half the listings is far less
informative than "titanium" appearing in 2%.

**Truncated SVD on a TF-IDF matrix is Latent Semantic Analysis.** It finds the
directions of greatest variance in term space. Words that co-occur collapse onto
shared components, so the model can generalise from "laptop" to "notebook"
without ever seeing them as related — something raw bag-of-words cannot do, since
every word is an independent dimension.

**`min(self.components, matrix.shape[1] - 1)`** guards a real failure: SVD cannot
produce more components than the matrix has columns minus one. On a tiny training
set with a small vocabulary, asking for 300 components would raise. The clamp
makes the model work at any dataset size.

**`random_state=42`** — truncated SVD uses randomised algorithms, so the seed
makes the decomposition reproducible.

Measured: **$69.70, R² 0.322** — second best on error, and the negative-R²
problem is gone. The dimensionality reduction is doing the work: with 300 dense
features instead of 40,000 sparse ones, no single rare word can drive an
unbounded prediction.

### `RandomForestPricer`

```python
self.model = RandomForestRegressor(
    n_estimators=100, n_jobs=-1, random_state=42, min_samples_leaf=2
).fit(matrix, to_log([item.price for item in train]))
```

100 decision trees, each on a bootstrap sample with a random feature subset,
averaged.

**Why it wins.** A tree predicts the mean of its leaf. That prediction is
**bounded by the training data** — it can never extrapolate beyond the prices it
has seen. The bag-of-words linear model's $2,595 blowup is structurally
impossible here. Averaging 100 trees further smooths the variance.

**`min_samples_leaf=2`** stops the forest growing leaves containing a single
training item, which is memorisation.

**`n_jobs=-1`** uses every core; trees are independent so this is nearly linear
speedup.

**`rows: int = 20_000`** — the forest is trained on fewer rows than the linear
models. Tree fitting on a wide sparse matrix is expensive and scales poorly, and
in practice the extra rows did not pay for the time.

Measured: **best average error ($68.72), best RMSLE (0.705), best hit rate
(60%)** — but R² of only 0.316, lower than the residual net's 0.444. Bounded
predictions avoid catastrophes and also cap the upside: the forest never predicts
$800 for an $840 laptop, because averaging pulls it toward the middle.

### `XGBoostPricer`

```python
self.model = XGBRegressor(
    n_estimators=600, max_depth=8, learning_rate=0.06,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=4,
    early_stopping_rounds=40 if eval_set else None,
    tree_method="hist", n_jobs=-1, random_state=42,
)
```

Gradient boosting: trees fit **sequentially**, each correcting the previous
ensemble's residuals. Usually the strongest classical model on tabular data.

Every parameter is a regularisation decision:

| Parameter | Effect |
|---|---|
| `n_estimators=600` with `learning_rate=0.06` | Many small steps rather than few large ones; more robust than the reverse |
| `max_depth=8` | Bounds interaction order; deeper trees memorise |
| `subsample=0.8` | Each tree sees 80% of rows — stochastic gradient boosting, decorrelates trees |
| `colsample_bytree=0.6` | Each tree sees 60% of features; essential with a sparse text matrix where a few terms dominate |
| `min_child_weight=4` | Minimum evidence per leaf |
| `early_stopping_rounds=40` | Stop when validation stops improving for 40 rounds |
| `tree_method="hist"` | Bucket features into histograms — much faster on high-dimensional sparse data |

**The conditional early stopping:**

```python
eval_set = None
if validation:
    validation = validation[:5_000]
    eval_set = [(self.vectorizer.transform(...), to_log([...]))]
...
early_stopping_rounds=40 if eval_set else None
```

XGBoost raises if `early_stopping_rounds` is set without an `eval_set`. Tying
them to the same condition means `fit(train)` with no validation still works.

Capping validation at 5,000 rows keeps the per-round evaluation cheap — it runs
after every one of up to 600 rounds.

**Not in the measured table.** This machine lacks the OpenMP runtime XGBoost
links against, so it could not be run. That is stated plainly in the project
README rather than filled in with a plausible number.

---

## `pricing/models/neural.py`

### `best_device()`

```python
if torch.cuda.is_available(): return torch.device("cuda")
if torch.backends.mps.is_available(): return torch.device("mps")
return torch.device("cpu")
```

CUDA, then Apple Metal, then CPU. Measured runs used `mps`.

### `ResidualBlock`

```python
self.block = nn.Sequential(
    nn.Linear(width, width), nn.LayerNorm(width), nn.ReLU(), nn.Dropout(dropout),
    nn.Linear(width, width), nn.LayerNorm(width),
)
self.activation = nn.ReLU()

def forward(self, x):
    return self.activation(self.block(x) + x)
```

**The `+ x` is the entire point.** A skip connection means the block learns a
*residual* — a correction to its input — rather than a full transformation. Two
consequences:

1. **Gradients flow.** The identity path gives backpropagation a route to early
   layers that does not pass through every weight matrix, which is what prevents
   vanishing gradients in deep stacks.
2. **Identity is the default.** If a block's weights are near zero, it passes its
   input through unchanged. So adding blocks cannot make the network worse in
   principle — it can only fail to help.

**`LayerNorm`, not `BatchNorm`.** LayerNorm normalises across features within one
example; BatchNorm normalises across the batch. LayerNorm is independent of batch
size and behaves identically in training and inference, so there is no
train/eval discrepancy and no running-statistics state to get wrong. With a
batch size of 128 and sparse binary inputs, BatchNorm statistics would also be
noisy.

**The final `LayerNorm` has no activation after it** inside `self.block` —
`ReLU` is applied *after* the addition, in `forward`. That is the standard
post-activation residual ordering.

**Dropout at 0.2** inside the block, randomly zeroing 20% of activations during
training. With 13.5M parameters on 20,000 training rows, overfitting is the main
risk.

### `PriceNet`

```python
def __init__(self, inputs: int, width: int = 1024, blocks: int = 4, dropout: float = 0.2):
    self.stem = nn.Sequential(nn.Linear(inputs, width), nn.LayerNorm(width), nn.ReLU(), nn.Dropout(dropout))
    self.blocks = nn.ModuleList(ResidualBlock(width, dropout) for _ in range(blocks))
    self.head = nn.Linear(width, 1)
```

5,000 → 1,024 → four residual blocks at 1,024 → 1. **13,537,281 trainable
parameters**, as the training output reports.

`nn.ModuleList`, not a plain list — it registers the blocks as submodules so
their parameters are included in `model.parameters()` and move with `.to(device)`.
A plain Python list silently produces a network whose middle layers never train.

The head outputs **one unbounded number**: a standardised log price. No
activation, because the target can be any real number after standardisation.

### `NeuralPricer.__init__`

```python
self.vectorizer = HashingVectorizer(n_features=FEATURES, stop_words="english", binary=True)
torch.manual_seed(settings.seed)
np.random.seed(settings.seed)
```

**The hashing trick.** Unlike `CountVectorizer`, `HashingVectorizer` does not
build a vocabulary. It hashes each token to one of 5,000 buckets. Three
consequences:

1. **No fitting.** There is no vocabulary to learn, so `transform` works
   immediately and identically on train and test.
2. **Nothing to persist.** `save()` stores only weights and the scaler — the
   vectoriser is reconstructed from a constant. This is exactly why the deal
   discovery project can load these weights with three lines and no artefacts
   beyond the `.pt` file.
3. **Fixed memory.** 5,000 features regardless of corpus size.

The cost is **hash collisions** — different words sharing a bucket — and
**no inverse mapping**, so the model is not interpretable at the feature level.
For a neural net, neither matters much: collisions act like mild feature noise,
and it was never interpretable anyway.

**`binary=True`** records presence, not count. Whether a listing says "wireless"
three times or once carries little price signal, and binary features are better
conditioned.

### `fit()`

```python
x = self._vectorise(train)
y = torch.log1p(torch.FloatTensor([item.price for item in train])).unsqueeze(1)

self.mean, self.std = float(y.mean()), float(y.std())
y = (y - self.mean) / self.std
```

**Two transforms, composed.** First `log1p` for the same reason as the classical
models. Then standardisation to zero mean and unit variance, because AdamW's
default learning rate assumes a target in roughly that range — log prices centre
around 4.7 with a standard deviation near 1, and training on that raw is slower
to converge.

**`float(...)` on the scaler is a bug fix.** Originally these were kept as
tensors. `save()` wrote them into the checkpoint; `load()` used
`map_location=self.device`, which moved them to MPS; `predict()` computed on a
CPU tensor and multiplied by an MPS scalar:

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, mps:0 and cpu!
```

It only appeared when the deal discovery project *loaded* the saved weights — in
the training path the scaler was already on CPU, so it worked. Keeping them as
plain Python floats removes the device from the problem entirely.

**`.unsqueeze(1)`** makes `y` shape `(N, 1)` to match the model's output. Without
it, broadcasting between `(N,)` and `(N, 1)` silently produces an `(N, N)` loss —
a classic silent PyTorch bug that trains something meaningless.

### The training loop

```python
loss_function = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
```

**`SmoothL1Loss` (Huber).** Quadratic for small errors, linear for large ones. It
is the deliberate middle ground: MSE would let outliers dominate the gradient
(exactly the failure that gives the bag-of-words model R² of -0.899), while pure
L1 gives a constant gradient that converges poorly near the optimum.

**`AdamW` over `Adam`.** AdamW applies weight decay directly to the weights
rather than through the gradient, which is the mathematically correct form of L2
regularisation for adaptive optimisers. With 13.5M parameters on 20,000 rows,
regularisation is doing real work.

**`CosineAnnealingLR`.** The learning rate follows a cosine curve from 1e-3 to
near zero over the run. Large steps early to explore, small steps late to settle.
`scheduler.step()` is called **once per epoch**, matching `T_max=self.epochs`.

```python
loss.backward()
nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
optimizer.step()
```

**Gradient clipping** rescales the gradient if its norm exceeds 1.0. One
pathological batch — an unusual item with an extreme price — can otherwise produce
a huge gradient that destroys weights built over many epochs. The order matters:
after `backward()`, before `step()`.

`optimizer.zero_grad()` before the forward pass, because PyTorch accumulates
gradients by default.

### `_validation_error()`

```python
sample = validation[:size]          # size = 2_000
truths = np.array([item.price for item in sample])
guesses = np.array([self.predict(item) for item in sample])
return float(np.abs(guesses - truths).mean())
```

Reports validation error **in dollars**, not in loss units. The training loss is a
Huber loss on standardised log prices — a number like 0.31, which is
uninterpretable. Dollars are the thing anyone actually cares about, and printing
them per epoch shows immediately whether the model is still improving.

Note this calls `predict` in a Python loop, which is slow — hence the 2,000-row
cap. Batching would be faster; it runs once per epoch and the simplicity is worth
more here.

### `predict()`

```python
self.model.eval()
with torch.no_grad():
    x = self._vectorise([item]).to(self.device)
    scaled = float(self.model(x)[0].item())
    return max(math.expm1(scaled * self.std + self.mean), 0.0)
```

`model.eval()` disables dropout — with dropout active, the same item would get a
different prediction every call.

`torch.no_grad()` skips the autograd graph, saving memory and time.

The inverse chain is the exact reverse of `fit`: un-standardise
(`scaled * std + mean`), then `expm1` to undo `log1p`, then floor at 0.

`float(...)` and `math.expm1` keep everything in Python floats, which is the
device-safety fix described above.

### `save()` and `load()`

```python
torch.save({"state": self.model.state_dict(), "mean": self.mean, "std": self.std}, path)
```

**The scaler travels with the weights.** Without `mean` and `std` the weights are
useless — you would have a model predicting standardised log prices and no way to
convert them to dollars. Saving only `state_dict()` is a common and painful
mistake.

This artefact is the bridge between projects: the deal discovery system's
`NeuralAgent` loads this exact file.

---

## `pricing/models/frontier.py`

### The prompt

```python
SYSTEM_PROMPT = (
    "You estimate what a product sells for online in US dollars. "
    "Reply with a number only, no currency symbol and no explanation."
)
```

Short and directive. Asking for "a number only" fights the model's tendency to
explain, which both costs tokens and makes parsing harder.

```python
max_tokens=12
```

A hard cap. Twelve tokens is enough for `"249.99"` and not enough for a
paragraph. It bounds cost and enforces the format even when the model ignores the
instruction.

`temperature=0.0` — the same price for the same product every time, which is
required for the evaluation to mean anything.

### Returning a string

```python
def predict(self, item: Item) -> str:
    response = completion(...)
    # The harness pulls the number out, so a reply like "$249" is fine.
    return response.choices[0].message.content
```

The signature says `str` while every other `predict` returns `float`. That is
deliberate: `evaluation.to_price` handles both, so the extraction logic lives in
exactly one place rather than being duplicated into every LLM-backed predictor.

### `display_name`

```python
self.display_name = f"zero-shot {self.model.split('/')[-1]}"
```

An *instance* attribute, unlike every other model's class attribute. It has to
be — the name depends on which model was passed. `"openai/gpt-4.1-mini"` becomes
`"zero-shot gpt-4.1-mini"`, so a table comparing several providers is readable.

### Why LiteLLM

One interface to every provider. `openai/gpt-4.1-mini`,
`anthropic/claude-opus-4-5`, `gemini/gemini-2.5-flash`, `ollama/llama3.2` all work
through the same `completion()` call. Comparing providers is a CLI flag, and the
local Ollama path is what made it possible to exercise this code without a
working API key.

### The measured result, and its caveat

```
zero-shot llama3.2   avg $264.26   median $79.50   RMSLE 1.438   R² -18.614   hit 31.7%
```

Worse than predicting the training mean. The worst misses were extreme:

```
Projector, CRAZVIEW Projector with WiFi    truth $53.40   guess $4,500.00
LD Products Compatible Toner Cartridge     truth $49.98   guess $2,500.00
```

**The honest caveat:** this is a 3-billion-parameter local model. A frontier model
does substantially better. But the structural point survives the caveat: a
zero-shot LLM has no idea what *this specific* product sells for *now*. It is
pattern-matching on the description and producing a plausible-sounding number
with no anchor.

The fix is not a better prompt — it is **retrieval**. Show the model five similar
products with their real prices and the estimate becomes grounded. That is
precisely what the Frontier agent in the deal discovery project does, and this
result is the evidence for why it needs to.
