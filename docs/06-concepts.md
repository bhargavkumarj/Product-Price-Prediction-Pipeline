# Concepts Explained

The theory behind the project, with worked arithmetic, tied back to the measured
results.

---

## 1. Why price prediction from text is hard

Three properties drive every decision in the project.

**The target is heavy-tailed.** Prices span $0.50 to $999. The measured test
sample has mean $153.68 and median $107.00 — the mean sits well above the median,
which is the signature of a right-skewed distribution.

**The relationship is multiplicative, not additive.** Products are not "$50 more
expensive because they have Bluetooth". They are "40% more expensive because they
are a premium brand". Price responds to features in ratios.

**The signal thins out at the top.** A $200 and an $800 laptop have listings that
read much the same. Above a few hundred dollars, price is driven by brand equity
and market positioning — things the description barely encodes. This shows up
directly in the measured worst misses, where every model under-predicts expensive
branded electronics.

---

## 2. The log transform

### The problem with raw dollars

Squared error on raw prices treats a $100 miss identically at $120 (catastrophic)
and at $900 (reasonable). And because squared error grows quadratically, the
expensive tail dominates the gradient — the model spends its capacity on the few
hundred-dollar items and gets the bulk of the distribution wrong.

### The fix

```python
def to_log(prices):   return np.log1p(np.asarray(prices, dtype=float))
def from_log(values): return np.clip(np.expm1(np.asarray(values, dtype=float)), 0.5, None)
```

Fit on `log(1 + price)`, invert with `expm1`.

In log space, a fixed error is a fixed **ratio**:

| true | predicted | dollar error | log error |
|---|---|---|---|
| $10 | $20 | $10 | 0.66 |
| $500 | $1000 | $500 | 0.69 |

Both are 2× off, and both cost roughly the same in log space. That is what a
pricing decision actually cares about — "within 20%" is a meaningful statement at
every price point; "within $50" is not.

It also matches the data: a right-skewed, roughly log-normal price distribution
becomes approximately symmetric after a log, which is what linear models and
squared-error losses assume.

### Why `log1p` and `expm1` specifically

`log1p(x)` computes `log(1 + x)` with full precision for small `x`, where
`log(1 + x)` computed naively loses significance. It is also defined at `x = 0`,
where plain `log` is not. `expm1` is the exact inverse. Using `log`/`exp` with
manual `+1`/`-1` is numerically worse for cheap items, which are most of the
dataset.

### The floor

```python
np.clip(np.expm1(...), 0.5, None)
```

`expm1` of a sufficiently negative number approaches -1. A negative price is
meaningless, so predictions are floored at $0.50 — the dataset's actual minimum,
which is more honest than flooring at zero.

---

## 3. The metrics, with worked arithmetic

Five metrics, because they disagree and the disagreements are where the
information is.

Worked on a tiny example — four predictions:

| item | truth | guess | error | band |
|---|---|---|---|---|
| A | $100 | $110 | $10 | green (under $40) |
| B | $50 | $95 | $45 | amber ($45 < $80) |
| C | $800 | $750 | $50 | green (50/800 = 6.3% < 20%) |
| D | $60 | $700 | $640 | red |

### Average error (MAE)

```
(10 + 45 + 50 + 640) / 4 = $186.25
```

Intuitive units. Highly sensitive to the tail — item D alone contributes $160 of
the $186.

### Median error

```
sorted: [10, 45, 50, 640]  →  (45 + 50) / 2 = $47.50
```

Describes the typical case. Completely ignores D.

**The gap between the two is the finding.** In the real measured results:

```
random forest    avg $68.72   median $34.22    ratio 2.0×
bow linear       avg $82.97   median $33.55    ratio 2.5×
```

Every model has an average roughly double its median. Half of all predictions are
within about $34; the average is inflated by a minority of large misses. Report
only the average and you misrepresent typical behaviour. Report only the median
and you hide the tail.

### RMSLE

```
squared log errors:
A: (ln(111) - ln(101))² = (4.710 - 4.615)² = 0.0089
B: (ln(96)  - ln(51))²  = (4.564 - 3.932)² = 0.4001
C: (ln(751) - ln(801))² = (6.621 - 6.686)² = 0.0042
D: (ln(701) - ln(61))²  = (6.553 - 4.111)² = 5.9616

RMSLE = sqrt((0.0089 + 0.4001 + 0.0042 + 5.9616) / 4) = sqrt(1.594) = 1.262
```

Note how the ranking changes. In dollars, C ($50 error) looks worse than B ($45
error) — nearly the same. In log terms B is 96× the error of C, because B is
nearly 2× off while C is 6% off.

**RMSLE is the fairest single number for this problem** because it is scale-free.
It also penalises under-prediction more than over-prediction, since log
compresses the upper range — a defensible property for pricing, where
under-valuing an item is usually the more costly mistake.

### R² — and why it goes negative

```
R² = 1 - (SS_residual / SS_total)
   = 1 - Σ(truth - guess)² / Σ(truth - mean_truth)²
```

It compares your model against the trivial model that always predicts the mean.

- `R² = 1.0` — perfect
- `R² = 0.0` — exactly as good as predicting the mean
- `R² < 0` — **worse** than predicting the mean

The last case surprises people, but it follows directly: your squared error is
larger than the mean-predictor's, so the ratio exceeds 1 and `1 - ratio` goes
negative. There is no lower bound.

**The measured case:**

```
linear (bag of words)   avg $82.97   median $33.55   R² -0.899
```

Best median error in the whole comparison, and R² of -0.899. The mechanism is in
the measured worst miss:

```
For Volvo C70 S70 V70 V40 S40 AC Compressor   truth $280.41   guess $2,595.82
```

A $2,315 error contributes 5.4 million to the squared-error sum. Dozens of good
predictions cannot offset it. The median ignores it entirely; R² is dominated by
it.

The sanity check that the whole metric setup is correct: `AveragePricer` — which
literally predicts the mean — measured R² of **-0.001**, i.e. zero to rounding.
That is what it must be by construction.

### Hit rate

```
green: A, C  →  2/4 = 50%
```

The business metric. "What fraction of predictions would a buyer accept?" rather
than "how wrong on average?".

The dual criterion — within $40 **or** within 20% — is what makes it meaningful
across a price range spanning three orders of magnitude. An absolute-only rule
would mark a 6% miss on an $800 item as a failure; a relative-only rule would
mark a $5 miss on a $20 item as a failure. Both would be wrong.

---

## 4. Bag of words

### The representation

Build a vocabulary from the corpus; represent each document as a vector of word
counts.

```
"wireless bluetooth headphones wireless"

vocabulary: [bluetooth, headphones, laptop, wireless, ...]
vector:     [    1,         1,         0,       2,   ...]
```

Extremely sparse — a 20,000-word vocabulary with maybe 60 distinct words per
listing means 99.7% zeros. Scikit-learn stores it as a sparse matrix.

### What it can and cannot do

**Can:** learn that "titanium", "professional" and "4K" correlate with high
prices, and that "plastic" and "compatible" correlate with low ones. That is real
signal, and it is why the bag-of-words model beats the category average by $11.

**Cannot:**
- **Word order.** "cable for laptop" and "laptop for cable" are identical vectors.
- **Synonyms.** "laptop" and "notebook" are unrelated dimensions.
- **Negation.** "does not include battery" scores the battery positively.
- **Bound its output.** Every word adds its coefficient without limit, which is
  exactly the mechanism behind the $2,595 prediction.

### The hyperparameters

`max_features=20_000` — keep the 20,000 most frequent terms. Bounds memory and
drops the long tail of near-unique tokens.

`min_df=2` — drop terms appearing in only one document. With one observation, the
fitted coefficient is pure noise, and such terms are almost always typos or
part-number fragments.

`stop_words="english"` — remove "the", "and", "with". Highest frequency, zero
price signal; leaving them in lets them dominate the count matrix.

### Why Ridge rather than plain least squares

With 20,000 features and 20,000 rows, ordinary least squares is at the edge of
being underdetermined and will assign large coefficients to rare words to fit the
training data exactly. Ridge adds an L2 penalty `α·Σβ²`, shrinking coefficients
toward zero and penalising the large ones hardest.

Even with Ridge at `α=1.0` the model still produced a $2,595 prediction. Stronger
regularisation would help, and the real fix is a model class that cannot
extrapolate — which is the next section.

---

## 5. TF-IDF and SVD

### TF-IDF

```
tfidf(term, doc) = tf(term, doc) × log(N / df(term))
```

Term frequency times inverse document frequency. A term appearing in every
listing has `df ≈ N`, so `log(N/df) ≈ 0` and it is down-weighted to nothing. A
term appearing in 2% of listings gets a large multiplier.

For price prediction this matters a lot. "wireless" in half the listings is
weakly informative. "titanium" in 2% is strongly informative. Raw counts treat
them by frequency; TF-IDF treats them by distinctiveness.

### Truncated SVD / LSA

Factorise the `documents × terms` matrix into `documents × k` and `k × terms`,
keeping the `k` directions of greatest variance. With `k = 300`, 40,000 sparse
dimensions become 300 dense ones.

Truncated SVD on a TF-IDF matrix is **Latent Semantic Analysis**. The components
capture co-occurrence structure: terms that appear together load onto the same
component. So "laptop", "notebook" and "ultrabook" collapse toward a shared
direction, and the model can generalise across them — something raw bag-of-words
structurally cannot do, since every term is an independent dimension.

### Why it fixed the negative R²

```
bow linear     R² -0.899
svd + ridge    R²  0.322
```

Same underlying information, different conditioning. With 300 dense features
instead of 40,000 sparse ones, no single rare word owns a dimension, so no single
word can drive an unbounded prediction. The extreme over-predictions disappear
and R² becomes positive.

---

## 6. Trees, forests and boosting

### Why trees fixed what linear models could not

A decision tree splits on feature thresholds and predicts **the mean of the
training items in each leaf**. That means its output is **bounded by the training
data** — it can never predict $2,595 unless it has seen items around $2,595.

The bag-of-words linear model's catastrophic failure mode is structurally
impossible in a tree. That single property explains most of the ranking in the
measured table.

### Random forest — bagging

Fit 100 trees, each on a bootstrap sample of the rows and a random subset of
features at each split, then average their predictions.

Each individual tree overfits its sample. Because the samples and features
differ, the trees overfit in *uncorrelated* ways, and averaging cancels the
variance while preserving the shared signal. Decorrelation is the whole
mechanism — that is why the feature subsampling matters as much as the row
bootstrap.

**Measured: best average error ($68.72), best RMSLE (0.705), best hit rate
(60%).**

But R² of only 0.316, below the neural net's 0.444. The same bounded-output
property that prevents catastrophes also caps the upside: averaging 100 trees
pulls predictions toward the middle, so the forest rarely predicts $800 even for
a genuinely $840 item. It is reliable and conservative.

### Gradient boosting — the other ensemble

Trees are fit **sequentially**, each one trained on the *residuals* of the
ensemble so far. Formally, each tree approximates the negative gradient of the
loss with respect to the current predictions.

Bagging reduces **variance** (averaging independent overfit models). Boosting
reduces **bias** (each step corrects what is still wrong). Boosting usually wins
on tabular data and usually needs more careful regularisation, which is why
`XGBoostPricer` has seven tuning parameters and `RandomForestPricer` has three.

Not in the measured table — this machine lacks the OpenMP runtime XGBoost links
against. Stated rather than estimated.

---

## 7. The residual network

### The architecture

```
5000 (hashed features)
  → Linear(5000, 1024) → LayerNorm → ReLU → Dropout
  → ResidualBlock × 4                          each: Linear→Norm→ReLU→Drop→Linear→Norm, then +x, then ReLU
  → Linear(1024, 1)
```

13,537,281 trainable parameters.

### Why residual connections

```python
def forward(self, x):
    return self.activation(self.block(x) + x)
```

The `+ x` means the block learns a *correction* to its input rather than a full
transformation. Two consequences:

1. **Gradient flow.** Backpropagation has a path to early layers that does not
   pass through every weight matrix, which is what prevents vanishing gradients
   in deep stacks.
2. **Identity is the default.** If a block's weights are near zero it passes its
   input through unchanged, so adding depth cannot make the network worse in
   principle.

This is the ResNet insight, and it applies to MLPs as much as to convolutional
networks.

### LayerNorm, not BatchNorm

LayerNorm normalises across features within a single example. BatchNorm
normalises across the batch.

LayerNorm is independent of batch size, behaves identically in training and
inference, and keeps no running statistics — so there is no train/eval
discrepancy to get wrong. With sparse binary inputs and a batch of 128, BatchNorm
statistics would also be noisy.

### The hashing trick

```python
HashingVectorizer(n_features=5000, stop_words="english", binary=True)
```

Rather than building a vocabulary, hash each token into one of 5,000 buckets.

| Property | Consequence |
|---|---|
| No vocabulary to fit | `transform` works immediately; train and test are handled identically |
| Nothing to persist | The saved model is weights + two floats; the vectoriser is a constant |
| Fixed memory | 5,000 features regardless of corpus size |
| Hash collisions | Different words share buckets — mild feature noise |
| No inverse mapping | Not interpretable at the feature level |

For a neural network the costs are acceptable and the "nothing to persist"
property is genuinely valuable — it is why the deal discovery project can load
this model with three lines and a single `.pt` file.

### The loss

`SmoothL1Loss` (Huber): quadratic for small errors, linear for large ones.

The deliberate middle ground. MSE lets outliers dominate the gradient — precisely
the failure that gives the bag-of-words model R² of -0.899. Pure L1 has a
constant gradient magnitude, which converges poorly near the optimum. Huber is
quadratic where you want fine convergence and linear where you want robustness.

### Two stacked target transforms

```python
y = torch.log1p(prices)                    # 1. log, as everywhere else
self.mean, self.std = float(y.mean()), float(y.std())
y = (y - self.mean) / self.std             # 2. standardise
```

The log is for the reasons in section 2. The standardisation is for the
optimiser: AdamW's default learning rate assumes a target near zero mean and unit
variance, and log prices centre around 4.7.

Both must be inverted at prediction time, in reverse order — and the scaler must
be saved with the weights, or the model is useless.

### Measured result

```
residual net   avg $71.36   median $39.39   RMSLE 0.750   R² 0.444   hit 51.5%
```

**Best R² in the comparison.** It tracks the expensive end better than anything
else, because unlike a tree it can extrapolate. But its median error is worse
than the forest's — it is less reliable on the bulk of cheap items.

That is a genuine trade-off, not a ranking. Which model you ship depends on
whether you care more about typical accuracy or about not being badly wrong on
expensive items.

---

## 8. Zero-shot LLM pricing

### The setup

No training. One prompt: "You estimate what a product sells for online in US
dollars. Reply with a number only."

### Measured result

```
zero-shot llama3.2   avg $264.26   median $79.50   RMSLE 1.438   R² -18.614   hit 31.7%
```

Worse than predicting the training mean, by a wide margin. The worst misses:

```
Projector, CRAZVIEW Projector with WiFi   truth $53.40   guess $4,500.00
LD Products Compatible Toner Cartridge    truth $49.98   guess $2,500.00
Clementoni 1000 Piece Jigsaw Puzzle       truth $79.95   guess $1,500.00
```

Not just wrong — wrong by two orders of magnitude, and confidently.

### The honest caveat and the real lesson

The measured run used a local 3-billion-parameter model. A frontier model does
substantially better, and it would be dishonest to present this number as
representative of GPT-class performance.

But the structural point survives the caveat: **a zero-shot LLM has no anchor**.
It has never seen this product's current market price. It pattern-matches on the
description and produces a plausible-sounding number. Note the round figures —
$4,500, $2,500, $1,500. That is the signature of guessing, not estimating.

### The fix is retrieval, not prompting

No amount of prompt engineering gives the model information it does not have. The
fix is to *supply* the information: retrieve five similar products with their
real prices and put them in the prompt.

That is exactly what the Frontier agent in the deal discovery project does, and
these numbers are the evidence for why it has to. A zero-shot LLM is a weak
pricer; an LLM with retrieved comparables is a strong one.

---

## 9. Data curation — the highest-leverage work

The models here are standard. The curation is where the project's actual
engineering is.

### Rejection is most of the job

```python
MIN_PRICE = 0.5        # below this, a data error or shipping-only listing
MAX_PRICE = 999.49     # above this, price is brand-driven, not description-driven
MIN_CHARS = 600        # below this, nothing for a text model to work with
```

Most raw Amazon rows are discarded. That is correct. Keeping a row a model cannot
learn from adds irreducible error and teaches it nothing.

`999.49` is chosen so `round()` gives 999 — every price is at most three digits,
which keeps the fine-tuning prompt format consistent.

### Deduplication order matters

Shuffle **before** deduplicating. Dedup keeps the first occurrence; without a
shuffle, "first" means "first in catalogue order", which correlates with category
and listing age. Shuffling makes the survivor of each duplicate group random and,
being seeded, reproducible.

Two passes — unique titles, then unique bodies — because they are different
duplicates. Same title is the same product listed twice; same body is the same
product under different titles.

A duplicate spanning train and test is a **label leak** and makes the test score
meaningless.

### Price-balanced sampling

```python
scaled  = (prices - prices.min()) / (prices.max() - prices.min() + 1e-9)
weights = scaled ** 2
weights[categories == "Automotive"] *= 0.05
weights[categories == "Tools_and_Home_Improvement"] *= 0.5
```

Raw marketplace data is mostly cheap items in two categories. Train on it and the
model learns the base rate — always guess about $20 — which minimises training
error and is useless.

Squaring the scaled price makes a $900 item roughly 320× more likely to be
sampled than a $50 one, producing a flatter target distribution the model can
learn a *range* from.

Automotive is cut to 5% because it is the largest category and its listings are
mostly interchangeable parts whose price depends on the vehicle, not the text.

**This is deliberately unrepresentative sampling**, and it is the right call: the
goal is a model that can price across a range, not one that reproduces Amazon's
category mix.

### The final shuffle is load-bearing

```python
random.Random(settings.seed).shuffle(sample)
```

`np.random.choice` returns indices in **ascending order**. `split()` then slices.
Without this shuffle, the cheapest items would go to train and the most expensive
to test — a catastrophic distribution shift that would look like "the model
cannot predict expensive items" when the real cause is that it never saw any.

### Summarise once, not per prediction

An LLM rewrites each listing into five normalised lines during curation. Every
model reads `item.summary`.

If this happened at prediction time, the bag-of-words baseline would cost an LLM
call per prediction — making the cheap alternative more expensive than the
expensive one. Paying once at curation is what keeps the classical models cheap.

It also means every model sees **the same normalised text**, so the comparison
measures algorithms rather than input preprocessing choices.
