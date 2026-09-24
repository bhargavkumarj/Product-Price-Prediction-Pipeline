# Interview Questions — Product Price Prediction Pipeline

30 questions with full answers, grouped by what they test. The strongest answers
cite the measured numbers, so those are included.

---

## Part 1 — The data pipeline (1–8)

### Q1. Walk me through your data pipeline.

Five stages, each dropping data deliberately.

**Parse.** Each raw Amazon row goes through `parse()`, which rejects it if the
price is missing or outside $0.50–$999.49, or if there is under 600 characters of
text. Surviving rows have catalogue noise removed — part numbers, Best Sellers
Rank, battery flags — and get flattened to at most 4,000 characters.

**Deduplicate.** Shuffle first, then keep one of each unique title, then one of
each unique body. Two passes because they catch different duplicates.

**Balance.** Sample proportional to scaled price squared, with Automotive damped
to 5% and Tools to 50%.

**Split.** Slice into train/validation/test — which is fair only because the list
was just shuffled.

**Summarise.** An LLM rewrites each listing into a fixed five-line format, once,
so every downstream model reads the same normalised text.

The thing I would emphasise is that curation is mostly *subtractive*. Most raw
rows are discarded, and that is correct — a row a model cannot learn from only
adds irreducible error.

### Q2. Why did you deliberately skew your training distribution?

Raw marketplace data is overwhelmingly cheap items in two categories. A model
trained on that learns the base rate — always guess about $20 — which minimises
training error and is useless as a pricer.

So I sample proportional to scaled price squared. A $900 item becomes roughly
320× more likely to be sampled than a $50 one. I also damp Automotive to 5%
because it is the largest category and its listings are mostly interchangeable
parts whose price depends on the vehicle rather than the description.

The result is a flatter target distribution the model can learn a *range* from.

The caveat I would raise unprompted: train and test both come from the balanced
pool, so the evaluation is internally consistent, but it is **not** an estimate
of performance on the raw Amazon distribution. If that were the deployment
distribution I would either re-weight at evaluation time or report metrics per
price bucket.

### Q3. Why shuffle before deduplicating?

Deduplication keeps the first occurrence of each duplicate group. Without a
shuffle, "first" means "first in catalogue order", which correlates with category
and listing age — so the survivor of every group is systematically biased rather
than random.

I shuffle with a seeded `random.Random(42)` so it is reproducible.

The reason deduplication matters at all is leakage: a duplicate that lands in
both train and test means the model has memorised the answer, and the test score
is inflated by an amount you cannot measure.

### Q4. You have a final shuffle after sampling. Why?

That one is load-bearing and it prevents a genuinely nasty bug.

`np.random.choice` returns indices in **ascending order**. `split()` then slices
the list into train, validation and test. Without a shuffle in between, the
cheapest items go to train and the most expensive to test.

Every model would then appear unable to predict expensive items, and every metric
would agree. There would be no error anywhere — just consistently bad results
with an invisible cause. That class of bug is the most dangerous in an ML
pipeline, because it produces plausible output.

### Q5. Why do you strip part numbers with a regex?

Part numbers are near-unique strings. In a bag-of-words model each one becomes
its own vocabulary entry appearing exactly once, so its coefficient is fit to a
single observation — pure noise that also crowds out real words under the
`max_features` cap. For an LLM they consume tokens and invite spurious pattern
matching.

The regex uses three lookaheads: at least seven uppercase-alphanumeric characters
to a word boundary, containing at least one letter and at least one digit. So
`WH1000XM4` and `B08N5WRWNW` match, while `BLUETOOTH` (no digit), `2023` (no
letter) and `USB` (too short) survive.

I also tell the summarising LLM not to include part numbers. Belt and braces,
because the regex only catches uppercase runs.

### Q6. Why summarise the listings, and why as a separate stage?

Raw listings are up to 4,000 characters of marketing copy, spec tables and
shipping notes. Two products at the same price can have completely different
listing styles, which is noise for every model. The LLM rewrites each into five
fixed lines — title, category, brand, description, details.

Making it a separate stage matters for two reasons.

**Cost.** If it happened at prediction time, the bag-of-words baseline would cost
an LLM call per prediction — making the cheap alternative more expensive than the
expensive one it is meant to be compared against.

**Comparability.** Every model reads the same normalised text through
`Item.text`. Without that, comparing models would partly be comparing
preprocessing choices.

It is also resumable: `if not item.summary` means an interrupted 400,000-item run
picks up where it stopped, which is not optional at that scale.

### Q7. Why processes for parsing but threads for evaluation?

Different bottlenecks.

Parsing is CPU bound — JSON decoding and regex over millions of rows. The GIL
serialises threads for that, so threads would give no speedup at all. I use
`ProcessPoolExecutor` with 1,000-row chunks, because dispatching single rows to a
process pool spends all its time on inter-process serialisation.

Evaluation and summarisation are waiting — on HTTP for the LLM paths, or on
native numpy and torch code that releases the GIL for the local models. Threads
are the right tool, and they share the fitted model, which processes could not do
without pickling a 13-million-parameter network per worker.

This constraint is also why `parsing.py` is pure functions with no state:
everything a process worker touches has to be picklable and side-effect free.

### Q8. Why does `parse()` return `None` instead of raising?

Because rejection is the normal outcome, not an error. Most raw rows are
rejected. Using exceptions for the common path would mean try/except around every
row, and would conflate "this row is not useful" with "something is broken" —
which are things you want to handle very differently.

The caller filters with a list comprehension, which is the natural shape.

---

## Part 2 — Modelling (9–17)

### Q9. Why do you fit on log prices?

Prices span $0.50 to $999 with a long right tail. Two problems with raw dollars.

Squared error treats a $100 miss identically at $120, where it is catastrophic,
and at $900, where it is reasonable. And because squared error grows
quadratically, the expensive tail dominates the gradient — the model spends its
capacity on a few hundred items and gets the bulk of the distribution wrong.

In log space, equal errors are equal *ratios*. Being 2× off costs about 0.65
either at $10 or at $500. That matches what a pricing decision cares about —
"within 20%" is meaningful at every price point, "within $50" is not — and it
matches the data, since the price distribution is roughly log-normal.

I use `log1p`/`expm1` rather than `log`/`exp` with a manual ±1, because `log1p`
keeps precision for small values and most of this dataset is small values.
Predictions are clipped at $0.50, the dataset's actual minimum.

### Q10. Explain your results table.

```
model                  avg $ err  median $  RMSLE  R²      hit rate
random forest          68.72      34.22     0.705  0.316   60.0%
svd + ridge            69.70      34.86     0.756  0.322   56.0%
residual net           71.36      39.39     0.750  0.444   51.5%
linear (bag of words)  82.97      33.55     0.805  -0.899  55.0%
linear (features)      83.75      45.25     0.915  0.067   44.0%
category average       93.56      66.11     1.026  0.113   29.0%
training average       106.08     90.42     1.154  -0.001  18.5%
```

Read as a ladder. Predicting the training mean is $106 off. Knowing only the
category buys $12.50 — that is the value of one categorical feature with eight
levels, and it is the yardstick for everything else.

Four hand-built numeric features buy another $10. The first real text model — a
linear regression on word counts — buys another $9 but has a negative R². Models
that handle the text non-linearly fix that, and the random forest wins on average
error, RMSLE and hit rate.

The residual net has the best R² at 0.444, meaning it tracks expensive items
better than anything else, but a worse median — it is less reliable on the bulk
of cheap items. That is a genuine trade-off rather than a ranking.

`AveragePricer` scoring R² of -0.001 is my sanity check that the metric setup is
correct: a model that predicts the mean must score zero by construction.

### Q11. Your bag-of-words model has the best median error and a negative R². Explain that.

It is the most instructive row in the table.

R² measures variance explained relative to predicting the mean. Negative means
worse than the mean — your squared error exceeds the mean-predictor's, so the
ratio is above 1 and `1 - ratio` goes negative. There is no lower bound.

The mechanism is in the worst miss:

```
For Volvo C70 S70 V70 V40 S40 AC Compressor   truth $280.41   guess $2,595.82
```

Linear models extrapolate without bound. A rare word with a large positive
coefficient, appearing several times in one listing, produces an arbitrarily
large prediction. A single $2,315 error contributes 5.4 million to the
squared-error sum — dozens of good predictions cannot offset it.

Meanwhile the median ignores the tail entirely, so it stays excellent at $33.55,
the best in the table.

Both numbers are true and they describe different things: the model is usually
very good and occasionally catastrophic. That is exactly why I report five
metrics — any single one gives a false picture of this model.

### Q12. Why did the tree models fix that?

Because a decision tree predicts the **mean of the training items in its leaf**,
so its output is bounded by the training data. It can never predict $2,595 unless
it has seen items around $2,595. The linear model's failure mode is structurally
impossible.

A random forest then averages 100 such trees, each on a bootstrap sample with a
random feature subset. The trees overfit in *uncorrelated* ways, so averaging
cancels the variance while preserving the shared signal.

The result is the best average error, RMSLE and hit rate in the table — but only
0.316 R², below the neural net's 0.444. The same bounded-output property that
prevents catastrophes also caps the upside: averaging pulls predictions toward
the middle, so the forest rarely predicts $800 even for a genuinely $840 item.
Reliable and conservative.

### Q13. What does SVD do in your pipeline and why did it help?

TF-IDF produces a 40,000-dimension sparse matrix. Truncated SVD factorises it and
keeps the 300 directions of greatest variance, giving dense 300-dimension
vectors. On a TF-IDF matrix that is Latent Semantic Analysis.

It helps in two ways. **Generalisation** — terms that co-occur load onto the same
component, so "laptop", "notebook" and "ultrabook" collapse toward a shared
direction and the model can generalise across them. Raw bag-of-words structurally
cannot, because every term is an independent dimension.

**Conditioning** — R² went from -0.899 to 0.322 with the same underlying
information. With 300 dense features rather than 40,000 sparse ones, no single
rare word owns a dimension, so no single word can drive an unbounded prediction.
The extreme over-predictions disappear.

One implementation detail: I clamp `n_components` to `matrix.shape[1] - 1`,
because SVD cannot produce more components than the matrix has columns minus one.
Without that, the model crashes on a small dataset.

### Q14. Explain your neural network architecture and why residual connections.

5,000 hashed input features → a 1,024-wide stem → four residual blocks at 1,024 →
a single output. 13.5 million parameters.

Each residual block is Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm,
and then `forward` returns `ReLU(block(x) + x)`.

**The `+ x` is the point.** The block learns a *correction* to its input rather
than a full transformation. That gives backpropagation a path to early layers
that does not pass through every weight matrix, which prevents vanishing
gradients. And it makes identity the default: if a block's weights are near zero
it passes its input through unchanged, so adding depth cannot make the network
worse in principle.

**LayerNorm rather than BatchNorm** because LayerNorm normalises across features
within one example. It is independent of batch size, behaves identically in
training and inference, and keeps no running statistics — so there is no
train/eval discrepancy to get wrong.

The output is a single unbounded number — a standardised log price — with no
final activation, because the target can be any real number after
standardisation.

### Q15. What is the hashing trick and what does it cost you?

Instead of building a vocabulary, `HashingVectorizer` hashes each token into one
of 5,000 buckets.

**What it buys:** no vocabulary to fit, so `transform` works identically on train
and test; nothing to persist beyond the weights; and fixed memory regardless of
corpus size.

**What it costs:** hash collisions, where different words share a bucket, and no
inverse mapping, so the model is not interpretable at the feature level.

For a 13-million-parameter network both costs are acceptable — collisions act
like mild feature noise, and it was never interpretable anyway.

The "nothing to persist" property turned out to be genuinely valuable: my deal
discovery project loads this model with three lines and a single `.pt` file,
because the vectoriser is reconstructed from a constant rather than loaded from
an artefact.

### Q16. Why `SmoothL1Loss` and not MSE?

Huber loss is quadratic for small errors and linear for large ones.

MSE lets outliers dominate the gradient — which is exactly the failure that gives
my bag-of-words model R² of -0.899. One unusual item with an extreme price would
pull the whole network toward it. Pure L1 avoids that but has constant gradient
magnitude, so it converges poorly near the optimum.

Huber is quadratic where I want fine convergence and linear where I want
robustness. It is the same reasoning as the log transform, applied at the loss
level: the model should not be allowed to obsess over a handful of extreme items.

I also clip gradient norms at 1.0 for the same reason — one pathological batch
can otherwise destroy weights built over many epochs.

### Q17. Your zero-shot LLM was terrible. Is that a fair test?

Partly not, and I say so in the README. The measured run used a local
llama3.2 — a 3-billion-parameter model — because the API key available at the
time was not working. A frontier model does substantially better, and it would be
dishonest to present $264 average error as representative of GPT-class
performance.

But the structural point survives the caveat. Look at the failures:

```
Projector with WiFi       truth $53.40   guess $4,500.00
Toner Cartridge           truth $49.98   guess $2,500.00
1000 Piece Jigsaw Puzzle  truth $79.95   guess $1,500.00
```

Round numbers — $4,500, $2,500, $1,500. That is the signature of *guessing*, not
estimating. A zero-shot LLM has never seen this product's current market price;
it pattern-matches the description and produces a plausible-sounding figure.

The fix is not a better prompt, because no prompt supplies information the model
does not have. The fix is retrieval: show it five similar products with their
real prices. That is exactly what the Frontier agent in my deal discovery project
does, and this result is the evidence for why it has to.

---

## Part 3 — Evaluation (18–24)

### Q18. Why five metrics?

Because they disagree, and the disagreements are the information.

The clearest case is the bag-of-words model: best median error in the table, and
R² worse than predicting the mean. Pick either as *the* metric and you reach a
completely different, and wrong, conclusion.

They answer different questions. Average error asks how wrong on average and is
sensitive to the tail. Median asks how wrong typically and ignores the tail.
RMSLE asks how wrong proportionally and is scale-free. R² asks whether you beat
the mean and is the most tail-sensitive of all. Hit rate asks how often the
prediction is usable at all.

The gap between average ($68.72) and median ($34.22) is itself a finding — every
model has an average roughly double its median, which tells you the error
distribution is skewed and a minority of large misses is driving the headline
number.

### Q19. What is RMSLE and why use it here?

Root Mean Squared Log Error: take the log of prediction and truth, subtract,
square, average, square-root.

It is **scale-free**. Being 2× off costs roughly the same at $10 and at $500,
because it operates on ratios. For a target spanning three orders of magnitude
that makes it the fairest single number in the table.

A worked comparison from my docs: a $45 error on a $50 item and a $50 error on an
$800 item look almost identical in dollars. In log terms the first is 96× the
error of the second, because one is nearly 2× off and the other is 6% off. RMSLE
captures what your intuition says; raw dollar error does not.

It also penalises under-prediction slightly more than over-prediction, since log
compresses the upper range. For pricing that is defensible — under-valuing an
item is usually the costlier mistake.

### Q20. Explain negative R² to someone who has not seen it.

R² is `1 - (your squared error / the mean-predictor's squared error)`. It asks:
how much better are you than just guessing the average every time?

1.0 is perfect. 0.0 means exactly as good as guessing the average. Negative means
**worse** than guessing the average — your squared error is larger, so the ratio
exceeds 1 and `1 - ratio` goes below zero. There is no lower bound.

People find it surprising because they think of R² as "percentage of variance
explained" and percentages do not go negative. But that framing only holds for a
model fit by least squares on the same data it is evaluated on. On held-out data
with an extrapolating model, negative R² is entirely normal.

My sanity check: `AveragePricer`, which literally predicts the mean, scored
-0.001 — zero to rounding, exactly as it must.

### Q21. Why does your hit rate use two criteria?

```python
if error < 40 or error / truth < 0.2:
```

Within $40 **or** within 20%, and passing either is enough.

A $5 error on a $20 item is 25% — it fails the relative test but passes the
absolute one, and rightly so, because nobody cares about $5. A $60 error on a
$900 item is 6.7% — it fails the absolute test but passes the relative one, and
rightly so, because that is a good estimate of an expensive item.

Either rule alone mislabels one end of a price range spanning three orders of
magnitude. The dual rule mirrors how a person actually judges an estimate: close
in dollars *or* close in proportion.

I like hit rate as the metric to lead with for a non-technical audience, because
"60% of predictions are usable" is a sentence anyone can act on, whereas "RMSLE
0.705" is not.

### Q22. How do you make sure the model comparison is fair?

Several things, and they were all deliberate.

Every model goes through **one harness**. `evaluate()` takes any callable and
does identical work regardless of what is behind it.

Every model is scored on **exactly the same items** — `items[:size]`,
deterministically, not a random sample per model. A random sample would introduce
between-row variance that has nothing to do with the models.

Every model is fit on **the same train and validation splits**, in the same
process, within one `cmd_compare` loop.

Every model reads **the same input text** via `Item.text`, so I am comparing
algorithms and not preprocessing choices.

And the test items were never touched during development — the split happens once
during curation, with a fixed seed.

### Q23. What did the error analysis tell you?

More than the aggregate metrics did. The same items appear in every model's
worst-miss list:

```
Canon EOS Rebel T7 Digital SLR Camera   truth $819.00   guess $183.26
Dell XPS 13 9343-2727SLV                truth $839.99   guess $392.70
Bam France 2002XL Contoured Hightech    truth $713.00   guess $143.81
```

All expensive branded electronics, all under-predicted, by every model including
the neural net.

That is a **data** finding, not a model finding. At the top of the price range,
the description reads much the same whether the item costs $200 or $800 — the
price is driven by brand equity and market position, which the text barely
encodes.

The actionable conclusion is that trying a seventh model would not fix it. Better
*features* might: brand as an explicit extracted field rather than buried in
prose, or separate models per price band. That is a very different next step from
"tune the hyperparameters", and I only got there by looking at individual
failures.

### Q24. XGBoost is in your code but not in your results. Why?

This machine is missing the OpenMP runtime that XGBoost's native library links
against, so `import xgboost` fails. I could not run it, so I did not put a number
in the table — the README says plainly that it was not run.

Two things I would point out about how that was handled.

First, it did not break anything else. My model registry holds
`(module_path, class_name)` strings and imports on demand, so only asking for
XGBoost touches XGBoost. That design exists *because* of this — the first version
imported everything eagerly and the entire CLI died, including `--help`, on a
machine missing one optional dependency.

Second, I would expect XGBoost to be competitive with or better than the random
forest here. Boosting usually beats bagging on tabular data because it reduces
bias rather than variance. But that is an expectation, not a measurement, and I
would not put an expectation in a results table.

---

## Part 4 — Engineering and judgement (25–30)

### Q25. Tell me about a bug you found in this project.

The one I find most instructive is the neural network's target scaler.

The network predicts a standardised log price, so `predict` has to un-standardise
and then exponentiate. I stored `mean` and `std` as PyTorch tensors and saved
them in the checkpoint alongside the weights.

It worked perfectly in training. It failed when a *different project* loaded the
saved weights:

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, mps:0 and cpu!
```

The cause: `torch.load(path, map_location=self.device)` moved the scaler tensors
to MPS, while the prediction path computed on CPU. In the training path the
scaler had never left CPU, so the bug was invisible.

The fix was to store them as plain Python floats, which removes the device from
the problem entirely.

Two lessons. First, a value that is not a model parameter should not be a tensor
— being a tensor is what dragged device semantics into a piece of arithmetic that
did not need it. Second, the bug only surfaced at an integration boundary, which
is the argument for the cross-project integration being a real test rather than a
nice-to-have.

### Q26. Tell me about a design decision you had to revise.

The model registry.

The first version imported every model class at the top of `models/__init__.py`.
Then running `python run.py compare --models average` — which does not touch
XGBoost at all — produced:

```
xgboost.core.XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded.
Likely causes: OpenMP runtime is not installed
```

The entire CLI was dead, including `--help`, because of one optional model's
missing native dependency.

I rewrote the registry to hold `(module_path, class_name)` strings and resolve
them with `import_module` only when asked. `XGBoostPricer.fit` also imports
xgboost inside the method as a second line of defence.

That let me produce the whole results table on a machine where XGBoost cannot
load. The principle I took from it: **optional dependencies should fail at the
point of use, not at import.** A tool that dies on `--help` because of a model
you did not ask for is broken.

### Q27. How would you improve the accuracy from here?

In the order I would actually try them, which is driven by the error analysis
rather than by which model is fashionable.

**Better features first.** Every model under-predicts expensive branded
electronics. Brand is currently buried in prose. Extracting it as a categorical
feature and one-hot encoding it is cheap and targets the largest identified error
source directly.

**Ensemble the models I already have.** The forest is conservative and reliable;
the neural net has the best R² and tracks the expensive tail. They fail
differently, which is precisely when averaging helps. I have per-prediction data
in the JSON reports, so I can check the error correlation before building
anything.

**Fine-tune a small language model.** The `Item.make_prompt`/`test_prompt` pair
already exists for this. A QLoRA fine-tune of a 3B model on 400,000 curated
examples learns product-price associations in a way bag-of-words cannot, and it
is the natural next rung on the ladder.

**Retrieval-augmented pricing.** Given the zero-shot results, this is the
highest-ceiling option: retrieve similar products with known prices and put them
in the prompt. That is what I built in the deal discovery project.

**More data last.** 20,000 training rows is small. But I would do it last,
because the error analysis says the problem is missing *signal*, not missing
*examples* — the descriptions of a $200 and an $800 laptop genuinely look alike.

### Q28. How would you deploy this?

**Serving.** The classical models are milliseconds per prediction and pickle
cleanly. I would fit once, persist with joblib, and serve behind FastAPI. The
neural net already has `save`/`load`. `cmd_predict` currently refits on every
call, which is fine for a CLI and obviously wrong for a service.

**Retraining.** A scheduled job that re-runs curation on new data and retrains.
The key thing is to gate promotion on the evaluation harness — a new model ships
only if it beats the current one on the same held-out items.

**Monitoring.** Log every prediction with its input. Without ground truth you
cannot compute error live, so I would watch the *prediction distribution* for
drift — if the mean predicted price shifts, either the input mix changed or the
model degraded, and both need investigating.

**The metric to alert on** is hit rate, not average error, because it is the one
tied to a business threshold.

**Fallbacks.** The frontier path depends on an external API. The random forest
runs locally in milliseconds and is the obvious fallback when the API is down.

### Q29. When would you use the LLM over the classical models?

Given my numbers, not for this task as configured. The random forest is $68
average error, runs locally in about a millisecond, and costs nothing. The
zero-shot LLM was $264 and needs a network call per prediction.

The LLM wins in three situations.

**Cold start.** With no training data at all, a zero-shot LLM gives you something
immediately. The random forest needs 20,000 labelled examples that may not exist.

**Novel categories.** The classical models only know words they saw in training.
A product category that did not exist at training time is a blind spot for them
and not necessarily for an LLM.

**With retrieval.** This is the real answer. The failure mode in my results is
that the LLM has no anchor — it guesses round numbers like $4,500. Give it five
similar products with real prices and it becomes a strong estimator, because it
is now doing comparison rather than recall.

In practice I would ensemble them, which is exactly what I did in the deal
discovery project: an LLM with retrieval, a fine-tuned specialist, and a local
neural net, weighted together.

### Q30. What would you do differently if you started again?

**Extract brand during parsing.** The error analysis points at brand as the
largest missing signal, and I am currently asking every model to infer it from
prose. A single categorical feature would likely beat several of my modelling
refinements.

**Report metrics by price bucket from the start.** My aggregates hide that
performance is very different at $20 and at $800. Bucketed metrics would have
surfaced the expensive-item problem on the first run rather than through manual
error analysis.

**Write the evaluation harness before any model.** I did roughly this, but I
would be stricter about it. The harness is what made every subsequent decision
evidence-based, and it is the most reused file in the project.

**Check for duplicates between splits explicitly.** I deduplicate before
splitting, which should be sufficient, but I would rather assert it than assume
it — a cheap test that catches the most damaging possible bug.

What I would keep: the log transform everywhere, the single `Pricer` contract,
the lazy registry, reporting five metrics, and storing every individual
prediction in the report. Those are the choices that turned the project from a
set of models into something I can reason about.
