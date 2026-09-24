"""Classic ML on product text: features, bag of words, embeddings, trees.

Prices are log-transformed before fitting. The target spans $0.50 to $999 with a
long tail, so squared error on raw dollars lets a handful of expensive items
dominate every gradient; in log space the models optimise relative error, which
is what a pricing decision actually cares about.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge

from pricing.items import Item
from pricing.models.base import Pricer

MAX_TRAIN_ROWS = 50_000


def to_log(prices) -> np.ndarray:
    return np.log1p(np.asarray(prices, dtype=float))


def from_log(values) -> np.ndarray:
    return np.clip(np.expm1(np.asarray(values, dtype=float)), 0.5, None)


class FeaturePricer(Pricer):
    """Four hand-built numeric features — how far do the obvious signals get you?"""

    display_name = "linear (features)"

    @staticmethod
    def features(item: Item) -> dict:
        text = item.text
        return {
            "weight": item.weight or 0.0,
            "weight_missing": int(not item.weight),
            "text_length": len(text),
            "word_count": text.count(" ") + 1,
        }

    def fit(self, train, validation=None):
        frame = pd.DataFrame([self.features(item) for item in train])
        self.model = LinearRegression().fit(frame, to_log([item.price for item in train]))
        return self

    def predict(self, item: Item) -> float:
        frame = pd.DataFrame([self.features(item)])
        return float(from_log(self.model.predict(frame))[0])


class BagOfWordsLinearPricer(Pricer):
    display_name = "linear (bag of words)"

    def __init__(self, max_features: int = 20_000):
        self.vectorizer = CountVectorizer(
            max_features=max_features, stop_words="english", min_df=2
        )

    def fit(self, train, validation=None):
        train = train[:MAX_TRAIN_ROWS]
        matrix = self.vectorizer.fit_transform(item.text for item in train)
        self.model = Ridge(alpha=1.0).fit(matrix, to_log([item.price for item in train]))
        return self

    def predict(self, item: Item) -> float:
        matrix = self.vectorizer.transform([item.text])
        return float(from_log(self.model.predict(matrix))[0])


class WordVectorPricer(Pricer):
    """TF-IDF reduced with SVD — dense vectors instead of sparse word counts."""

    display_name = "svd + ridge"

    def __init__(self, components: int = 300):
        self.components = components
        self.vectorizer = TfidfVectorizer(max_features=40_000, stop_words="english", min_df=2)

    def fit(self, train, validation=None):
        from sklearn.decomposition import TruncatedSVD

        train = train[:MAX_TRAIN_ROWS]
        matrix = self.vectorizer.fit_transform(item.text for item in train)
        self.svd = TruncatedSVD(n_components=min(self.components, matrix.shape[1] - 1), random_state=42)
        reduced = self.svd.fit_transform(matrix)
        self.model = Ridge(alpha=1.0).fit(reduced, to_log([item.price for item in train]))
        return self

    def predict(self, item: Item) -> float:
        reduced = self.svd.transform(self.vectorizer.transform([item.text]))
        return float(from_log(self.model.predict(reduced))[0])


class RandomForestPricer(Pricer):
    display_name = "random forest"

    def __init__(self, max_features: int = 10_000, trees: int = 100, rows: int = 20_000):
        self.rows = rows
        self.trees = trees
        self.vectorizer = CountVectorizer(
            max_features=max_features, stop_words="english", min_df=2
        )

    def fit(self, train, validation=None):
        train = train[: self.rows]
        matrix = self.vectorizer.fit_transform(item.text for item in train)
        self.model = RandomForestRegressor(
            n_estimators=self.trees, n_jobs=-1, random_state=42, min_samples_leaf=2
        ).fit(matrix, to_log([item.price for item in train]))
        return self

    def predict(self, item: Item) -> float:
        matrix = self.vectorizer.transform([item.text])
        return float(from_log(self.model.predict(matrix))[0])


class XGBoostPricer(Pricer):
    display_name = "xgboost"

    def __init__(self, max_features: int = 20_000, rounds: int = 600):
        self.rounds = rounds
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words="english", min_df=2
        )

    def fit(self, train, validation=None):
        from xgboost import XGBRegressor

        train = train[:MAX_TRAIN_ROWS]
        matrix = self.vectorizer.fit_transform(item.text for item in train)
        target = to_log([item.price for item in train])

        eval_set = None
        if validation:
            validation = validation[:5_000]
            eval_set = [
                (
                    self.vectorizer.transform(item.text for item in validation),
                    to_log([item.price for item in validation]),
                )
            ]

        self.model = XGBRegressor(
            n_estimators=self.rounds,
            max_depth=8,
            learning_rate=0.06,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=4,
            early_stopping_rounds=40 if eval_set else None,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(matrix, target, eval_set=eval_set, verbose=False)
        return self

    def predict(self, item: Item) -> float:
        matrix = self.vectorizer.transform([item.text])
        return float(from_log(self.model.predict(matrix))[0])
