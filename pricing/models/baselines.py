"""The floor. Anything that cannot beat these is not learning from the text."""

import random
import statistics

from pricing.items import Item
from pricing.models.base import Pricer


class RandomPricer(Pricer):
    display_name = "random guess"
    needs_training = False

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def predict(self, item: Item) -> float:
        return self.rng.randrange(1, 1000)


class AveragePricer(Pricer):
    display_name = "training average"

    def fit(self, train, validation=None):
        self.average = statistics.fmean(item.price for item in train)
        return self

    def predict(self, item: Item) -> float:
        return self.average


class CategoryAveragePricer(Pricer):
    display_name = "category average"

    def fit(self, train, validation=None):
        prices: dict[str, list[float]] = {}
        for item in train:
            prices.setdefault(item.category, []).append(item.price)
        self.by_category = {name: statistics.fmean(values) for name, values in prices.items()}
        self.overall = statistics.fmean(item.price for item in train)
        return self

    def predict(self, item: Item) -> float:
        return self.by_category.get(item.category, self.overall)
