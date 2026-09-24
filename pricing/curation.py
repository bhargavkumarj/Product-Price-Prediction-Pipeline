import random

import numpy as np

from pricing.config import CATEGORIES, settings
from pricing.items import Item
from pricing.loaders import CategoryLoader

# Raw marketplace data is wildly skewed: mostly cheap items, mostly Automotive
# and Tools. Sampling proportional to price squared, with those two categories
# damped, gives a flatter target distribution and a model that is not just
# predicting "about $20" every time.
PRICE_POWER = 2
CATEGORY_WEIGHTS = {"Tools_and_Home_Improvement": 0.5, "Automotive": 0.05}


def load(categories: list[str] | None = None) -> list[Item]:
    items: list[Item] = []
    for category in categories or CATEGORIES:
        items.extend(CategoryLoader(category).load())
    print(f"Parsed {len(items):,} items across {len(categories or CATEGORIES)} categories")
    return items


def deduplicate(items: list[Item]) -> list[Item]:
    """Drop repeated listings, keeping one of each title and each body of text."""
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

    print(f"After deduplication: {len(items):,} items")
    return items


def balance(items: list[Item], size: int) -> list[Item]:
    prices = np.array([item.price for item in items], dtype=float)
    categories = np.array([item.category for item in items])

    scaled = (prices - prices.min()) / (prices.max() - prices.min() + 1e-9)
    weights = scaled**PRICE_POWER
    for category, factor in CATEGORY_WEIGHTS.items():
        weights[categories == category] *= factor
    weights /= weights.sum()

    rng = np.random.default_rng(settings.seed)
    size = min(size, len(items))
    chosen = rng.choice(len(items), size=size, replace=False, p=weights)

    sample = [items[index] for index in chosen]
    random.Random(settings.seed).shuffle(sample)
    print(f"Sampled {len(sample):,} items (mean ${np.mean([i.price for i in sample]):.2f})")
    return sample


def split(items: list[Item], validation: int = 10_000, test: int = 2_000):
    """The list is already shuffled, so slicing is a fair split."""
    validation = min(validation, len(items) // 10)
    test = min(test, len(items) // 10)
    cut = len(items) - validation - test
    train, val, held_out = items[:cut], items[cut : cut + validation], items[cut + validation :]
    for index, item in enumerate(train + val + held_out):
        item.id = index
    print(f"Split: train={len(train):,} validation={len(val):,} test={len(held_out):,}")
    return train, val, held_out


def describe(items: list[Item]) -> dict:
    prices = [item.price for item in items]
    by_category: dict[str, int] = {}
    for item in items:
        by_category[item.category] = by_category.get(item.category, 0) + 1
    return {
        "count": len(items),
        "mean_price": float(np.mean(prices)),
        "median_price": float(np.median(prices)),
        "min_price": min(prices),
        "max_price": max(prices),
        "categories": dict(sorted(by_category.items(), key=lambda kv: -kv[1])),
    }


def curate(categories: list[str] | None = None, size: int = 400_000):
    return split(balance(deduplicate(load(categories)), size))
