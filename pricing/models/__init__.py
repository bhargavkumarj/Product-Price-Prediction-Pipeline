"""Model registry.

Imports are resolved on demand: torch, xgboost and litellm are heavy, and a run
that only needs the linear baselines should not pay for them or fail when one of
them is missing a system library.
"""

from importlib import import_module

REGISTRY = {
    "random": ("pricing.models.baselines", "RandomPricer"),
    "average": ("pricing.models.baselines", "AveragePricer"),
    "category-average": ("pricing.models.baselines", "CategoryAveragePricer"),
    "features": ("pricing.models.classical", "FeaturePricer"),
    "bow-linear": ("pricing.models.classical", "BagOfWordsLinearPricer"),
    "svd-ridge": ("pricing.models.classical", "WordVectorPricer"),
    "random-forest": ("pricing.models.classical", "RandomForestPricer"),
    "xgboost": ("pricing.models.classical", "XGBoostPricer"),
    "neural": ("pricing.models.neural", "NeuralPricer"),
    "frontier": ("pricing.models.frontier", "FrontierPricer"),
}


def get(name: str):
    module, attribute = REGISTRY[name]
    return getattr(import_module(module), attribute)


def create(name: str, **kwargs):
    return get(name)(**kwargs)


__all__ = ["REGISTRY", "create", "get"]
