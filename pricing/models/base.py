from pricing.items import Item


class Pricer:
    """Every model implements fit/predict and is callable on a single Item."""

    display_name = "pricer"
    needs_training = True

    def fit(self, train: list[Item], validation: list[Item] | None = None) -> "Pricer":
        return self

    def predict(self, item: Item) -> float:
        raise NotImplementedError

    def __call__(self, item: Item) -> float:
        return self.predict(item)
