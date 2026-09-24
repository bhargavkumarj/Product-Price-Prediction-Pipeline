from typing import Self

from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class Item(BaseModel):
    """One product: the text a model sees and the price it has to predict."""

    title: str
    category: str
    price: float
    full: str | None = None
    weight: float | None = None
    summary: str | None = None
    prompt: str | None = None
    id: int | None = None

    @property
    def text(self) -> str:
        return self.summary or self.full or self.title

    def make_prompt(self) -> str:
        self.prompt = f"{QUESTION}\n\n{self.text}\n\n{PREFIX}{round(self.price)}.00"
        return self.prompt

    def test_prompt(self) -> str:
        return f"{QUESTION}\n\n{self.text}\n\n{PREFIX}"

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    @classmethod
    def from_hub(cls, name: str) -> tuple[list[Self], list[Self], list[Self]]:
        data = load_dataset(name)
        return tuple(
            [cls.model_validate(row) for row in data[split]]
            for split in ("train", "validation", "test")
        )

    @staticmethod
    def push_to_hub(name: str, train, validation, test) -> None:
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in validation]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(name)
