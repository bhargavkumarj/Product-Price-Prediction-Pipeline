import json
from pathlib import Path

from pricing.config import settings
from pricing.items import Item

SPLITS = ("train", "validation", "test")


def save_local(train, validation, test, folder: Path | None = None) -> Path:
    folder = folder or settings.data / "curated"
    folder.mkdir(parents=True, exist_ok=True)
    for name, items in zip(SPLITS, (train, validation, test)):
        with open(folder / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item.model_dump()) + "\n")
    return folder


def load_local(folder: Path) -> tuple[list[Item], list[Item], list[Item]]:
    splits = []
    for name in SPLITS:
        path = folder / f"{name}.jsonl"
        with open(path, encoding="utf-8") as handle:
            splits.append([Item(**json.loads(line)) for line in handle if line.strip()])
    return tuple(splits)


def load(source: str | None = None) -> tuple[list[Item], list[Item], list[Item]]:
    """Accepts a HuggingFace dataset name or a folder of curated jsonl files."""
    source = source or settings.dataset
    folder = Path(source)
    if folder.exists() and folder.is_dir():
        train, validation, test = load_local(folder)
    else:
        train, validation, test = Item.from_hub(source)
    print(
        f"Loaded {source}: {len(train):,} train / {len(validation):,} validation / "
        f"{len(test):,} test"
    )
    return train, validation, test
