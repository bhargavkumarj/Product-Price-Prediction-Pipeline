import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

from datasets import load_dataset
from tqdm import tqdm

from pricing.config import settings
from pricing.items import Item
from pricing.parsing import parse

CHUNK_SIZE = 1000
WORKERS = max((os.cpu_count() or 2) - 1, 1)


class CategoryLoader:
    """Streams one Amazon category off the Hub and parses it in parallel.

    Each category is millions of rows; parsing is CPU bound, so it is farmed out
    to processes in chunks rather than done row by row.
    """

    def __init__(self, category: str):
        self.category = category
        self.dataset = None

    def _chunks(self):
        size = len(self.dataset)
        for start in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(start, min(start + CHUNK_SIZE, size)))

    def _parse_chunk(self, chunk) -> list[Item]:
        parsed = (parse(row, self.category) for row in chunk)
        return [item for item in parsed if item is not None]

    def load(self, workers: int = WORKERS) -> list[Item]:
        started = datetime.now()
        print(f"Loading {self.category}", flush=True)
        self.dataset = load_dataset(
            settings.source_dataset,
            f"raw_meta_{self.category}",
            split="full",
            trust_remote_code=True,
        )

        items: list[Item] = []
        expected = (len(self.dataset) // CHUNK_SIZE) + 1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(self._parse_chunk, self._chunks()), total=expected):
                items.extend(batch)

        minutes = (datetime.now() - started).total_seconds() / 60
        print(f"{self.category}: {len(items):,} usable items in {minutes:.1f} min", flush=True)
        return items
