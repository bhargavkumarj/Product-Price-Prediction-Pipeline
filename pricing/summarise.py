"""Rewrite raw listing text into a short, comparable product description.

Listings are 4,000 characters of marketing copy, spec tables and shipping notes.
Every model downstream does better on a normalised five-line summary, and it is
far cheaper to build that once here than to pay for it on every prediction.
"""

import concurrent.futures as futures

from litellm import completion
from tqdm import tqdm

from pricing.config import settings
from pricing.items import Item

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Summariser:
    def __init__(self, model: str | None = None):
        self.model = model or settings.summariser_model
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0

    def summarise(self, text: str) -> str:
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        usage = response.usage
        self.input_tokens += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens
        self.cost += response._hidden_params.get("response_cost") or 0.0
        return response.choices[0].message.content.strip()

    def apply(self, items: list[Item], workers: int = 8) -> list[Item]:
        def work(item: Item) -> Item:
            if not item.summary:
                item.summary = self.summarise(item.full or item.title)
            return item

        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(tqdm(pool.map(work, items), total=len(items), desc="summarising"))
        print(f"Summarised {len(items):,} items for ${self.cost:.2f}")
        return items
