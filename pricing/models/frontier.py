"""Zero-shot pricing with a frontier LLM — no training, just the product text.

LiteLLM keeps the provider swappable, so the same class prices with GPT, Claude
or Gemini and the harness compares them like any other model.
"""

from litellm import completion

from pricing.items import Item
from pricing.models.base import Pricer

SYSTEM_PROMPT = (
    "You estimate what a product sells for online in US dollars. "
    "Reply with a number only, no currency symbol and no explanation."
)


class FrontierPricer(Pricer):
    needs_training = False

    def __init__(self, model: str | None = None, temperature: float = 0.0, **kwargs):
        from pricing.config import settings

        self.model = model or settings.frontier_model
        self.temperature = temperature
        self.kwargs = kwargs
        self.display_name = f"zero-shot {self.model.split('/')[-1]}"

    def messages(self, item: Item) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"How much does this cost, to the nearest dollar?\n\n{item.text}",
            },
        ]

    def predict(self, item: Item) -> str:
        response = completion(
            model=self.model,
            messages=self.messages(item),
            temperature=self.temperature,
            max_tokens=12,
            **self.kwargs,
        )
        # The harness pulls the number out, so a reply like "$249" is fine.
        return response.choices[0].message.content
