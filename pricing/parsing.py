import json
import re

from pricing.items import Item

MIN_CHARS = 600
MIN_PRICE = 0.5
MAX_PRICE = 999.49
MAX_CHARS_PER_FIELD = 3000
MAX_CHARS_TOTAL = 4000

# Catalogue noise: none of this tells you anything about what a product is worth.
DROP_DETAILS = [
    "Part Number",
    "Best Sellers Rank",
    "Batteries Included?",
    "Batteries Required?",
    "Item model number",
]

# Long alphanumeric tokens are model/part numbers. They blow up the vocabulary
# for the bag-of-words models and distract the LLMs, so they go.
PART_NUMBER = re.compile(r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b")

WEIGHT_IN_POUNDS = {
    "pounds": 1.0,
    "ounces": 1 / 16,
    "grams": 1 / 453.592,
    "milligrams": 1 / 453592,
    "kilograms": 1 / 0.453592,
}


def flatten(value) -> str:
    text = str(value).replace("\n", " ").replace("\r", "").replace("\t", "")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()[:MAX_CHARS_PER_FIELD]


def clean(title: str, description, features, details: dict) -> str:
    for key in DROP_DETAILS:
        details.pop(key, None)

    parts = [title]
    if description:
        parts.append(flatten(description))
    if features:
        parts.append(flatten(features))
    if details:
        parts.append(json.dumps(details))

    return PART_NUMBER.sub("", "\n".join(parts)).strip()[:MAX_CHARS_TOTAL]


def weight_in_pounds(details: dict) -> float:
    raw = details.get("Item Weight")
    if not raw:
        return 0.0
    parts = raw.split(" ")
    try:
        amount, unit = float(parts[0]), parts[1].lower()
    except (IndexError, ValueError):
        return 0.0
    if unit == "hundredths" and len(parts) > 2 and parts[2].lower() == "pounds":
        return amount / 100
    return amount * WEIGHT_IN_POUNDS.get(unit, 0.0)


def parse(row: dict, category: str) -> Item | None:
    """Turn a raw Amazon row into an Item, or None if it is not usable.

    Rows are dropped when the price is missing, outside the range we model, or
    when there is too little text for any model to work with.
    """
    try:
        price = float(row["price"])
    except (TypeError, ValueError):
        return None
    if not MIN_PRICE <= price <= MAX_PRICE:
        return None

    try:
        details = json.loads(row["details"])
    except (TypeError, ValueError, json.JSONDecodeError):
        details = {}

    full = clean(row["title"], row["description"], row["features"], details)
    if len(full) < MIN_CHARS:
        return None

    return Item(
        title=row["title"],
        category=category,
        price=price,
        full=full,
        weight=weight_in_pounds(details),
    )
