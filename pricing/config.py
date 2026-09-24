import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    data: Path = ROOT / "data"
    artifacts: Path = ROOT / "artifacts"
    reports: Path = ROOT / "reports"

    dataset: str = os.getenv("PRICER_DATASET", "ed-donner/items_lite")
    source_dataset: str = "McAuley-Lab/Amazon-Reviews-2023"

    summariser_model: str = os.getenv("SUMMARISER_MODEL", "groq/openai/gpt-oss-20b")
    frontier_model: str = os.getenv("FRONTIER_MODEL", "openai/gpt-4.1-mini")

    seed: int = 42

    def __post_init__(self):
        for folder in (self.data, self.artifacts, self.reports):
            folder.mkdir(parents=True, exist_ok=True)


settings = Settings()

CATEGORIES = [
    "Automotive",
    "Electronics",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Cell_Phones_and_Accessories",
    "Toys_and_Games",
    "Appliances",
    "Musical_Instruments",
]
