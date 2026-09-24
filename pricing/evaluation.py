"""One harness every model family is scored with.

Predictors are just callables that take an Item and return a price, so a linear
regression, a neural net and a zero-shot LLM are all measured the same way on
the same held-out items.
"""

import json
import math
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import r2_score
from tqdm import tqdm

from pricing.config import settings
from pricing.items import Item

GREEN, AMBER, RED, RESET = "\033[92m", "\033[93m", "\033[91m", "\033[0m"
NUMBER = re.compile(r"[-+]?\d*\.\d+|\d+")


def to_price(value) -> float:
    """Predictors may answer with a float or with text like '$249.99'."""
    if isinstance(value, (int, float)):
        return float(value)
    match = NUMBER.search(str(value).replace("$", "").replace(",", ""))
    return float(match.group()) if match else 0.0


def band(error: float, truth: float) -> str:
    if error < 40 or error / truth < 0.2:
        return "green"
    if error < 80 or error / truth < 0.4:
        return "amber"
    return "red"


@dataclass
class Prediction:
    title: str
    truth: float
    guess: float

    @property
    def error(self) -> float:
        return abs(self.guess - self.truth)

    @property
    def squared_log_error(self) -> float:
        return (math.log(max(self.guess, 0) + 1) - math.log(self.truth + 1)) ** 2

    @property
    def band(self) -> str:
        return band(self.error, self.truth)


@dataclass
class Result:
    name: str
    predictions: list[Prediction] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def average_error(self) -> float:
        return statistics.fmean(p.error for p in self.predictions)

    @property
    def median_error(self) -> float:
        return statistics.median(p.error for p in self.predictions)

    @property
    def rmsle(self) -> float:
        return math.sqrt(statistics.fmean(p.squared_log_error for p in self.predictions))

    @property
    def r2(self) -> float:
        return r2_score(
            [p.truth for p in self.predictions], [p.guess for p in self.predictions]
        )

    @property
    def hit_rate(self) -> float:
        """Share of predictions a buyer would accept: within $40 or 20%."""
        return sum(p.band == "green" for p in self.predictions) / len(self.predictions)

    def summary(self) -> dict:
        return {
            "model": self.name,
            "n": len(self.predictions),
            "avg_error": self.average_error,
            "median_error": self.median_error,
            "rmsle": self.rmsle,
            "r2": self.r2,
            "hit_rate": self.hit_rate,
            "seconds": self.seconds,
        }

    def worst(self, n: int = 5) -> list[Prediction]:
        return sorted(self.predictions, key=lambda p: p.error, reverse=True)[:n]


def evaluate(
    predictor, items: list[Item], name: str | None = None, size: int = 250, workers: int = 8
) -> Result:
    name = name or getattr(predictor, "display_name", predictor.__class__.__name__)
    sample = items[:size]
    started = datetime.now()

    def run(item: Item) -> Prediction:
        guess = to_price(predictor(item))
        return Prediction(title=item.title[:40], truth=item.price, guess=max(guess, 0.0))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        predictions = list(tqdm(pool.map(run, sample), total=len(sample), desc=name))

    result = Result(name=name, predictions=predictions)
    result.seconds = (datetime.now() - started).total_seconds()
    return result


def table(results: list[Result]) -> str:
    headers = ["model", "avg $ err", "median $", "RMSLE", "R²", "hit rate", "secs"]
    rows = [
        [
            r.name,
            f"{r.average_error:,.2f}",
            f"{r.median_error:,.2f}",
            f"{r.rmsle:.3f}",
            f"{r.r2:.3f}",
            f"{r.hit_rate:.1%}",
            f"{r.seconds:.0f}",
        ]
        for r in sorted(results, key=lambda r: r.average_error)
    ]
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]

    def line(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()

    return "\n".join([line(headers), line("-" * w for w in widths), *(line(r) for r in rows)])


def diagnostics(result: Result) -> str:
    bands = {"green": 0, "amber": 0, "red": 0}
    for prediction in result.predictions:
        bands[prediction.band] += 1
    total = len(result.predictions)
    lines = [
        f"{result.name}: "
        f"{GREEN}{bands['green']} good{RESET} / "
        f"{AMBER}{bands['amber']} ok{RESET} / "
        f"{RED}{bands['red']} poor{RESET} out of {total}",
        "worst misses:",
    ]
    for prediction in result.worst():
        lines.append(
            f"  {prediction.title:<42} truth ${prediction.truth:>7,.2f}  "
            f"guess ${prediction.guess:>7,.2f}"
        )
    return "\n".join(lines)


def save(results: list[Result], label: str = "comparison") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = settings.reports / f"{label}-{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "generated": stamp,
                "dataset": settings.dataset,
                "results": [
                    {
                        **result.summary(),
                        "predictions": [
                            {"title": p.title, "truth": p.truth, "guess": p.guess}
                            for p in result.predictions
                        ],
                    }
                    for result in results
                ],
            },
            indent=2,
        )
    )
    return path


def scatter(results: list[Result], path: Path | None = None) -> Path:
    """Predicted vs actual, one panel per model. Needs plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    colours = {"green": "#2e9e5b", "amber": "#e0a100", "red": "#d3453e"}
    figure = make_subplots(rows=1, cols=len(results), subplot_titles=[r.name for r in results])

    for column, result in enumerate(results, start=1):
        top = max(max(p.truth, p.guess) for p in result.predictions)
        figure.add_trace(
            go.Scatter(
                x=[p.truth for p in result.predictions],
                y=[p.guess for p in result.predictions],
                mode="markers",
                marker=dict(
                    size=5, color=[colours[p.band] for p in result.predictions], opacity=0.75
                ),
                text=[p.title for p in result.predictions],
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[0, top],
                y=[0, top],
                mode="lines",
                line=dict(dash="dash", width=1, color="#888"),
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(title_text="actual $", row=1, col=column)
    figure.update_yaxes(title_text="predicted $", row=1, col=1)
    figure.update_layout(height=420, width=420 * len(results), template="plotly_white")

    path = path or settings.reports / "predicted_vs_actual.html"
    figure.write_html(str(path))
    return path
