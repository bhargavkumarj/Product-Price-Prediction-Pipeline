"""Product price prediction — curate data, then compare model families on it."""

import argparse
import statistics
from pathlib import Path

from pricing import dataset
from pricing.config import CATEGORIES, settings
from pricing.evaluation import diagnostics, evaluate, save, table
from pricing.models import REGISTRY, create


def cmd_curate(args) -> None:
    from pricing.curation import curate, describe

    train, validation, test = curate(args.categories, args.size)
    print(describe(train))
    folder = dataset.save_local(train, validation, test, args.out)
    print(f"Wrote curated splits to {folder}")
    if args.push_to:
        from pricing.items import Item

        Item.push_to_hub(args.push_to, train, validation, test)
        print(f"Pushed to {args.push_to}")


def cmd_summarise(args) -> None:
    from pricing.summarise import Summariser

    folder = args.out or settings.data / "curated"
    train, validation, test = dataset.load_local(folder)
    summariser = Summariser(args.model)
    for split in (train, validation, test):
        summariser.apply(split[: args.limit] if args.limit else split)
    dataset.save_local(train, validation, test, folder)
    print(f"Updated summaries in {folder}")


def out_folder(value: str) -> Path:
    """A bare name lands under data/; anything with a separator is used as given."""
    return Path(value) if "/" in value else settings.data / value


def build(name: str, args):
    if name == "frontier":
        return create("frontier", model=args.frontier_model)
    return create(name)


def cmd_compare(args) -> None:
    train, validation, test = dataset.load(args.dataset)
    prices = [item.price for item in test[: args.size]]
    print(
        f"Test sample: {len(prices)} items, mean ${statistics.fmean(prices):.2f}, "
        f"median ${statistics.median(prices):.2f}\n"
    )

    results = []
    for name in args.models:
        model = build(name, args)
        if model.needs_training:
            model.fit(train, validation)
        result = evaluate(model, test, name=model.display_name, size=args.size, workers=args.workers)
        results.append(result)
        print(f"\n{diagnostics(result)}\n")

    print(table(results))
    print(f"\nWrote {save(results)}")

    if args.chart:
        from pricing.evaluation import scatter

        print(f"Wrote {scatter(results)}")


def cmd_train(args) -> None:
    """Train the residual net once and keep the weights for reuse."""
    train, validation, test = dataset.load(args.dataset)
    model = create("neural", epochs=args.epochs, rows=args.rows)
    model.fit(train, validation)
    print(f"Saved weights to {model.save()}")

    result = evaluate(model, test, name=model.display_name, size=args.size)
    print(f"\n{diagnostics(result)}\n")
    print(table([result]))


def cmd_predict(args) -> None:
    from pricing.items import Item

    train, validation, _ = dataset.load(args.dataset)
    model = build(args.model, args)
    if model.needs_training:
        model.fit(train, validation)
    item = Item(title=args.text[:60], category="unknown", price=0.0, full=args.text)
    print(f"${model.predict(item):,.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    curate = sub.add_parser("curate", help="build the dataset from Amazon Reviews 2023")
    curate.add_argument("--categories", nargs="+", default=CATEGORIES)
    curate.add_argument("--size", type=int, default=100_000)
    curate.add_argument("--out", type=out_folder)
    curate.add_argument("--push-to", help="HuggingFace dataset name to push to")
    curate.set_defaults(func=cmd_curate)

    summarise = sub.add_parser("summarise", help="add LLM-written product summaries")
    summarise.add_argument("--model", default=settings.summariser_model)
    summarise.add_argument("--limit", type=int, help="only summarise the first N of each split")
    summarise.add_argument("--out", type=out_folder)
    summarise.set_defaults(func=cmd_summarise)

    compare = sub.add_parser("compare", help="train and score model families")
    compare.add_argument("--dataset", default=settings.dataset)
    compare.add_argument("--models", nargs="+", default=["average", "bow-linear", "xgboost"],
                         choices=sorted(REGISTRY))
    compare.add_argument("--size", type=int, default=250, help="held-out items to score on")
    compare.add_argument("--workers", type=int, default=8)
    compare.add_argument("--frontier-model", default=settings.frontier_model)
    compare.add_argument("--chart", action="store_true", help="write a predicted-vs-actual html chart")
    compare.set_defaults(func=cmd_compare)

    train = sub.add_parser("train", help="train the residual net and save its weights")
    train.add_argument("--dataset", default=settings.dataset)
    train.add_argument("--epochs", type=int, default=6)
    train.add_argument("--rows", type=int, default=100_000)
    train.add_argument("--size", type=int, default=250)
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="price one product description")
    predict.add_argument("text")
    predict.add_argument("--dataset", default=settings.dataset)
    predict.add_argument("--model", default="xgboost", choices=sorted(REGISTRY))
    predict.add_argument("--frontier-model", default=settings.frontier_model)
    predict.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
