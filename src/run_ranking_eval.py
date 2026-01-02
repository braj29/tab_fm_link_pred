"""Run ranking evaluation using binary classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data import load_splits, prepare_data
from ranking_evaluation import evaluate_ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-neg-per-pos", type=int, default=1)
    parser.add_argument("--hard-negatives", action="store_true")
    parser.add_argument("--corrupt-head-prob", type=float, default=0.5)
    parser.add_argument("--no-filter-unseen", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=str, default="ranking_metrics.json")
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    import importlib.util
    from pathlib import Path
    import sys

    src_dir = Path(__file__).resolve().parent
    model_path = src_dir / "model.py"
    spec = importlib.util.spec_from_file_location("tab_fm_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model definitions from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if args.model == "tabicl":
        return module.build_tabicl()
    if args.model == "tabpfn":
        return module.build_tabpfn(device=args.device)
    if args.model == "limix":
        return module.build_limix(device=args.device, limix_path=None)
    if args.model == "tabdpt":
        return module.build_tabdpt(device=None if args.device == "auto" else args.device)
    if args.model == "saint":
        return module.build_saint(device=None if args.device == "auto" else args.device)
    if args.model == "kgbert":
        return module.build_kgbert(device=None if args.device == "auto" else args.device)
    if args.model == "rotatee":
        return module.build_rotatee(device=None if args.device == "auto" else args.device)
    raise ValueError(f"Unknown model: {args.model}")


def main() -> None:
    args = parse_args()
    max_train = args.max_train if args.max_train is not None else args.max_samples
    max_valid = args.max_valid if args.max_valid is not None else args.max_samples
    max_test = args.max_test if args.max_test is not None else args.max_samples

    print("=== Preparing binary data ===")
    X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_data(
        max_train=max_train,
        max_valid=max_valid,
        max_test=max_test,
        n_neg_per_pos=args.n_neg_per_pos,
        filter_unseen=not args.no_filter_unseen,
        hard_negatives=args.hard_negatives,
        corrupt_head_prob=args.corrupt_head_prob,
    )

    print("=== Training model ===")
    model = load_model(args)
    model.fit(X_train, y_train)

    print("=== Loading positive triples for ranking ===")
    train_df, valid_df, test_df = load_splits(
        max_train=max_train,
        max_valid=max_valid,
        max_test=max_test,
    )
    all_pos = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    entity_ids = sorted(set(train_df["head"]).union(train_df["tail"]))

    print("=== Ranking evaluation (tail) ===")
    val_tail = evaluate_ranking(
        valid_df.itertuples(index=False, name=None),
        model,
        entity_ids,
        all_pos.itertuples(index=False, name=None),
        mode="tail",
        batch_size=args.batch_size,
    )
    test_tail = evaluate_ranking(
        test_df.itertuples(index=False, name=None),
        model,
        entity_ids,
        all_pos.itertuples(index=False, name=None),
        mode="tail",
        batch_size=args.batch_size,
    )

    print("=== Ranking evaluation (head) ===")
    val_head = evaluate_ranking(
        valid_df.itertuples(index=False, name=None),
        model,
        entity_ids,
        all_pos.itertuples(index=False, name=None),
        mode="head",
        batch_size=args.batch_size,
    )
    test_head = evaluate_ranking(
        test_df.itertuples(index=False, name=None),
        model,
        entity_ids,
        all_pos.itertuples(index=False, name=None),
        mode="head",
        batch_size=args.batch_size,
    )

    metrics = {
        "val_tail": val_tail,
        "test_tail": test_tail,
        "val_head": val_head,
        "test_head": test_head,
    }

    print(json.dumps(metrics, indent=2))
    output_path = Path(args.output)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
