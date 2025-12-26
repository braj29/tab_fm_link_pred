"""Shared experiment entry point for TabICL/TabPFN runs."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import pandas as pd

from data import prepare_data
from model import build_tabicl, build_tabpfn
from metrics import filtered_ranking_metrics_binary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="tabicl",
        choices=["tabicl", "tabpfn"],
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional train split size cap (None keeps full FB15k-237 train set).",
    )
    parser.add_argument(
        "--max-valid",
        type=int,
        default=None,
        help="Optional validation split size cap.",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Optional test split size cap.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap applied to train/valid/test splits (debug).",
    )
    parser.add_argument(
        "--overfit-small",
        action="store_true",
        help="Train on 200 triples and evaluate on the same set.",
    )
    parser.add_argument(
        "--n-neg-per-pos",
        type=int,
        default=1,
        help="Number of negatives to sample per positive triple.",
    )
    parser.add_argument(
        "--no-filter-unseen",
        action="store_true",
        help="Keep validation/test triples with unseen entities.",
    )
    parser.add_argument(
        "--hard-negatives",
        action="store_true",
        help="Sample relation-consistent negatives (harder).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_metrics.json",
        help="Path to write metrics JSON.",
    )
    return parser.parse_args(argv)


def run_experiment(args: argparse.Namespace) -> None:
    start = time.time()
    print("=== Starting experiment ===")
    print(f"Model: {args.model} | device={args.device}")
    print(
        "Splits cap -> train: "
        f"{args.max_train}, valid: {args.max_valid}, test: {args.max_test}"
    )

    try:
        print("=== Loading data (small experiment) ===")
        max_train = args.max_train if args.max_train is not None else args.max_samples
        max_valid = args.max_valid if args.max_valid is not None else args.max_samples
        max_test = args.max_test if args.max_test is not None else args.max_samples
        if args.max_samples is not None:
            print(
                "Using max-samples cap for splits -> "
                f"train: {max_train}, valid: {max_valid}, test: {max_test}"
            )
        X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_data(
            max_train=max_train,
            max_valid=max_valid,
            max_test=max_test,
            n_neg_per_pos=args.n_neg_per_pos,
            filter_unseen=not args.no_filter_unseen,
            hard_negatives=args.hard_negatives,
        )
        if args.overfit_small:
            X_valid, y_valid = X_train.copy(), y_train.copy()
            X_test, y_test = X_train.copy(), y_train.copy()
            print("=== Overfit sanity mode on 200 triples (train == valid == test) ===")

        if args.model == "tabicl":
            print("=== Building TabICL ===")
            clf = build_tabicl()
        elif args.model == "tabpfn":
            print("=== Building TabPFN ===")
            clf = build_tabpfn(device=args.device)
        else:
            raise ValueError(f"Unknown model type: {args.model}")

        print("=== Fitting ===")
        clf.fit(X_train, y_train)

        print("=== Task definition ===")
        print("Binary link prediction (head, relation, tail -> true/false). Classes: 2")

        train_pos = X_train[y_train == 1].copy()
        valid_pos = X_valid[y_valid == 1].copy()
        test_pos = X_test[y_test == 1].copy()
        all_pos = pd.concat([train_pos, valid_pos, test_pos], ignore_index=True)

        candidate_entities = sorted(set(train_pos["head"]).union(train_pos["tail"]))
        print(f"=== Candidate entities (train): {len(candidate_entities)} ===")

        print("=== Validation ranking metrics ===")
        val_lp = filtered_ranking_metrics_binary(
            clf,
            valid_pos,
            candidate_entities,
            all_pos,
        )
        print(val_lp)

        print("=== Test ranking metrics ===")
        test_lp = filtered_ranking_metrics_binary(
            clf,
            test_pos,
            candidate_entities,
            all_pos,
        )
        print(test_lp)

        metrics = {
            "model": args.model,
            "device": args.device,
            "max_train": max_train,
            "max_valid": max_valid,
            "max_test": max_test,
            "overfit_small": args.overfit_small,
            "val_link_prediction": val_lp,
            "test_link_prediction": test_lp,
            "elapsed_seconds": round(time.time() - start, 2),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2))
        print(f"=== Wrote metrics to {output_path} ===")
        print("=== Metrics summary ===")
        print(json.dumps(metrics, indent=2))
    except Exception:
        print("=== Experiment failed ===")
        traceback.print_exc()
        raise
