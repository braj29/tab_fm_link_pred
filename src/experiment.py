"""Shared experiment entry point for TabICL/TabPFN runs."""

from __future__ import annotations

import argparse

from data import prepare_data
from model import build_tabicl, build_tabpfn
from metrics import classification_accuracy, link_prediction_metrics


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
    return parser.parse_args(argv)


def run_experiment(args: argparse.Namespace) -> None:
    print("=== Loading data (small experiment) ===")
    X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_data(
        max_train=args.max_train,
        max_valid=args.max_valid,
        max_test=args.max_test,
    )
    print(
        f"Splits -> train: {len(X_train)}, valid: {len(X_valid)}, test: {len(X_test)}"
    )

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

    print("=== Validation accuracy ===")
    val_acc = classification_accuracy(clf, X_valid, y_valid)
    print(f"Val Accuracy: {val_acc:.4f}")

    print("=== Test metrics ===")
    test_acc = classification_accuracy(clf, X_test, y_test)
    lp = link_prediction_metrics(clf, X_test, y_test)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(lp)
