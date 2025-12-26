"""Shared experiment entry point for TabICL/TabPFN runs."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from data import prepare_data
from model import build_tabicl, build_tabpfn
from metrics import classification_accuracy, link_prediction_metrics

def _filter_known_labels(
    X: "pd.DataFrame",
    y: "pd.Series",
    classes: list[str] | "np.ndarray",
    split_name: str,
) -> tuple["pd.DataFrame", "pd.Series", int]:
    known = set(classes)
    mask = y.isin(known)
    dropped = int((~mask).sum())
    if dropped:
        print(f"=== Dropping {dropped} {split_name} rows with unseen labels ===")
    return X.loc[mask].copy(), y.loc[mask].copy(), dropped


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
        X_valid_f, y_valid_f, dropped_valid = _filter_known_labels(
            X_valid, y_valid, clf.classes_, "validation"
        )
        if len(y_valid_f) == 0:
            raise ValueError("No validation labels match training classes.")
        val_acc = classification_accuracy(clf, X_valid_f, y_valid_f)
        print(f"Val Accuracy: {val_acc:.4f}")

        print("=== Test metrics ===")
        X_test_f, y_test_f, dropped_test = _filter_known_labels(
            X_test, y_test, clf.classes_, "test"
        )
        if len(y_test_f) == 0:
            raise ValueError("No test labels match training classes.")
        test_acc = classification_accuracy(clf, X_test_f, y_test_f)
        lp = link_prediction_metrics(clf, X_test_f, y_test_f)

        print(f"Test Accuracy: {test_acc:.4f}")
        print(lp)

        metrics = {
            "model": args.model,
            "device": args.device,
            "max_train": args.max_train,
            "max_valid": args.max_valid,
            "max_test": args.max_test,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "link_prediction": lp,
            "dropped_valid_unseen": dropped_valid,
            "dropped_test_unseen": dropped_test,
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
