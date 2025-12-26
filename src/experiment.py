"""Shared experiment entry point for TabICL/TabPFN runs."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path

import pandas as pd

from data import load_splits
from model import build_tabicl, build_tabpfn
from metrics import classification_accuracy, classification_log_loss, link_prediction_metrics


def _build_filter_map(
    splits: list[pd.DataFrame],
    key_cols: tuple[str, str],
    target_col: str,
) -> dict[tuple[str, str], set[str]]:
    mapping: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split in splits:
        keys = split.loc[:, list(key_cols)].itertuples(index=False, name=None)
        for key, target in zip(keys, split[target_col], strict=False):
            mapping[key].add(target)
    return mapping


def _log_split_stats(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
    train_entities = set(train["head"]).union(train["tail"])
    valid_entities = set(valid["head"]).union(valid["tail"])
    test_entities = set(test["head"]).union(test["tail"])
    all_entities = train_entities.union(valid_entities).union(test_entities)
    all_relations = set(train["relation"]).union(valid["relation"]).union(test["relation"])

    print("=== Dataset stats ===")
    print(
        f"Entities: {len(all_entities)} | Relations: {len(all_relations)} | "
        f"Train: {len(train)} | Valid: {len(valid)} | Test: {len(test)}"
    )
    print(f"Entities in valid not in train: {len(valid_entities - train_entities)}")
    print(f"Entities in test not in train: {len(test_entities - train_entities)}")
    print(f"Tail entities in valid not in train: {len(set(valid['tail']) - set(train['tail']))}")
    print(f"Tail entities in test not in train: {len(set(test['tail']) - set(train['tail']))}")

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
        "--eval-head",
        action="store_true",
        help="Also evaluate head prediction (relation + tail -> head).",
    )
    parser.add_argument(
        "--no-filtered",
        action="store_true",
        help="Disable filtered ranking metrics (use raw ranking).",
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
        train_df, valid_df, test_df = load_splits(
            max_train=max_train,
            max_valid=max_valid,
            max_test=max_test,
        )
        if args.overfit_small:
            cap = min(200, len(train_df))
            train_df = train_df.sample(n=cap, random_state=42).copy()
            valid_df = train_df.copy()
            test_df = train_df.copy()
            print(f"=== Overfit sanity mode on {cap} triples ===")

        _log_split_stats(train_df, valid_df, test_df)

        X_train, y_train = train_df[["head", "relation"]], train_df["tail"]
        X_valid, y_valid = valid_df[["head", "relation"]], valid_df["tail"]
        X_test, y_test = test_df[["head", "relation"]], test_df["tail"]

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
        print(
            "Tail prediction (head + relation -> tail). "
            f"Classes: {len(clf.classes_)} tail entities seen in training."
        )

        print("=== Validation accuracy ===")
        X_valid_f, y_valid_f, dropped_valid = _filter_known_labels(
            X_valid, y_valid, clf.classes_, "validation"
        )
        if len(y_valid_f) == 0:
            raise ValueError("No validation labels match training classes.")
        val_acc = classification_accuracy(clf, X_valid_f, y_valid_f)
        val_loss = classification_log_loss(clf, X_valid_f, y_valid_f)
        print(f"Val Accuracy (tail classification): {val_acc:.4f}")
        print(f"Val LogLoss (tail classification): {val_loss:.4f}")

        tail_filter_map = _build_filter_map(
            [train_df, valid_df, test_df], ("head", "relation"), "tail"
        )
        print("=== Validation ranking metrics ===")
        val_lp = link_prediction_metrics(
            clf,
            X_valid_f,
            y_valid_f,
            filter_map=None if args.no_filtered else tail_filter_map,
            query_cols=("head", "relation"),
        )
        print(val_lp)

        print("=== Test metrics ===")
        X_test_f, y_test_f, dropped_test = _filter_known_labels(
            X_test, y_test, clf.classes_, "test"
        )
        if len(y_test_f) == 0:
            raise ValueError("No test labels match training classes.")
        test_acc = classification_accuracy(clf, X_test_f, y_test_f)
        test_loss = classification_log_loss(clf, X_test_f, y_test_f)
        lp = link_prediction_metrics(
            clf,
            X_test_f,
            y_test_f,
            filter_map=None if args.no_filtered else tail_filter_map,
            query_cols=("head", "relation"),
        )

        print(f"Test Accuracy (tail classification): {test_acc:.4f}")
        print(f"Test LogLoss (tail classification): {test_loss:.4f}")
        print(lp)

        if args.eval_head:
            print("=== Building head-prediction model ===")
            head_clf = build_tabicl() if args.model == "tabicl" else build_tabpfn(device=args.device)
            X_train_h, y_train_h = train_df[["relation", "tail"]], train_df["head"]
            head_clf.fit(X_train_h, y_train_h)

            head_filter_map = _build_filter_map(
                [train_df, valid_df, test_df], ("relation", "tail"), "head"
            )
            X_valid_h, y_valid_h = valid_df[["relation", "tail"]], valid_df["head"]
            X_valid_h, y_valid_h, _ = _filter_known_labels(
                X_valid_h, y_valid_h, head_clf.classes_, "validation head"
            )
            print("=== Validation head ranking metrics ===")
            val_head = link_prediction_metrics(
                head_clf,
                X_valid_h,
                y_valid_h,
                filter_map=None if args.no_filtered else head_filter_map,
                query_cols=("relation", "tail"),
            )
            print(val_head)

            X_test_h, y_test_h = test_df[["relation", "tail"]], test_df["head"]
            X_test_h, y_test_h, _ = _filter_known_labels(
                X_test_h, y_test_h, head_clf.classes_, "test head"
            )
            print("=== Test head ranking metrics ===")
            test_head = link_prediction_metrics(
                head_clf,
                X_test_h,
                y_test_h,
                filter_map=None if args.no_filtered else head_filter_map,
                query_cols=("relation", "tail"),
            )
            print(test_head)

        metrics = {
            "model": args.model,
            "device": args.device,
            "max_train": max_train,
            "max_valid": max_valid,
            "max_test": max_test,
            "overfit_small": args.overfit_small,
            "val_accuracy": val_acc,
            "val_log_loss": val_loss,
            "test_accuracy": test_acc,
            "test_log_loss": test_loss,
            "val_link_prediction": val_lp,
            "test_link_prediction": lp,
            "dropped_valid_unseen": dropped_valid,
            "dropped_test_unseen": dropped_test,
            "elapsed_seconds": round(time.time() - start, 2),
        }
        if args.eval_head:
            metrics["val_head_link_prediction"] = val_head
            metrics["test_head_link_prediction"] = test_head
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
