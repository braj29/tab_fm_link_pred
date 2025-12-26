# data.py
"""Utilities for loading and preparing the FB15k-237 triples."""

from typing import Iterable, List, Optional, Sequence, Tuple

import random

import pandas as pd
from datasets import DatasetDict, load_dataset

_TRIPLE_COLUMN_CANDIDATES: Sequence[Tuple[str, str, str]] = [
    ("head", "relation", "tail"),
    ("from", "rel", "to"),
    ("from", "relation", "to"),
    ("subject", "predicate", "object"),
    ("s", "p", "o"),
    ("head_id", "relation_id", "tail_id"),
]


def _resolve_triple_columns(ds: DatasetDict) -> Optional[Tuple[str, str, str]]:
    """Return the triple column names if they can be inferred.

    Args:
        ds: Hugging Face dataset dict containing FB15k-237 splits.

    Returns:
        Tuple of `(head, relation, tail)` column names when resolvable; otherwise
        ``None`` which signals fallback parsing logic.
    """

    column_names = ds["train"].column_names
    column_lookup = {col.lower(): col for col in column_names}
    for cols in _TRIPLE_COLUMN_CANDIDATES:
        lowered = [col.lower() for col in cols]
        if all(col in column_lookup for col in lowered):
            return tuple(column_lookup[col] for col in lowered)
    if len(column_names) >= 3:
        return tuple(column_names[:3])
    return None


def _parse_text_triples(text_rows: Iterable[str]) -> Tuple[List[str], List[str], List[str]]:
    """Parse tab- or space-delimited triples from a text column.

    Args:
        text_rows: Iterable with one string per row from the dataset.

    Returns:
        Three lists with heads, relations, and tails.

    Raises:
        ValueError: If a row cannot be split into at least three tokens or no
            triples were parsed at all.
    """

    heads: List[str] = []
    rels: List[str] = []
    tails: List[str] = []
    for row in text_rows:
        line = row.strip()
        if not line:
            continue
        if "\t" in line:
            parts = [part.strip() for part in line.split("\t") if part.strip()]
        else:
            parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Cannot parse triple from line: {row[:80]}")
        head, relation = parts[0], parts[1]
        tail = parts[2] if len(parts) == 3 else " ".join(parts[2:])
        heads.append(head)
        rels.append(relation)
        tails.append(tail)
    if not heads:
        raise ValueError("No triples parsed from text column.")
    return heads, rels, tails


def load_fb15k237() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load FB15k-237 splits from Hugging Face and normalize column names."""

    ds = load_dataset("KGraph/FB15k-237")
    head_rel_tail = _resolve_triple_columns(ds)

    def to_df(split):
        if head_rel_tail is not None:
            head_col, rel_col, tail_col = head_rel_tail
            return pd.DataFrame({
                "head": ds[split][head_col],
                "relation": ds[split][rel_col],
                "tail": ds[split][tail_col],
            })
        if "text" in ds[split].column_names:
            heads, rels, tails = _parse_text_triples(ds[split]["text"])
            return pd.DataFrame({"head": heads, "relation": rels, "tail": tails})
        raise ValueError(
            "Unsupported FB15k-237 format: expected triple columns or a 'text' column."
        )

    return to_df("train"), to_df("validation"), to_df("test")


def subsample(df: pd.DataFrame, max_n: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """Return a copy of ``df`` with at most ``max_n`` rows.

    Args:
        df: Input dataframe.
        max_n: Optional cap on the number of rows.
        seed: RNG seed for reproducible sampling when ``max_n`` is smaller
            than the dataframe length.

    Returns:
        A dataframe copy (subsampled if needed).
    """

    if max_n is None or len(df) <= max_n:
        return df.copy()
    return df.sample(n=max_n, random_state=seed).copy()


def _log_dataset_stats(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
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


def _negative_sample_split(
    df: pd.DataFrame,
    entity_pool: Sequence[str],
    positives: set[tuple[str, str, str]],
    n_neg_per_pos: int,
    seed: int,
    max_tries: int = 50,
) -> pd.DataFrame:
    rng = random.Random(seed)
    negatives: list[tuple[str, str, str]] = []
    skipped = 0

    for head, relation, tail in df.itertuples(index=False, name=None):
        for _ in range(n_neg_per_pos):
            found = False
            for _ in range(max_tries):
                corrupt_tail = rng.choice(entity_pool)
                if corrupt_tail == tail:
                    continue
                candidate = (head, relation, corrupt_tail)
                if candidate not in positives:
                    negatives.append(candidate)
                    found = True
                    break
            if not found:
                skipped += 1
    if skipped:
        print(f"=== Warning: skipped {skipped} negatives after {max_tries} tries ===")

    neg_df = pd.DataFrame(negatives, columns=["head", "relation", "tail"])
    neg_df["label"] = 0
    return neg_df


def load_splits(
    max_train: Optional[int] = None,
    max_valid: Optional[int] = None,
    max_test: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load FB15k-237 splits and optionally subsample each split."""

    train, valid, test = load_fb15k237()

    train = subsample(train, max_train)
    valid = subsample(valid, max_valid)
    test = subsample(test, max_test)

    return train, valid, test


def prepare_data(
    max_train: Optional[int] = None,
    max_valid: Optional[int] = None,
    max_test: Optional[int] = None,
    n_neg_per_pos: int = 1,
    filter_unseen: bool = True,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load, optionally subsample, and split FB15k-237 into binary features/labels."""

    train, valid, test = load_splits(max_train=max_train, max_valid=max_valid, max_test=max_test)

    _log_dataset_stats(train, valid, test)

    train_entities = sorted(set(train["head"]).union(train["tail"]))
    if filter_unseen:
        before_valid = len(valid)
        before_test = len(test)
        valid = valid[valid["head"].isin(train_entities) & valid["tail"].isin(train_entities)]
        test = test[test["head"].isin(train_entities) & test["tail"].isin(train_entities)]
        if len(valid) != before_valid:
            print(f"=== Filtered {before_valid - len(valid)} valid triples with unseen entities ===")
        if len(test) != before_test:
            print(f"=== Filtered {before_test - len(test)} test triples with unseen entities ===")

    positives = set(train.itertuples(index=False, name=None))
    positives.update(valid.itertuples(index=False, name=None))
    positives.update(test.itertuples(index=False, name=None))

    train_pos = train.copy()
    train_pos["label"] = 1
    valid_pos = valid.copy()
    valid_pos["label"] = 1
    test_pos = test.copy()
    test_pos["label"] = 1

    train_neg = _negative_sample_split(
        train,
        train_entities,
        positives,
        n_neg_per_pos=n_neg_per_pos,
        seed=seed,
    )
    valid_neg = _negative_sample_split(
        valid,
        train_entities,
        positives,
        n_neg_per_pos=n_neg_per_pos,
        seed=seed + 1,
    )
    test_neg = _negative_sample_split(
        test,
        train_entities,
        positives,
        n_neg_per_pos=n_neg_per_pos,
        seed=seed + 2,
    )

    train_labeled = pd.concat([train_pos, train_neg], ignore_index=True)
    valid_labeled = pd.concat([valid_pos, valid_neg], ignore_index=True)
    test_labeled = pd.concat([test_pos, test_neg], ignore_index=True)

    train_labeled = train_labeled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    valid_labeled = valid_labeled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_labeled = test_labeled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Keep raw string/object dtypes so TabICL/TabPFN can run their own preprocessing.
    X_train, y_train = train_labeled[["head", "relation", "tail"]], train_labeled["label"]
    X_valid, y_valid = valid_labeled[["head", "relation", "tail"]], valid_labeled["label"]
    X_test, y_test = test_labeled[["head", "relation", "tail"]], test_labeled["label"]

    return X_train, y_train, X_valid, y_valid, X_test, y_test
