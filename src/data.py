# data.py
import pandas as pd
from datasets import load_dataset

_TRIPLE_COLUMN_CANDIDATES = [
    ("head", "relation", "tail"),
    ("from", "rel", "to"),
    ("from", "relation", "to"),
    ("subject", "predicate", "object"),
    ("s", "p", "o"),
    ("head_id", "relation_id", "tail_id"),
]


def _resolve_triple_columns(ds):
    column_names = ds["train"].column_names
    column_lookup = {col.lower(): col for col in column_names}
    for cols in _TRIPLE_COLUMN_CANDIDATES:
        lowered = [col.lower() for col in cols]
        if all(col in column_lookup for col in lowered):
            return tuple(column_lookup[col] for col in lowered)
    if len(column_names) >= 3:
        return tuple(column_names[:3])
    raise ValueError(
        f"Could not find triple columns in FB15k-237 dataset. "
        f"Available columns: {sorted(column_lookup.values())}"
    )


def load_fb15k237():
    ds = load_dataset("KGraph/FB15k-237")
    head_col, rel_col, tail_col = _resolve_triple_columns(ds)

    def to_df(split):
        return pd.DataFrame({
            "head": ds[split][head_col],
            "relation": ds[split][rel_col],
            "tail": ds[split][tail_col],
        })

    return to_df("train"), to_df("validation"), to_df("test")


def make_categorical(df):
    """Convert columns to pandas categorical while keeping raw strings."""
    for col in df.columns:
        df[col] = df[col].astype("category")
    return df


def subsample(df, max_n=None, seed=42):
    if max_n is None or len(df) <= max_n:
        return df.copy()
    return df.sample(n=max_n, random_state=seed).copy()


def prepare_data(max_train=None, max_valid=None, max_test=None):
    """
    Load + clean the FB15k-237 dataset and optionally subsample.

    Args default to None which keeps the full split sizes.
    """
    train, valid, test = load_fb15k237()

    train = subsample(train, max_train)
    valid = subsample(valid, max_valid)
    test = subsample(test, max_test)

    # Convert to categorical with raw identifiers preserved
    train = make_categorical(train)
    valid = make_categorical(valid)
    test = make_categorical(test)

    # Features (head, relation), label (tail)
    X_train, y_train = train[["head", "relation"]], train["tail"]
    X_valid, y_valid = valid[["head", "relation"]], valid["tail"]
    X_test, y_test = test[["head", "relation"]], test["tail"]

    return X_train, y_train, X_valid, y_valid, X_test, y_test
