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
    return None


def _parse_text_triples(text_rows):
    heads, rels, tails = [], [], []
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


def load_fb15k237():
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
