# data.py
import pandas as pd
from datasets import load_dataset

def load_fb15k237():
    ds = load_dataset("KGraph/FB15k-237")

    def to_df(split):
        return pd.DataFrame({
            "head": ds[split]["head"],
            "relation": ds[split]["relation"],
            "tail": ds[split]["tail"],
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
