# metrics.py
"""Evaluation helpers for classification and link prediction."""

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score


def classification_accuracy(
    clf: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """Return accuracy for ``clf`` on ``(X, y)``."""

    return accuracy_score(y, clf.predict(X))


def link_prediction_metrics(
    clf: ClassifierMixin,
    X: pd.DataFrame,
    y_true: pd.Series,
    hits_ks: Sequence[int] = (1, 3, 10),
) -> Mapping[str, float]:
    """Compute MRR and Hits@k link-prediction metrics.

    Args:
        clf: Fitted classifier exposing ``predict_proba`` and ``classes_``.
        X: Feature dataframe.
        y_true: Ground-truth labels for ``X``.
        hits_ks: The ``k`` cutoffs for Hits@k.

    Returns:
        Dictionary with ``MRR`` plus ``Hits@k`` entries.
    """

    proba = clf.predict_proba(X)   # shape: [n, n_classes]
    sorted_idx = np.argsort(-proba, axis=1)

    ranks = []
    hits = {k: 0 for k in hits_ks}

    # Map ground-truth labels to the model's class ordering
    classes = np.asarray(clf.classes_)
    y_codes = pd.Categorical(y_true, categories=classes).codes
    if (y_codes < 0).any():
        missing = y_true.iloc[y_codes < 0].unique()
        raise ValueError(f"Found labels not present in classifier classes: {missing}")

    for i, gold in enumerate(y_codes):
        rank = np.where(sorted_idx[i] == gold)[0][0] + 1
        ranks.append(rank)
        for k in hits_ks:
            if rank <= k:
                hits[k] += 1

    ranks = np.array(ranks, dtype=float)

    metrics = {"MRR": float(np.mean(1.0 / ranks))}
    for k in hits_ks:
        metrics[f"Hits@{k}"] = hits[k] / len(ranks)

    return metrics
