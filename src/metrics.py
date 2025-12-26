# metrics.py
"""Evaluation helpers for classification and link prediction."""

from typing import Iterable, Mapping, Sequence, Dict, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss


def classification_accuracy(
    clf: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """Return accuracy for ``clf`` on ``(X, y)``."""

    return accuracy_score(y, clf.predict(X))


def classification_log_loss(
    clf: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """Return log loss for ``clf`` on ``(X, y)``."""

    proba = clf.predict_proba(X)
    return float(log_loss(y, proba, labels=clf.classes_))


def _build_filter_map(
    triples: pd.DataFrame,
) -> Dict[Tuple[str, str], Set[str]]:
    mapping: Dict[Tuple[str, str], Set[str]] = {}
    keys = triples.loc[:, ["head", "relation"]].itertuples(index=False, name=None)
    for key, tail in zip(keys, triples["tail"], strict=False):
        mapping.setdefault(key, set()).add(tail)
    return mapping


def filtered_ranking_metrics_binary(
    clf: ClassifierMixin,
    triples: pd.DataFrame,
    candidate_entities: Sequence[str],
    positives: pd.DataFrame,
    hits_ks: Sequence[int] = (1, 3, 10),
) -> Mapping[str, float]:
    """Compute filtered MRR/MR/Hits@k for binary link prediction.

    Args:
        clf: Fitted classifier exposing ``predict_proba`` and ``classes_``.
        triples: Positive triples to evaluate (columns: head, relation, tail).
        candidate_entities: Entity list to rank as candidate tails.
        positives: All known positive triples (train/valid/test) for filtering.
        hits_ks: The ``k`` cutoffs for Hits@k.
    """

    if 1 not in clf.classes_:
        raise ValueError("Classifier does not expose label 1 for positive class.")
    pos_index = int(list(clf.classes_).index(1))
    filter_map = _build_filter_map(positives)
    entity_to_idx = {ent: idx for idx, ent in enumerate(candidate_entities)}

    ranks = []
    hits = {k: 0 for k in hits_ks}

    for head, relation, tail in triples.itertuples(index=False, name=None):
        candidates = pd.DataFrame({
            "head": [head] * len(candidate_entities),
            "relation": [relation] * len(candidate_entities),
            "tail": list(candidate_entities),
        })
        scores = clf.predict_proba(candidates)[:, pos_index]

        for other_tail in filter_map.get((head, relation), set()):
            if other_tail == tail:
                continue
            idx = entity_to_idx.get(other_tail)
            if idx is None:
                continue
            scores[idx] = -np.inf

        gold_idx = entity_to_idx.get(tail)
        if gold_idx is None:
            continue
        sorted_idx = np.argsort(-scores)
        rank = int(np.where(sorted_idx == gold_idx)[0][0]) + 1
        ranks.append(rank)
        for k in hits_ks:
            if rank <= k:
                hits[k] += 1

    if not ranks:
        raise ValueError("No valid ranks computed (check candidate entities list).")

    ranks_arr = np.array(ranks, dtype=float)
    metrics = {
        "MRR": float(np.mean(1.0 / ranks_arr)),
        "MR": float(np.mean(ranks_arr)),
    }
    for k in hits_ks:
        metrics[f"Hits@{k}"] = hits[k] / len(ranks_arr)
    return metrics


def link_prediction_metrics(
    clf: ClassifierMixin,
    X: pd.DataFrame,
    y_true: pd.Series,
    hits_ks: Sequence[int] = (1, 3, 10),
    filter_map: Mapping[tuple[str, str], set[str]] | None = None,
    query_cols: Sequence[str] = ("head", "relation"),
) -> Mapping[str, float]:
    """Compute MRR and Hits@k link-prediction metrics (optionally filtered).

    Args:
        clf: Fitted classifier exposing ``predict_proba`` and ``classes_``.
        X: Feature dataframe.
        y_true: Ground-truth labels for ``X``.
        hits_ks: The ``k`` cutoffs for Hits@k.
        filter_map: Optional mapping from query tuple to known true labels
            for filtered ranking (removed from candidates except the target).
        query_cols: Column names in ``X`` that define the query key.

    Returns:
        Dictionary with ``MRR`` plus ``Hits@k`` entries.
    """

    proba = clf.predict_proba(X)   # shape: [n, n_classes]

    ranks = []
    hits = {k: 0 for k in hits_ks}

    classes = np.asarray(clf.classes_)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_codes = pd.Categorical(y_true, categories=classes).codes
    if (y_codes < 0).any():
        missing = y_true.iloc[y_codes < 0].unique()
        raise ValueError(f"Found labels not present in classifier classes: {missing}")

    queries: Iterable[tuple[str, str]] = X.loc[:, list(query_cols)].itertuples(
        index=False,
        name=None,
    )

    for i, (gold, query) in enumerate(zip(y_codes, queries)):
        scores = proba[i].copy()
        if filter_map is not None:
            for label in filter_map.get(query, set()):
                if label == classes[gold]:
                    continue
                idx = class_to_idx.get(label)
                if idx is not None:
                    scores[idx] = -np.inf
        sorted_idx = np.argsort(-scores)
        rank = np.where(sorted_idx == gold)[0][0] + 1
        ranks.append(rank)
        for k in hits_ks:
            if rank <= k:
                hits[k] += 1

    ranks = np.array(ranks, dtype=float)

    metrics = {"MRR": float(np.mean(1.0 / ranks))}
    for k in hits_ks:
        metrics[f"Hits@{k}"] = hits[k] / len(ranks)

    return metrics
