# metrics.py
"""Evaluation helpers for classification and link prediction."""

from typing import Iterable, Mapping, Sequence, Dict, Tuple, Set, Optional

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


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


def binary_classification_metrics(
    clf: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> Mapping[str, float]:
    """Return accuracy/F1/ROC-AUC for binary classifiers."""

    proba = clf.predict_proba(X)
    classes = getattr(clf, "classes_", np.array([0, 1]))
    pos_idx = int(list(classes).index(1))
    y_score = proba[:, pos_idx]
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_score)),
        "pr_auc": float(average_precision_score(y, y_score)),
    }


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
    predict: str = "tail",
    show_progress: bool = True,
    batch_size: Optional[int] = None,
    max_candidates: Optional[int] = None,
    seed: int = 42,
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
    if predict not in {"tail", "head"}:
        raise ValueError("predict must be 'tail' or 'head'")

    if predict == "tail":
        filter_map = _build_filter_map(positives)
    else:
        head_map: Dict[Tuple[str, str], Set[str]] = {}
        keys = positives.loc[:, ["relation", "tail"]].itertuples(index=False, name=None)
        for key, head in zip(keys, positives["head"], strict=False):
            head_map.setdefault(key, set()).add(head)
        filter_map = head_map
    use_sampling = max_candidates is not None and max_candidates > 0 and max_candidates < len(candidate_entities)
    rng = np.random.default_rng(seed) if use_sampling else None
    if not use_sampling:
        entity_to_idx = {ent: idx for idx, ent in enumerate(candidate_entities)}

    ranks = []
    hits = {k: 0 for k in hits_ks}

    iterator = triples.itertuples(index=False, name=None)
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(triples), desc=f"ranking/{predict}")

    for head, relation, tail in iterator:
        if predict == "tail":
            filter_key = (head, relation)
            gold_entity = tail
        else:
            filter_key = (relation, tail)
            gold_entity = head

        if use_sampling:
            if gold_entity not in candidate_entities:
                continue
            pool = [ent for ent in candidate_entities if ent != gold_entity]
            sample_size = max_candidates - 1
            if sample_size <= 0:
                continue
            if sample_size >= len(pool):
                sampled = pool
            else:
                sampled = rng.choice(pool, size=sample_size, replace=False).tolist()
            cand_entities = [gold_entity] + sampled
            entity_to_idx = {ent: idx for idx, ent in enumerate(cand_entities)}
        else:
            cand_entities = list(candidate_entities)

        if predict == "tail":
            candidates = pd.DataFrame({
                "head": [head] * len(cand_entities),
                "relation": [relation] * len(cand_entities),
                "tail": cand_entities,
            })
        else:
            candidates = pd.DataFrame({
                "head": cand_entities,
                "relation": [relation] * len(cand_entities),
                "tail": [tail] * len(cand_entities),
            })

        if batch_size is None or batch_size <= 0 or batch_size >= len(cand_entities):
            scores = clf.predict_proba(candidates)[:, pos_index]
        else:
            scores = np.empty(len(cand_entities), dtype=float)
            for start in range(0, len(cand_entities), batch_size):
                end = min(start + batch_size, len(cand_entities))
                chunk_entities = cand_entities[start:end]
                if predict == "tail":
                    chunk_df = pd.DataFrame({
                        "head": [head] * len(chunk_entities),
                        "relation": [relation] * len(chunk_entities),
                        "tail": list(chunk_entities),
                    })
                else:
                    chunk_df = pd.DataFrame({
                        "head": list(chunk_entities),
                        "relation": [relation] * len(chunk_entities),
                        "tail": [tail] * len(chunk_entities),
                    })
                scores[start:end] = clf.predict_proba(chunk_df)[:, pos_index]

        for other_entity in filter_map.get(filter_key, set()):
            if other_entity == gold_entity:
                continue
            idx = entity_to_idx.get(other_entity)
            if idx is None:
                continue
            scores[idx] = -np.inf

        gold_idx = entity_to_idx.get(gold_entity)
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


def sampled_ranking_metrics_binary(
    clf: ClassifierMixin,
    triples: pd.DataFrame,
    candidate_entities: Sequence[str],
    positives: pd.DataFrame,
    n_neg_per_pos: int,
    hits_ks: Sequence[int] = (1, 3, 10),
    predict: str = "tail",
    seed: int = 42,
    max_tries: int = 50,
    show_progress: bool = True,
) -> Mapping[str, float]:
    """Compute MRR/Hits@k by ranking each positive among sampled negatives."""

    if 1 not in clf.classes_:
        raise ValueError("Classifier does not expose label 1 for positive class.")
    pos_index = int(list(clf.classes_).index(1))
    if predict not in {"tail", "head"}:
        raise ValueError("predict must be 'tail' or 'head'")

    rng = np.random.default_rng(seed)
    positives_set = set(positives.itertuples(index=False, name=None))
    entity_pool = list(candidate_entities)

    ranks = []
    hits = {k: 0 for k in hits_ks}

    iterator = triples.itertuples(index=False, name=None)
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(triples), desc=f"ranking-sampled/{predict}")

    for head, relation, tail in iterator:
        negatives: list[tuple[str, str, str]] = []
        for _ in range(n_neg_per_pos):
            found = False
            for _ in range(max_tries):
                corrupt_entity = rng.choice(entity_pool)
                if predict == "tail":
                    if corrupt_entity == tail:
                        continue
                    candidate = (head, relation, corrupt_entity)
                else:
                    if corrupt_entity == head:
                        continue
                    candidate = (corrupt_entity, relation, tail)
                if candidate not in positives_set:
                    negatives.append(candidate)
                    found = True
                    break
            if not found:
                break

        candidates = [(head, relation, tail)] + negatives
        cand_df = pd.DataFrame(candidates, columns=["head", "relation", "tail"])
        scores = clf.predict_proba(cand_df)[:, pos_index]
        gold_score = scores[0]
        rank = int((scores > gold_score).sum()) + 1
        ranks.append(rank)
        for k in hits_ks:
            if rank <= k:
                hits[k] += 1

    if not ranks:
        raise ValueError("No valid sampled ranks computed.")

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
