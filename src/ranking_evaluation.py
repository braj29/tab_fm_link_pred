"""Ranking evaluation utilities for binary link prediction models."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def score_triple(triple: Tuple[str, str, str], model) -> float:
    """Return the positive-class score for a single triple."""

    df = pd.DataFrame([triple], columns=["head", "relation", "tail"])
    proba = model.predict_proba(df)
    classes = getattr(model, "classes_", np.array([0, 1]))
    pos_idx = int(list(classes).index(1))
    return float(proba[0, pos_idx])


def _batch_scores(candidates: pd.DataFrame, model) -> np.ndarray:
    proba = model.predict_proba(candidates)
    classes = getattr(model, "classes_", np.array([0, 1]))
    pos_idx = int(list(classes).index(1))
    return proba[:, pos_idx]


def evaluate_ranking(
    test_triples: Iterable[Tuple[str, str, str]],
    model,
    entity_ids: Sequence[str],
    filtered_triples: Iterable[Tuple[str, str, str]],
    mode: str = "tail",
    batch_size: int = 256,
) -> Mapping[str, float]:
    """Compute filtered ranking metrics for head or tail prediction."""

    if mode not in {"head", "tail"}:
        raise ValueError("mode must be 'head' or 'tail'")

    filtered = set(filtered_triples)
    entity_to_idx = {ent: idx for idx, ent in enumerate(entity_ids)}

    ranks = []
    hits = {1: 0, 3: 0, 10: 0}

    for head, relation, tail in test_triples:
        if mode == "tail":
            candidates = pd.DataFrame({
                "head": [head] * len(entity_ids),
                "relation": [relation] * len(entity_ids),
                "tail": list(entity_ids),
            })
            gold_entity = tail
            filter_key = (head, relation)
            filtered_set = {t for (h, r, t) in filtered if (h, r) == filter_key}
        else:
            candidates = pd.DataFrame({
                "head": list(entity_ids),
                "relation": [relation] * len(entity_ids),
                "tail": [tail] * len(entity_ids),
            })
            gold_entity = head
            filter_key = (relation, tail)
            filtered_set = {h for (h, r, t) in filtered if (r, t) == filter_key}

        scores = np.empty(len(entity_ids), dtype=np.float32)
        for start in range(0, len(entity_ids), batch_size):
            end = start + batch_size
            scores[start:end] = _batch_scores(candidates.iloc[start:end], model)

        for other in filtered_set:
            if other == gold_entity:
                continue
            idx = entity_to_idx.get(other)
            if idx is not None:
                scores[idx] = -np.inf

        gold_idx = entity_to_idx.get(gold_entity)
        if gold_idx is None:
            continue
        rank = int(np.where(np.argsort(-scores) == gold_idx)[0][0]) + 1
        ranks.append(rank)
        for k in hits:
            if rank <= k:
                hits[k] += 1

    if not ranks:
        raise ValueError("No ranks computed; check entity_ids and triples.")

    ranks_arr = np.array(ranks, dtype=float)
    metrics = {
        "MRR": float(np.mean(1.0 / ranks_arr)),
        "MR": float(np.mean(ranks_arr)),
        "Hits@1": hits[1] / len(ranks_arr),
        "Hits@3": hits[3] / len(ranks_arr),
        "Hits@10": hits[10] / len(ranks_arr),
    }
    return metrics
