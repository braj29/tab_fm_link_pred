"""TAG-style graph feature extraction for FB15k-237 link prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np


@dataclass
class TagGraphFeatures:
    node_features: Dict[str, "np.ndarray"]
    entity_to_id: Dict[str, int]


def _build_graph(entity_count: int, heads: np.ndarray, tails: np.ndarray):
    import dgl

    g = dgl.graph((heads, tails), num_nodes=entity_count)
    g = dgl.to_bidirected(g)
    g = dgl.to_simple(g)
    return g


def _compute_lp_features(g, input_feats, n_hops: int):
    import torch
    import dgl.function as fn

    preprocess_device = "cpu"
    dim = input_feats.size(1)
    lp = torch.zeros(n_hops, g.number_of_nodes(), dim).to(preprocess_device)

    g.ndata["LP"] = input_feats.to(preprocess_device)
    for hop_idx in range(n_hops):
        g.update_all(fn.copy_u("LP", "temp"), fn.mean("temp", "LP"))
        lp[hop_idx] = g.ndata["LP"].clone()
    return {f"L{l + 1}": x for l, x in enumerate(lp)}


def _compute_random_walk_features(g, input_feats, walk_length: int = 20):
    try:
        from node_classification.utils.random_walk_pe import AddRandomWalkPE
    except Exception:
        return {}

    g_pyg = _dgl_to_pyg(g, input_feats)
    g_pyg = AddRandomWalkPE(walk_length=walk_length)(g_pyg)
    return {f"RW{walk_length}": g_pyg.random_walk_pe.to("cpu")}


def _compute_laplacian_eigenvector_features(g, input_feats, k: int = 20):
    try:
        from torch_geometric.transforms import AddLaplacianEigenvectorPE
    except Exception:
        return {}

    g_pyg = _dgl_to_pyg(g, input_feats)
    g_pyg = AddLaplacianEigenvectorPE(k=k)(g_pyg)
    return {f"LE{k}": g_pyg.laplacian_eigenvector_pe.to("cpu")}


def _compute_gpse_features(g, input_feats):
    try:
        from torch_geometric.transforms import AddGPSE
        from torch_geometric.nn import GPSE
    except Exception:
        return {}

    gpse_feat_dict = {}
    for dataset in ["chembl"]:
        g_pyg = _dgl_to_pyg(g, input_feats)
        model = GPSE.from_pretrained(dataset, root="GPSE_pretrained")
        g_pyg = AddGPSE(model=model)(g_pyg)
        gpse_feat_dict[f"GPSE_{dataset}"] = g_pyg.pestat_GPSE.to("cpu")
    return gpse_feat_dict


def _dgl_to_pyg(g, input_feats):
    from torch_geometric.utils import from_dgl
    import torch

    g_pyg = from_dgl(g)
    g_pyg.x = input_feats
    g_pyg.edge_index = g_pyg.edge_index.to(torch.int64)
    return g_pyg


def build_tag_graph_features(
    train_triples: Iterable[tuple[str, str, str]],
    node_feature_dim: int = 64,
    n_hops: int = 4,
    feature_names: Sequence[str] = ("X", "L1", "L2", "L3", "L4", "RW20", "LE20"),
    seed: int = 42,
    include_gpse: bool = False,
) -> TagGraphFeatures:
    import torch

    entities = {}
    for head, _, tail in train_triples:
        if head not in entities:
            entities[head] = len(entities)
        if tail not in entities:
            entities[tail] = len(entities)

    heads = np.array([entities[h] for h, _, _ in train_triples], dtype=np.int64)
    tails = np.array([entities[t] for _, _, t in train_triples], dtype=np.int64)
    g = _build_graph(len(entities), heads, tails)

    rng = np.random.default_rng(seed)
    base_feats = torch.tensor(
        rng.standard_normal(size=(len(entities), node_feature_dim)).astype(np.float32)
    )

    feature_dict: Dict[str, "torch.Tensor"] = {"X": base_feats}
    if any(name.startswith("L") for name in feature_names):
        lp_feats = _compute_lp_features(g, base_feats, n_hops=n_hops)
        feature_dict.update(lp_feats)
    if any(name.startswith("RW") for name in feature_names):
        feature_dict.update(_compute_random_walk_features(g, base_feats))
    if any(name.startswith("LE") for name in feature_names):
        feature_dict.update(_compute_laplacian_eigenvector_features(g, base_feats))
    if include_gpse:
        feature_dict.update(_compute_gpse_features(g, base_feats))

    selected = {name: feature_dict[name] for name in feature_names if name in feature_dict}
    return TagGraphFeatures(
        node_features={k: v.numpy() for k, v in selected.items()},
        entity_to_id=entities,
    )


def build_triple_features(
    triples: Iterable[tuple[str, str, str]],
    node_features: Dict[str, np.ndarray],
    entity_to_id: Dict[str, int],
    relation_dim: int = 16,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    relations = {}
    for _, rel, _ in triples:
        if rel not in relations:
            relations[rel] = rng.standard_normal(size=(relation_dim,)).astype(np.float32)

    feature_matrix = np.concatenate(list(node_features.values()), axis=1)
    rows: List[np.ndarray] = []
    for head, rel, tail in triples:
        h_idx = entity_to_id[head]
        t_idx = entity_to_id[tail]
        rel_vec = relations[rel]
        row = np.concatenate([feature_matrix[h_idx], feature_matrix[t_idx], rel_vec], axis=0)
        rows.append(row)
    return np.stack(rows, axis=0)
