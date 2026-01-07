# model.py
"""Factory helpers for tabular foundation models."""

from typing import Literal, Optional, Sequence

import os
import sys

from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier


def build_tabicl() -> TabICLClassifier:
    """Build a TabICL classifier configured for FB15k-237."""

    return TabICLClassifier(
        n_estimators=16,
        use_hierarchical=True,  # important for many tail classes
        checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
        verbose=True,
    )


def build_tabpfn(device: Literal["auto", "cpu", "cuda"] = "auto") -> TabPFNClassifier:
    """Build a TabPFN classifier with the desired compute device."""

    return TabPFNClassifier(
        n_estimators=16,
        device=device,
    )


class LimiXBinaryClassifier:
    """Lightweight wrapper to use LimiX as a binary classifier."""

    def __init__(
        self,
        device: Literal["auto", "cpu", "cuda"] = "auto",
        model_path: Optional[str] = None,
        limix_path: Optional[str] = None,
        model_id: str = "stableai-org/LimiX-16M",
        model_file: str = "LimiX-16M.ckpt",
        cache_dir: str = "./cache",
        inference_config: str = "config/cls_default_noretrieval.json",
        categorical_feature_indices: Optional[Sequence[int]] = None,
    ) -> None:
        import os

        import numpy as np
        import pandas as pd

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        import torch
        from huggingface_hub import hf_hub_download

        if limix_path:
            limix_path = os.path.abspath(limix_path)
            sys.path = [p for p in sys.path if p != limix_path]
            sys.path.insert(0, limix_path)
            # Ensure our local src/ does not shadow LimiX's model package.
            sys.path = [p for p in sys.path if not p.endswith("tab_fm_link_pred/src")]
            if "model" in sys.modules:
                sys.modules.pop("model", None)
            for key in list(sys.modules):
                if key.startswith("model."):
                    sys.modules.pop(key, None)

        try:
            from inference.predictor import LimiXPredictor
        except ImportError as exc:
            raise ImportError(
                "LimiX is not installed. Follow https://github.com/limix-ldm/LimiX to install."
            ) from exc

        if device == "auto":
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch_device = torch.device(device)

        if model_path is None:
            model_path = hf_hub_download(
                repo_id=model_id,
                filename=model_file,
                local_dir=cache_dir,
            )

        self._np = np
        self._pd = pd
        self._columns: list[str] = []
        self._categories: dict[str, list[str]] = {}
        self._x_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self.classes_ = np.array([0, 1])
        self._cat_indices = (
            list(categorical_feature_indices)
            if categorical_feature_indices is not None
            else None
        )
        if inference_config and not os.path.isabs(inference_config):
            if limix_path:
                inference_config = os.path.join(limix_path, inference_config)

        self._predictor = LimiXPredictor(
            device=torch_device,
            model_path=model_path,
            inference_config=inference_config,
            categorical_features_indices=self._cat_indices,
        )

    def _encode(self, X) -> "np.ndarray":
        pd = self._pd
        np = self._np
        if not self._columns:
            raise ValueError("LimiXBinaryClassifier must be fit before calling predict.")
        cols = []
        for col in self._columns:
            cat = pd.Categorical(X[col], categories=self._categories[col])
            cols.append(cat.codes.astype(np.int64))
        return np.stack(cols, axis=1)

    def fit(self, X, y) -> "LimiXBinaryClassifier":
        pd = self._pd
        self._columns = list(X.columns)
        self._categories = {
            col: pd.Categorical(X[col]).categories.tolist() for col in self._columns
        }
        self._x_train = self._encode(X)
        self._y_train = self._np.asarray(y, dtype=self._np.int64)
        if self._cat_indices is None:
            self._cat_indices = list(range(len(self._columns)))
        return self

    def predict_proba(self, X):
        if self._x_train is None or self._y_train is None:
            raise ValueError("LimiXBinaryClassifier must be fit before predict_proba.")
        X_enc = self._encode(X)
        return self._predictor.predict(self._x_train, self._y_train, X_enc)


class TAGBinaryClassifier:
    """TAG-style wrapper using TabICL/TabPFN on tabular triples."""

    def __init__(
        self,
        tag_path: Optional[str] = None,
        base_model: Literal["tabicl", "tabpfn"] = "tabicl",
        device: Optional[str] = None,
        max_train_rows: int = 10000,
        max_cells_per_batch: int = 5_000_000,
        seed: int = 42,
    ) -> None:
        import numpy as np
        import pandas as pd

        if tag_path:
            tag_path = os.path.abspath(tag_path)
            tag_src = os.path.join(tag_path, "src")
            if tag_src not in sys.path:
                sys.path.insert(0, tag_src)

        try:
            from tag.ensemble_config import FeatureSubset
            from tag.ensemble_models import BatchedInferenceClassifier, RandomFeatureSubsetRandomRowClassifier
        except ImportError as exc:
            raise ImportError(
                "TAG is not installed or not on PYTHONPATH. Point --tag-path to the repo."
            ) from exc

        self._np = np
        self._pd = pd
        self._FeatureSubset = FeatureSubset
        self._BatchedInferenceClassifier = BatchedInferenceClassifier
        self._RandomFeatureSubsetRandomRowClassifier = RandomFeatureSubsetRandomRowClassifier
        self._base_model = base_model
        self._device = device
        self._max_train_rows = max_train_rows
        self._max_cells_per_batch = max_cells_per_batch
        self._seed = seed
        self._columns: list[str] = []
        self._categories: dict[str, list[str]] = {}
        self._clf = None
        self.classes_ = np.array([0, 1])

    def _encode(self, X) -> "np.ndarray":
        pd = self._pd
        np = self._np
        if not self._columns:
            raise ValueError("TAGBinaryClassifier must be fit before calling predict.")
        cols = []
        for col in self._columns:
            categories = self._categories[col]
            values = X[col].where(X[col].isin(categories), "__UNK__")
            cat = pd.Categorical(values, categories=categories)
            cols.append(cat.codes.astype(np.int64))
        return np.stack(cols, axis=1)

    def _build_base_model(self):
        if self._base_model == "tabicl":
            return TabICLClassifier(
                n_estimators=16,
                use_hierarchical=True,
                checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
                verbose=True,
            )
        if self._base_model == "tabpfn":
            return TabPFNClassifier(
                n_estimators=16,
                device=self._device or "auto",
            )
        raise ValueError(f"Unsupported TAG base model: {self._base_model}")

    def fit(self, X, y) -> "TAGBinaryClassifier":
        pd = self._pd
        self._columns = list(X.columns)
        self._categories = {
            col: pd.Categorical(X[col]).categories.tolist() + ["__UNK__"]
            for col in self._columns
        }
        X_enc = self._encode(X)
        y_enc = self._np.asarray(y, dtype=self._np.int64)

        base = self._build_base_model()
        batched = self._BatchedInferenceClassifier(
            base_classifier=base,
            max_num_of_cells_per_batch=self._max_cells_per_batch,
        )

        n_features = X_enc.shape[1]
        partition_dict = {"X": self._np.arange(n_features)}
        feature_subsets = [self._FeatureSubset(features=["X"], num_columns=n_features)]
        n_rows = min(self._max_train_rows, X_enc.shape[0])
        self._clf = self._RandomFeatureSubsetRandomRowClassifier(
            base_classifier=batched,
            partition_dict=partition_dict,
            feature_subsets_list=feature_subsets,
            random_state=self._seed,
            n_sample_rows=n_rows,
        )
        self._clf.fit(X_enc, y_enc)
        return self

    def predict_proba(self, X):
        if self._clf is None:
            raise ValueError("TAGBinaryClassifier must be fit before predict_proba.")
        X_enc = self._encode(X)
        probs = self._clf.predict_proba(X_enc)
        if probs.shape[1] == 1:
            return self._np.hstack([1.0 - probs, probs])
        return probs


class TAGGraphBinaryClassifier:
    """TAG graph-feature wrapper for FB15k-237 link prediction."""

    def __init__(
        self,
        tag_path: Optional[str] = None,
        base_model: Literal["tabicl", "tabpfn"] = "tabicl",
        device: Optional[str] = None,
        node_feature_dim: int = 64,
        relation_feature_dim: int = 16,
        n_hops: int = 4,
        feature_names: Sequence[str] = ("X", "L1", "L2", "L3", "L4", "RW20", "LE20"),
        include_gpse: bool = False,
        max_train_rows: int = 10000,
        max_cells_per_batch: int = 5_000_000,
        seed: int = 42,
    ) -> None:
        import numpy as np

        if tag_path:
            tag_path = os.path.abspath(tag_path)
            tag_src = os.path.join(tag_path, "src")
            if tag_src not in sys.path:
                sys.path.insert(0, tag_src)

        try:
            from tag.ensemble_config import FeatureSubset
            from tag.ensemble_models import BatchedInferenceClassifier, RandomFeatureSubsetRandomRowClassifier
        except ImportError as exc:
            raise ImportError(
                "TAG is not installed or not on PYTHONPATH. Point --tag-path to the repo."
            ) from exc

        self._np = np
        self._FeatureSubset = FeatureSubset
        self._BatchedInferenceClassifier = BatchedInferenceClassifier
        self._RandomFeatureSubsetRandomRowClassifier = RandomFeatureSubsetRandomRowClassifier
        self._base_model = base_model
        self._device = device
        self._node_feature_dim = node_feature_dim
        self._relation_feature_dim = relation_feature_dim
        self._n_hops = n_hops
        self._feature_names = list(feature_names)
        self._include_gpse = include_gpse
        self._max_train_rows = max_train_rows
        self._max_cells_per_batch = max_cells_per_batch
        self._seed = seed
        self._clf = None
        self._entity_to_id: dict[str, int] = {}
        self._node_features: dict[str, "np.ndarray"] = {}
        self.classes_ = np.array([0, 1])

    def _build_base_model(self):
        if self._base_model == "tabicl":
            return TabICLClassifier(
                n_estimators=16,
                use_hierarchical=True,
                checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
                verbose=True,
            )
        if self._base_model == "tabpfn":
            return TabPFNClassifier(
                n_estimators=16,
                device=self._device or "auto",
            )
        raise ValueError(f"Unsupported TAG base model: {self._base_model}")

    def _build_features(self, X) -> "np.ndarray":
        from tag_graph import build_triple_features

        triples = list(zip(X["head"], X["relation"], X["tail"], strict=False))
        return build_triple_features(
            triples,
            self._node_features,
            self._entity_to_id,
            relation_dim=self._relation_feature_dim,
            seed=self._seed,
        )

    def fit(self, X, y) -> "TAGGraphBinaryClassifier":
        from tag_graph import build_tag_graph_features

        train_triples = list(zip(X["head"], X["relation"], X["tail"], strict=False))
        features = build_tag_graph_features(
            train_triples=train_triples,
            node_feature_dim=self._node_feature_dim,
            n_hops=self._n_hops,
            feature_names=self._feature_names,
            seed=self._seed,
            include_gpse=self._include_gpse,
        )
        self._entity_to_id = features.entity_to_id
        self._node_features = features.node_features

        X_enc = self._build_features(X)
        y_enc = self._np.asarray(y, dtype=self._np.int64)

        base = self._build_base_model()
        batched = self._BatchedInferenceClassifier(
            base_classifier=base,
            max_num_of_cells_per_batch=self._max_cells_per_batch,
        )

        n_features = X_enc.shape[1]
        partition_dict = {"X": self._np.arange(n_features)}
        feature_subsets = [self._FeatureSubset(features=["X"], num_columns=n_features)]
        n_rows = min(self._max_train_rows, X_enc.shape[0])
        self._clf = self._RandomFeatureSubsetRandomRowClassifier(
            base_classifier=batched,
            partition_dict=partition_dict,
            feature_subsets_list=feature_subsets,
            random_state=self._seed,
            n_sample_rows=n_rows,
        )
        self._clf.fit(X_enc, y_enc)
        return self

    def predict_proba(self, X):
        if self._clf is None:
            raise ValueError("TAGGraphBinaryClassifier must be fit before predict_proba.")
        X_enc = self._build_features(X)
        probs = self._clf.predict_proba(X_enc)
        if probs.shape[1] == 1:
            return self._np.hstack([1.0 - probs, probs])
        return probs


class TabDPTBinaryClassifier:
    """Lightweight wrapper to use TabDPT as a binary classifier."""

    def __init__(
        self,
        device: Optional[str] = None,
        model_weight_path: Optional[str] = None,
        tabdpt_path: Optional[str] = None,
        n_ensembles: int = 8,
        temperature: float = 0.8,
        context_size: int = 2048,
        permute_classes: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        import numpy as np
        import pandas as pd

        if tabdpt_path:
            if tabdpt_path not in sys.path:
                sys.path.insert(0, tabdpt_path)

        try:
            from tabdpt import TabDPTClassifier
        except ImportError as exc:
            raise ImportError(
                "TabDPT is not installed. Follow https://github.com/layer6ai-labs/TabDPT-inference."
            ) from exc

        self._np = np
        self._pd = pd
        self._columns: list[str] = []
        self._categories: dict[str, list[str]] = {}
        self._clf = TabDPTClassifier(device=device, model_weight_path=model_weight_path)
        self._n_ensembles = n_ensembles
        self._temperature = temperature
        self._context_size = context_size
        self._permute_classes = permute_classes
        self._seed = seed
        self.classes_ = np.array([0, 1])

    def _encode(self, X) -> "np.ndarray":
        pd = self._pd
        np = self._np
        if not self._columns:
            raise ValueError("TabDPTBinaryClassifier must be fit before calling predict.")
        cols = []
        for col in self._columns:
            categories = self._categories[col]
            values = X[col].where(X[col].isin(categories), "__UNK__")
            cat = pd.Categorical(values, categories=categories)
            cols.append(cat.codes.astype(np.int64))
        return np.stack(cols, axis=1)

    def fit(self, X, y) -> "TabDPTBinaryClassifier":
        pd = self._pd
        self._columns = list(X.columns)
        self._categories = {
            col: pd.Categorical(X[col]).categories.tolist() + ["__UNK__"]
            for col in self._columns
        }
        X_enc = self._encode(X)
        y_arr = self._np.asarray(y, dtype=self._np.int64)
        self._clf.fit(X_enc, y_arr)
        self.classes_ = self._np.array(sorted(set(y_arr.tolist())))
        return self

    def predict_proba(self, X):
        X_enc = self._encode(X)
        if self._n_ensembles and self._n_ensembles > 1:
            return self._clf.ensemble_predict_proba(
                X_enc,
                n_ensembles=self._n_ensembles,
                temperature=self._temperature,
                context_size=self._context_size,
                permute_classes=self._permute_classes,
                seed=self._seed,
            )
        return self._clf.predict_proba(
            X_enc,
            temperature=self._temperature,
            context_size=self._context_size,
            seed=self._seed,
        )


class SAINTBinaryClassifier:
    """Wrapper to use SAINT for binary classification."""

    def __init__(
        self,
        device: Optional[str] = None,
        saint_path: Optional[str] = None,
        embedding_size: int = 32,
        transformer_depth: int = 6,
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        attentiontype: str = "colrow",
        cont_embeddings: str = "MLP",
        lr: float = 1e-4,
        epochs: int = 20,
        batchsize: int = 256,
        seed: int = 42,
    ) -> None:
        import numpy as np
        import pandas as pd
        import torch

        if saint_path is None:
            env_path = os.environ.get("SAINT_PATH")
            if env_path:
                saint_path = env_path
            else:
                candidates = [
                    "saint",
                    "SAINT",
                    "saint-main",
                    "../saint",
                    "../SAINT",
                    "../saint-main",
                ]
                for candidate in candidates:
                    candidate_path = os.path.abspath(candidate)
                    if os.path.isfile(os.path.join(candidate_path, "models", "pretrainmodel.py")):
                        saint_path = candidate_path
                        break

        if saint_path:
            saint_path = os.path.abspath(saint_path)
            if saint_path not in sys.path:
                sys.path.insert(0, saint_path)

        try:
            from models.pretrainmodel import SAINT
            from augmentations import embed_data_mask
            from data_openml import DataSetCatCon
        except ImportError as exc:
            raise ImportError(
                "SAINT is not installed. Follow https://github.com/somepago/saint."
            ) from exc

        self._np = np
        self._pd = pd
        self._torch = torch
        self._embed_data_mask = embed_data_mask
        self._dataset_cls = DataSetCatCon
        self._SAINT = SAINT
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._embedding_size = embedding_size
        self._transformer_depth = transformer_depth
        self._attention_heads = attention_heads
        self._attention_dropout = attention_dropout
        self._ff_dropout = ff_dropout
        self._attentiontype = attentiontype
        self._cont_embeddings = cont_embeddings
        self._lr = lr
        self._epochs = epochs
        self._batchsize = batchsize
        self._seed = seed
        self._columns: list[str] = []
        self._categories: dict[str, list[str]] = {}
        self._cat_sizes: list[int] = []
        self._model: Optional[torch.nn.Module] = None
        self.classes_ = np.array([0, 1])

    def _encode(self, X) -> "np.ndarray":
        pd = self._pd
        np = self._np
        cols = []
        for col in self._columns:
            categories = self._categories[col]
            values = X[col].where(X[col].isin(categories), "__UNK__")
            cat = pd.Categorical(values, categories=categories)
            cols.append(cat.codes.astype(np.int64))
        return np.stack(cols, axis=1)

    def _make_dataset(self, X, y):
        np = self._np
        data = self._encode(X)
        mask = np.ones_like(data)
        X_dict = {"data": data, "mask": mask}
        y_arr = np.asarray(y, dtype=np.int64).reshape(-1, 1)
        y_dict = {"data": y_arr}
        cat_cols = list(range(data.shape[1]))
        return self._dataset_cls(X_dict, y_dict, cat_cols, task="clf")

    def fit(self, X, y) -> "SAINTBinaryClassifier":
        torch = self._torch
        np = self._np
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        self._columns = list(X.columns)
        self._categories = {
            col: self._pd.Categorical(X[col]).categories.tolist() + ["__UNK__"]
            for col in self._columns
        }
        self._cat_sizes = [1] + [len(self._categories[col]) for col in self._columns]

        model = self._SAINT(
            categories=self._cat_sizes,
            num_continuous=0,
            dim=self._embedding_size,
            depth=self._transformer_depth,
            heads=self._attention_heads,
            attn_dropout=self._attention_dropout,
            ff_dropout=self._ff_dropout,
            cont_embeddings=self._cont_embeddings,
            attentiontype=self._attentiontype,
            y_dim=2,
        ).to(self._device)

        dataset = self._make_dataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._batchsize,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=self._lr)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for _ in range(self._epochs):
            for x_categ, x_cont, y_gts, cat_mask, con_mask in loader:
                x_categ = x_categ.to(self._device)
                x_cont = x_cont.to(self._device)
                y_gts = y_gts.to(self._device)
                cat_mask = cat_mask.to(self._device)
                con_mask = con_mask.to(self._device)

                _, x_categ_enc, x_cont_enc = self._embed_data_mask(
                    x_categ, x_cont, cat_mask, con_mask, model
                )
                reps = model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, 0, :]
                y_outs = model.mlpfory(y_reps)
                loss = criterion(y_outs, y_gts.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._model = model
        return self

    def predict_proba(self, X):
        if self._model is None:
            raise ValueError("SAINTBinaryClassifier must be fit before predict_proba.")
        torch = self._torch
        model = self._model
        dataset = self._make_dataset(X, self._np.zeros(len(X), dtype=int))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self._batchsize, shuffle=False)
        model.eval()
        probs = []
        with torch.no_grad():
            for x_categ, x_cont, y_gts, cat_mask, con_mask in loader:
                x_categ = x_categ.to(self._device)
                x_cont = x_cont.to(self._device)
                cat_mask = cat_mask.to(self._device)
                con_mask = con_mask.to(self._device)
                _, x_categ_enc, x_cont_enc = self._embed_data_mask(
                    x_categ, x_cont, cat_mask, con_mask, model
                )
                reps = model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, 0, :]
                y_outs = model.mlpfory(y_reps)
                probs.append(torch.softmax(y_outs, dim=1).cpu().numpy())
        return self._np.concatenate(probs, axis=0)


def build_limix(
    device: Literal["auto", "cpu", "cuda"] = "auto",
    model_path: Optional[str] = None,
    limix_path: Optional[str] = None,
    model_id: str = "stableai-org/LimiX-16M",
    model_file: str = "LimiX-16M.ckpt",
    cache_dir: str = "./cache",
    inference_config: str = "config/cls_default_noretrieval.json",
) -> LimiXBinaryClassifier:
    """Build a LimiX wrapper classifier."""

    return LimiXBinaryClassifier(
        device=device,
        model_path=model_path,
        limix_path=limix_path,
        model_id=model_id,
        model_file=model_file,
        cache_dir=cache_dir,
        inference_config=inference_config,
    )


def build_tag(
    tag_path: Optional[str] = None,
    base_model: Literal["tabicl", "tabpfn"] = "tabicl",
    device: Optional[str] = None,
    max_train_rows: int = 10000,
    max_cells_per_batch: int = 5_000_000,
    seed: int = 42,
) -> TAGBinaryClassifier:
    """Build a TAG wrapper classifier."""

    return TAGBinaryClassifier(
        tag_path=tag_path,
        base_model=base_model,
        device=device,
        max_train_rows=max_train_rows,
        max_cells_per_batch=max_cells_per_batch,
        seed=seed,
    )


def build_tag_graph(
    tag_path: Optional[str] = None,
    base_model: Literal["tabicl", "tabpfn"] = "tabicl",
    device: Optional[str] = None,
    node_feature_dim: int = 64,
    relation_feature_dim: int = 16,
    n_hops: int = 4,
    feature_names: Sequence[str] = ("X", "L1", "L2", "L3", "L4", "RW20", "LE20"),
    include_gpse: bool = False,
    max_train_rows: int = 10000,
    max_cells_per_batch: int = 5_000_000,
    seed: int = 42,
) -> TAGGraphBinaryClassifier:
    """Build a TAG graph-feature wrapper classifier."""

    return TAGGraphBinaryClassifier(
        tag_path=tag_path,
        base_model=base_model,
        device=device,
        node_feature_dim=node_feature_dim,
        relation_feature_dim=relation_feature_dim,
        n_hops=n_hops,
        feature_names=feature_names,
        include_gpse=include_gpse,
        max_train_rows=max_train_rows,
        max_cells_per_batch=max_cells_per_batch,
        seed=seed,
    )


def build_tabdpt(
    device: Optional[str] = None,
    model_weight_path: Optional[str] = None,
    tabdpt_path: Optional[str] = None,
    n_ensembles: int = 8,
    temperature: float = 0.8,
    context_size: int = 2048,
    permute_classes: bool = True,
    seed: Optional[int] = 42,
) -> TabDPTBinaryClassifier:
    """Build a TabDPT wrapper classifier."""

    return TabDPTBinaryClassifier(
        device=device,
        model_weight_path=model_weight_path,
        tabdpt_path=tabdpt_path,
        n_ensembles=n_ensembles,
        temperature=temperature,
        context_size=context_size,
        permute_classes=permute_classes,
        seed=seed,
    )


class KGBERTBinaryClassifier:
    """Binary classifier wrapper using a BERT sequence classifier."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
        max_length: int = 64,
        lr: float = 2e-5,
        epochs: int = 3,
        batchsize: int = 16,
        seed: int = 42,
    ) -> None:
        import numpy as np
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._np = np
        self._torch = torch
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).to(self._device)
        special_tokens = {"additional_special_tokens": ["[HEAD]", "[REL]", "[TAIL]"]}
        self._tokenizer.add_special_tokens(special_tokens)
        self._model.resize_token_embeddings(len(self._tokenizer))
        self._max_length = max_length
        self._lr = lr
        self._epochs = epochs
        self._batchsize = batchsize
        self._seed = seed
        self.classes_ = np.array([0, 1])

    def _format_text(self, X):
        return [
            f"[HEAD] {h} [REL] {r} [TAIL] {t}"
            for h, r, t in zip(X["head"], X["relation"], X["tail"], strict=False)
        ]

    def fit(self, X, y) -> "KGBERTBinaryClassifier":
        torch = self._torch
        torch.manual_seed(self._seed)
        inputs = self._tokenizer(
            self._format_text(X),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        labels = torch.tensor(self._np.asarray(y, dtype=self._np.int64))
        dataset = torch.utils.data.TensorDataset(
            inputs["input_ids"], inputs["attention_mask"], labels
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self._batchsize, shuffle=True)
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._lr)
        self._model.train()
        for _ in range(self._epochs):
            for input_ids, attention_mask, labels in loader:
                input_ids = input_ids.to(self._device)
                attention_mask = attention_mask.to(self._device)
                labels = labels.to(self._device)
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        torch = self._torch
        self._model.eval()
        inputs = self._tokenizer(
            self._format_text(X),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        dataset = torch.utils.data.TensorDataset(
            inputs["input_ids"], inputs["attention_mask"]
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self._batchsize, shuffle=False)
        probs = []
        with torch.no_grad():
            for input_ids, attention_mask in loader:
                input_ids = input_ids.to(self._device)
                attention_mask = attention_mask.to(self._device)
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                probs.append(torch.softmax(outputs.logits, dim=1).cpu().numpy())
        return self._np.concatenate(probs, axis=0)


def build_kgbert(
    model_name: str = "bert-base-uncased",
    device: Optional[str] = None,
    max_length: int = 64,
    lr: float = 2e-5,
    epochs: int = 3,
    batchsize: int = 16,
    seed: int = 42,
) -> KGBERTBinaryClassifier:
    """Build a KG-BERT wrapper classifier."""

    return KGBERTBinaryClassifier(
        model_name=model_name,
        device=device,
        max_length=max_length,
        lr=lr,
        epochs=epochs,
        batchsize=batchsize,
        seed=seed,
    )


def build_saint(
    device: Optional[str] = None,
    saint_path: Optional[str] = None,
    embedding_size: int = 32,
    transformer_depth: int = 6,
    attention_heads: int = 8,
    attention_dropout: float = 0.1,
    ff_dropout: float = 0.1,
    attentiontype: str = "colrow",
    cont_embeddings: str = "MLP",
    lr: float = 1e-4,
    epochs: int = 20,
    batchsize: int = 256,
    seed: int = 42,
) -> SAINTBinaryClassifier:
    """Build a SAINT wrapper classifier."""

    return SAINTBinaryClassifier(
        device=device,
        saint_path=saint_path,
        embedding_size=embedding_size,
        transformer_depth=transformer_depth,
        attention_heads=attention_heads,
        attention_dropout=attention_dropout,
        ff_dropout=ff_dropout,
        attentiontype=attentiontype,
        cont_embeddings=cont_embeddings,
        lr=lr,
        epochs=epochs,
        batchsize=batchsize,
        seed=seed,
    )


class RotatEBinaryClassifier:
    """Binary classifier wrapper using RotatE scores."""

    def __init__(
        self,
        embedding_dim: int = 200,
        epochs: int = 100,
        batchsize: int = 1024,
        lr: float = 1e-3,
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        import numpy as np
        import torch

        self._np = np
        self._torch = torch
        self._embedding_dim = embedding_dim
        self._epochs = epochs
        self._batchsize = batchsize
        self._lr = lr
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._seed = seed
        self.classes_ = np.array([0, 1])
        self._model = None
        self._entity_to_id = {}
        self._relation_to_id = {}

    def fit(self, X, y) -> "RotatEBinaryClassifier":
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory

        X_pos = X[y == 1]
        triples = X_pos[["head", "relation", "tail"]].to_numpy(dtype=str)
        train_tf = TriplesFactory.from_labeled_triples(triples)

        result = pipeline(
            training=train_tf,
            testing=train_tf,
            model="RotatE",
            model_kwargs={"embedding_dim": self._embedding_dim},
            training_kwargs={"num_epochs": self._epochs, "batch_size": self._batchsize},
            optimizer_kwargs={"lr": self._lr},
            training_loop="sLCWA",
            device=str(self._device),
            random_seed=self._seed,
        )

        self._model = result.model
        self._entity_to_id = train_tf.entity_to_id
        self._relation_to_id = train_tf.relation_to_id
        return self

    def predict_proba(self, X):
        if self._model is None:
            raise ValueError("RotatEBinaryClassifier must be fit before predict_proba.")
        torch = self._torch

        heads = X["head"].map(self._entity_to_id.get).to_numpy()
        rels = X["relation"].map(self._relation_to_id.get).to_numpy()
        tails = X["tail"].map(self._entity_to_id.get).to_numpy()
        known_mask = (heads != None) & (rels != None) & (tails != None)

        scores = self._np.full(len(X), -1e9, dtype=self._np.float32)
        if known_mask.any():
            hrt = torch.tensor(
                self._np.stack(
                    [
                        heads[known_mask].astype(self._np.int64),
                        rels[known_mask].astype(self._np.int64),
                        tails[known_mask].astype(self._np.int64),
                    ],
                    axis=1,
                ),
                device=self._device,
            )
            with torch.no_grad():
                batch_scores = self._model.score_hrt(hrt).cpu().numpy()
            scores[known_mask] = batch_scores.reshape(-1)

        probs = 1.0 / (1.0 + self._np.exp(-scores))
        return self._np.stack([1.0 - probs, probs], axis=1)


def build_rotatee(
    embedding_dim: int = 200,
    epochs: int = 100,
    batchsize: int = 1024,
    lr: float = 1e-3,
    device: Optional[str] = None,
    seed: int = 42,
) -> RotatEBinaryClassifier:
    """Build a RotatE wrapper classifier."""

    return RotatEBinaryClassifier(
        embedding_dim=embedding_dim,
        epochs=epochs,
        batchsize=batchsize,
        lr=lr,
        device=device,
        seed=seed,
    )


class ComplExBinaryClassifier:
    """Binary classifier wrapper using ComplEx scores."""

    def __init__(
        self,
        embedding_dim: int = 200,
        epochs: int = 100,
        batchsize: int = 1024,
        lr: float = 1e-3,
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        import numpy as np
        import torch

        self._np = np
        self._torch = torch
        self._embedding_dim = embedding_dim
        self._epochs = epochs
        self._batchsize = batchsize
        self._lr = lr
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._seed = seed
        self.classes_ = np.array([0, 1])
        self._model = None
        self._entity_to_id = {}
        self._relation_to_id = {}

    def fit(self, X, y) -> "ComplExBinaryClassifier":
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory

        X_pos = X[y == 1]
        triples = X_pos[["head", "relation", "tail"]].to_numpy(dtype=str)
        train_tf = TriplesFactory.from_labeled_triples(triples)

        result = pipeline(
            training=train_tf,
            testing=train_tf,
            model="ComplEx",
            model_kwargs={"embedding_dim": self._embedding_dim},
            training_kwargs={"num_epochs": self._epochs, "batch_size": self._batchsize},
            optimizer_kwargs={"lr": self._lr},
            training_loop="sLCWA",
            device=str(self._device),
            random_seed=self._seed,
        )

        self._model = result.model
        self._entity_to_id = train_tf.entity_to_id
        self._relation_to_id = train_tf.relation_to_id
        return self

    def predict_proba(self, X):
        if self._model is None:
            raise ValueError("ComplExBinaryClassifier must be fit before predict_proba.")
        torch = self._torch

        heads = X["head"].map(self._entity_to_id.get).to_numpy()
        rels = X["relation"].map(self._relation_to_id.get).to_numpy()
        tails = X["tail"].map(self._entity_to_id.get).to_numpy()
        known_mask = (heads != None) & (rels != None) & (tails != None)

        scores = self._np.full(len(X), -1e9, dtype=self._np.float32)
        if known_mask.any():
            hrt = torch.tensor(
                self._np.stack(
                    [
                        heads[known_mask].astype(self._np.int64),
                        rels[known_mask].astype(self._np.int64),
                        tails[known_mask].astype(self._np.int64),
                    ],
                    axis=1,
                ),
                device=self._device,
            )
            with torch.no_grad():
                batch_scores = self._model.score_hrt(hrt).cpu().numpy()
            scores[known_mask] = batch_scores.reshape(-1)

        probs = 1.0 / (1.0 + self._np.exp(-scores))
        return self._np.stack([1.0 - probs, probs], axis=1)


def build_complex(
    embedding_dim: int = 200,
    epochs: int = 100,
    batchsize: int = 1024,
    lr: float = 1e-3,
    device: Optional[str] = None,
    seed: int = 42,
) -> ComplExBinaryClassifier:
    """Build a ComplEx wrapper classifier."""

    return ComplExBinaryClassifier(
        embedding_dim=embedding_dim,
        epochs=epochs,
        batchsize=batchsize,
        lr=lr,
        device=device,
        seed=seed,
    )
