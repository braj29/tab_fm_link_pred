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
