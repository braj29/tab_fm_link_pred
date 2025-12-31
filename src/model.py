# model.py
"""Factory helpers for tabular foundation models."""

from typing import Literal, Optional, Sequence

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
            if limix_path not in sys.path:
                sys.path.insert(0, limix_path)
            existing_model = sys.modules.get("model")
            if existing_model is not None:
                existing_path = getattr(existing_model, "__file__", "")
                if existing_path.endswith("src/model.py"):
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
