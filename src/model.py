# model.py
"""Factory helpers for tabular foundation models."""

from typing import Literal

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
