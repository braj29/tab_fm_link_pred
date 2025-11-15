# model.py
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier


def build_tabicl():
    return TabICLClassifier(
        n_estimators=16,
        use_hierarchical=True,  # important for many tail classes
        checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
        verbose=True,
    )


def build_tabpfn(device="auto"):
    return TabPFNClassifier(
        n_estimators=16,
        device=device,
    )
