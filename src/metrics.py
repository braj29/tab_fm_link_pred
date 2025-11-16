# metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def classification_accuracy(clf, X, y):
    return accuracy_score(y, clf.predict(X))


def link_prediction_metrics(clf, X, y_true, hits_ks=(1, 3, 10)):
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
