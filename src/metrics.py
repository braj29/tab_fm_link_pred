# metrics.py
import numpy as np
from sklearn.metrics import accuracy_score


def classification_accuracy(clf, X, y):
    return accuracy_score(y, clf.predict(X))


def link_prediction_metrics(clf, X, y_true, hits_ks=(1, 3, 10)):
    proba = clf.predict_proba(X)   # shape: [n, n_classes]
    sorted_idx = np.argsort(-proba, axis=1)

    ranks = []
    hits = {k: 0 for k in hits_ks}

    # Convert y_true categories → category codes
    y_codes = y_true.cat.codes.to_numpy()

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
