from math import isclose
import unittest

import numpy as np
import pandas as pd

from metrics import classification_accuracy, link_prediction_metrics


class DummyClf:
    def __init__(self, proba: np.ndarray, classes: list[str]):
        self._proba = proba
        self.classes_ = np.array(classes)

    def predict_proba(self, X):
        return self._proba

    def predict(self, X):
        idx = np.argmax(self._proba, axis=1)
        return self.classes_[idx]


class MetricsTestCase(unittest.TestCase):
    def test_classification_accuracy_matches_predict(self):
        proba = np.array([[0.1, 0.9], [0.6, 0.4]])
        clf = DummyClf(proba, classes=["a", "b"])
        X = pd.DataFrame({"head": ["h1", "h2"], "relation": ["r1", "r2"]})
        y = pd.Series(["b", "a"])
        acc = classification_accuracy(clf, X, y)
        self.assertEqual(acc, 1.0)

    def test_link_prediction_metrics_returns_expected_ranks(self):
        proba = np.array(
            [
                [0.6, 0.3, 0.1],  # gold at idx 0 -> rank 1
                [0.2, 0.5, 0.3],  # gold at idx 2 -> rank 2
                [0.1, 0.2, 0.7],  # gold at idx 1 -> rank 3
            ]
        )
        clf = DummyClf(proba, classes=["x", "y", "z"])
        X = pd.DataFrame(
            {"head": ["h1", "h2", "h3"], "relation": ["r1", "r2", "r3"]}
        )
        y = pd.Series(["x", "z", "y"])

        metrics = link_prediction_metrics(clf, X, y, hits_ks=(1, 2, 3))

        expected_mrr = (1 / 1 + 1 / 2 + 1 / 3) / 3
        self.assertTrue(isclose(metrics["MRR"], expected_mrr))
        self.assertTrue(isclose(metrics["Hits@1"], 1 / 3))
        self.assertTrue(isclose(metrics["Hits@2"], 2 / 3))
        self.assertTrue(isclose(metrics["Hits@3"], 1.0))

    def test_link_prediction_metrics_raises_on_unknown_labels(self):
        proba = np.array([[0.5, 0.5]])
        clf = DummyClf(proba, classes=["foo", "bar"])
        X = pd.DataFrame({"head": ["h1"], "relation": ["r1"]})
        y = pd.Series(["baz"])  # not in classes

        with self.assertRaises(ValueError):
            link_prediction_metrics(clf, X, y)


if __name__ == "__main__":
    unittest.main()
