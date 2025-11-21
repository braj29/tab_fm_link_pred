import unittest
from typing import Any

import pandas as pd

from data import _parse_text_triples, _resolve_triple_columns, prepare_data, subsample


class DummySplit:
    def __init__(self, column_names):
        self.column_names = column_names


class DummyDataset(dict):
    def __getitem__(self, item: Any):
        return super().__getitem__(item)


class DataTestCase(unittest.TestCase):
    def test_resolve_triple_columns_prefers_known_candidates(self):
        ds = DummyDataset(
            train=DummySplit(["from", "rel", "to"]),
            validation=None,
            test=None,
        )
        head, rel, tail = _resolve_triple_columns(ds)
        self.assertEqual((head, rel, tail), ("from", "rel", "to"))

    def test_resolve_triple_columns_falls_back_to_first_three(self):
        ds = DummyDataset(
            train=DummySplit(["a", "b", "c", "d"]),
            validation=None,
            test=None,
        )
        head, rel, tail = _resolve_triple_columns(ds)
        self.assertEqual((head, rel, tail), ("a", "b", "c"))

    def test_parse_text_triples_handles_tabs_and_spaces(self):
        rows = ["h1\trel1\tt1", "h2 rel2 t2 extra"]
        heads, rels, tails = _parse_text_triples(rows)
        self.assertEqual(heads, ["h1", "h2"])
        self.assertEqual(rels, ["rel1", "rel2"])
        self.assertEqual(tails, ["t1", "t2 extra"])

    def test_parse_text_triples_raises_on_malformed(self):
        with self.assertRaises(ValueError):
            _parse_text_triples(["onlytwo fields"])

    def test_subsample_caps_size(self):
        df = pd.DataFrame({"head": range(10)})
        out = subsample(df, max_n=5, seed=0)
        self.assertEqual(len(out), 5)
        self.assertEqual(len(df), 10, "original should stay intact")

    def test_prepare_data_subsamples_splits(self):
        # prepare_data relies on load_fb15k237; mock via monkeypatch-like override
        original = prepare_data.__globals__["load_fb15k237"]

        def fake_loader():
            df = pd.DataFrame(
                {
                    "head": [f"h{i}" for i in range(4)],
                    "relation": [f"r{i}" for i in range(4)],
                    "tail": [f"t{i}" for i in range(4)],
                }
            )
            return df, df, df

        prepare_data.__globals__["load_fb15k237"] = fake_loader
        try:
            X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_data(
                max_train=2, max_valid=3, max_test=1
            )
        finally:
            prepare_data.__globals__["load_fb15k237"] = original

        self.assertEqual(len(X_train), 2)
        self.assertEqual(len(X_valid), 3)
        self.assertEqual(len(X_test), 1)
        self.assertListEqual(list(X_train.columns), ["head", "relation"])
        self.assertTrue(all(series.name == "tail" for series in [y_train, y_valid, y_test]))


if __name__ == "__main__":
    unittest.main()
