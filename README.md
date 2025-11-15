
# Tabular FM Link Prediction

This repo hosts a tiny benchmark that compares two tabular foundation models—[TabICL](https://github.com/yandex-research/tabicl) and [TabPFN](https://github.com/automl/TabPFN)—on the FB15k-237 knowledge graph link-prediction task. We turn the triples (head, relation, tail) into a tabular classification problem where the model must predict the tail entity given the head and relation. Experiments default to the full FB15k-237 splits, but you can cap them for quick local smoke tests.

## Environment

- Python ≥ 3.11
- Dependencies listed in `requirements.txt` (TabICL, TabPFN, Hugging Face `datasets`, pandas, numpy, scikit-learn)

A minimal setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The FB15k-237 dataset downloads automatically the first time you run an experiment via `datasets.load_dataset("KGraph/FB15k-237")`. Make sure you are logged into Hugging Face if your environment requires it.

## Running an experiment

Use the `src/run.py` entry point (remember to point `PYTHONPATH` at `src/` if you run from the repo root):

```bash
PYTHONPATH=src python src/run.py --model tabicl
```

For a faster smoke test you can limit each split, e.g.:

```bash
PYTHONPATH=src python src/run.py --model tabicl --max-train 2000 --max-valid 500 --max-test 500
```

Key arguments:

- `--model {tabicl, tabpfn}` (default: `tabicl`)
- `--device {auto,cpu,cuda}` (TabPFN only; forwarded to `TabPFNClassifier`)
- `--max-train/--max-valid/--max-test` to optionally subsample each split (defaults: `None`, meaning full data)

The script performs the following steps:

1. Loads FB15k-237 via `src/data.py` (subsampling only if you pass `--max-*` flags).
2. Builds the requested classifier (`TabICLClassifier` or `TabPFNClassifier`) from `src/model.py`.
3. Trains on the tabular data and prints validation accuracy plus test accuracy and link-prediction metrics (MRR, Hits@k) computed in `src/metrics.py`.

Example output snippet:

```
=== Loading data (small experiment) ===
=== Building TabICL ===
=== Fitting ===
=== Validation accuracy ===
Val Accuracy: 0.4270
=== Test metrics ===
Test Accuracy: 0.4120
{'MRR': 0.512, 'Hits@1': 0.32, 'Hits@3': 0.58, 'Hits@10': 0.74}
```

## Project structure

```
src/
  data.py      # dataset loading, categorical conversion, subsampling utilities
  model.py     # builders for TabICL and TabPFN classifiers
  metrics.py   # accuracy + link prediction metrics (MRR, Hits@k)
  run.py       # CLI experiment runner
requirements.txt
pyproject.toml
```

Feel free to adapt the subsampling sizes via CLI, add more models, or integrate richer evaluation/reporting as you iterate on the experiments.
