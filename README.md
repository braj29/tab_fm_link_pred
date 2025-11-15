
# Tabular FM Link Prediction

This repo hosts a tiny benchmark that compares two tabular foundation models—[TabICL](https://github.com/yandex-research/tabicl) and [TabPFN](https://github.com/automl/TabPFN)—on the FB15k-237 knowledge graph link-prediction task. We turn the triples (head, relation, tail) into a tabular classification problem where the model must predict the tail entity given the head and relation. The current setup keeps everything intentionally small so it can be run on a laptop while still exercising the full training/evaluation pipeline.

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

Key arguments:

- `--model {tabicl, tabpfn}` (default: `tabicl`)
- `--device {auto,cpu,cuda}` (TabPFN only; forwarded to `TabPFNClassifier`)

The script performs the following steps:

1. Loads FB15k-237 and subsamples to a small train/valid/test split (2k/500/500) via `src/data.py`.
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

Feel free to adapt the subsampling sizes, add more models, or integrate richer evaluation/reporting as you iterate on the experiments.
