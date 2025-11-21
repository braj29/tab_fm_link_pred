
# Tabular FM Link Prediction

This repo hosts a tiny benchmark that compares two tabular foundation models—[TabICL](https://github.com/yandex-research/tabicl) and [TabPFN](https://github.com/automl/TabPFN)—on the FB15k-237 knowledge graph link-prediction task. We turn the triples (head, relation, tail) into a tabular classification problem where the model must predict the tail entity given the head and relation. Experiments default to the full FB15k-237 splits, but you can cap them for quick local smoke tests.

Both models operate directly on pandas DataFrames: we keep the raw string identifiers in `head`, `relation`, and `tail` columns and rely on each model's built-in preprocessing to detect categorical features. TabICL automatically ordinal-encodes string/object dtypes and assigns a dedicated category for missing values, while TabPFN accepts the same schema via pandas' internal encoding.

## Environment

- Python ≥ 3.11
- Dependencies listed in `requirements.txt` (TabICL, TabPFN, Hugging Face `datasets`, pandas, numpy, scikit-learn)

A minimal setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### uv workflow (recommended)

[`uv`](https://github.com/astral-sh/uv) is already configured via `pyproject.toml` and `uv.lock`:

```bash
uv sync        # create .venv and install dependencies
uv run main.py --model tabicl --max-train 2000 --max-valid 500 --max-test 500
```

To refresh locks after editing dependencies, run `uv lock`.

### Formatting and linting (ruff)

- Install dev extras: `uv sync --extra dev`
- Format: `uv run ruff format`
- Lint: `uv run ruff check`

The FB15k-237 dataset downloads automatically the first time you run an experiment via `datasets.load_dataset("KGraph/FB15k-237")`. Make sure you are logged into Hugging Face if your environment requires it.

## Running an experiment

Use `main.py` (remember to point `PYTHONPATH` at `src/` if you run from the repo root):

```bash
PYTHONPATH=src python main.py --model tabicl
```

For a faster smoke test you can limit each split, e.g.:

```bash
PYTHONPATH=src python main.py --model tabicl --max-train 2000 --max-valid 500 --max-test 500
```

Key arguments:

- `--model {tabicl, tabpfn}` (default: `tabicl`).
- `--device {auto,cpu,cuda}` (TabPFN only; forwarded to `TabPFNClassifier`).
- `--max-train/--max-valid/--max-test` to optionally subsample each split (defaults: `None`, meaning full data).

Under the hood `src/run.py` orchestrates the following pipeline:

1. `src/data.py.prepare_data` loads FB15k-237 via Hugging Face, resolves the triple columns (falling back to parsing tab-separated text when needed), and optionally subsamples each split. The function returns `(X_train, y_train, X_valid, y_valid, X_test, y_test)` where `X_*` contains `head` and `relation` columns (dtype `object`) and `y_*` is the tail identifier.
2. `src/model.py` instantiates either `TabICLClassifier` (with hierarchical classification enabled so it can exceed the base 10-class limit) or `TabPFNClassifier` with the requested compute device.
3. `src/metrics.py` computes both vanilla classification accuracy and link-prediction metrics (MRR plus Hits@k). Since the labels stay as strings, the helper aligns ground-truth labels with the classifier's internal `classes_` ordering before computing ranks.

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
  data.py      # dataset loading, parsing, and subsampling utilities
  model.py     # builders for TabICL and TabPFN classifiers
  metrics.py   # accuracy + link prediction metrics (MRR, Hits@k)
  run.py       # CLI experiment runner
requirements.txt
pyproject.toml
```

### Notes on label cardinality

- **TabICL** natively supports up to 10 classes. We keep `use_hierarchical=True` so it automatically trains a tree of experts whenever the tail cardinality exceeds that limit (the FB15k-237 train set has 175 unique tails).
- **TabPFN** supports up to 100 classes per model. If you want to run TabPFN on the full dataset you must either (a) limit the label space yourself (e.g., subsample or map rare tails to `Other`) or (b) integrate the [`tabpfn-extensions` many-class wrapper](https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/many_class/many_class_classifier.py).

Feel free to adapt the subsampling sizes via CLI, add more models, or integrate richer evaluation/reporting as you iterate on the experiments.

## Running on Kubernetes (kubejobs)

- Build and push the training image from the repo root (replace the registry/tag with your own):

  ```bash
  docker build -t <REGISTRY>/tab-fm-link-pred:latest .
  docker push <REGISTRY>/tab-fm-link-pred:latest
  ```

- From a machine that has access to your university cluster, install kubejobs and make sure `kubectl` points to the right context:

  ```bash
  pip install kubejobs
  kubectl config current-context  # verify cluster/namespace
  ```

- Submit a job with the helper wrapper (this writes `kube_job.yaml` and applies it):

  ```bash
  python kube/submit_job.py \
    --image <REGISTRY>/tab-fm-link-pred:latest \
    --job-name tabfm-tabicl \
    --user-email you@university.edu \
    --queue informatics-user-queue \
    --namespace <k8s-namespace> \
    --gpu-product NVIDIA-A100-SXM4-40GB \
    --gpu-limit 1 \
    --max-train 2000 --max-valid 500 --max-test 500
  ```

- Use `--model tabpfn --device cuda` to run TabPFN. Add `--env HF_TOKEN=<token>` if your cluster needs an auth token for Hugging Face downloads. Pass `--pvc-name <your-pvc> --pvc-mount /workspace/data` to reuse a shared PVC for caches or datasets.
- Include `--dry-run` to only generate the YAML; you can then inspect or submit it yourself via `kubectl apply -f kube_job.yaml`.
