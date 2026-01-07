
# Tabular FM Link Prediction

This repo hosts a small benchmark that compares multiple tabular foundation models on the FB15k-237 knowledge graph link-prediction task: [TabICL](https://github.com/yandex-research/tabicl), [TabPFN](https://github.com/automl/TabPFN), [LimiX](https://github.com/limix-ldm/LimiX), [TabDPT](https://github.com/layer6ai-labs/TabDPT-inference), and [SAINT](https://github.com/somepago/saint). It also includes RotatE and ComplEx baselines via [PyKEEN](https://github.com/pykeen/pykeen). We turn triples (head, relation, tail) into a binary classification problem over corrupted triples and evaluate with filtered ranking metrics (MRR/MR/Hits@K). Experiments default to the full FB15k-237 splits, but you can cap them for quick local smoke tests.

For non-tabular baselines, KG-BERT is included as a binary text classifier over triples.

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

- `--model {tabicl, tabpfn, limix, tabdpt, saint, kgbert, rotatee, complex, complex-kge}` (default: `tabicl`).
- `--device {auto,cpu,cuda}` (TabPFN only; forwarded to `TabPFNClassifier`).
- `--max-train/--max-valid/--max-test` to optionally subsample each split (defaults: `None`, meaning full data).
- `--max-samples` to apply a single cap to train/valid/test for quick debugging.
- `--overfit-small` to train on 200 triples and evaluate on the same set (sanity check).
- `--n-neg-per-pos` to control the number of negatives per positive triple (default: 1).
- `--no-filter-unseen` to keep validation/test triples with unseen entities.
- `--hard-negatives` to sample relation-consistent negatives (harder).
- `--classification-metrics` to also compute accuracy/F1/ROC-AUC on the binary triple classification task.
- `--classification-threshold` to set the decision threshold for classification metrics (default: 0.5).

Under the hood `src/run.py` orchestrates the following pipeline:

1. `src/data.py.prepare_data` loads FB15k-237 via Hugging Face, resolves the triple columns (falling back to parsing tab-separated text when needed), and optionally subsamples each split. It then adds negative samples by corrupting head or tail (50/50 by default) and returns `(X_train, y_train, X_valid, y_valid, X_test, y_test)` where `X_*` contains `head`, `relation`, and `tail` columns (dtype `object`) and `y_*` is a binary label (`1` = true triple, `0` = corrupted).
2. `src/model.py` instantiates the requested model (TabICL, TabPFN, LimiX, TabDPT, SAINT, KG-BERT, RotatE, or ComplEx).
3. `src/metrics.py` computes filtered ranking metrics (MRR, MR, Hits@k) by scoring candidate tails with the binary classifier.

Example output snippet:

```
=== Loading data (small experiment) ===
=== Building TabICL ===
=== Fitting ===
=== Validation ranking metrics ===
{'MRR': 0.412, 'MR': 78.2, 'Hits@1': 0.28, 'Hits@3': 0.45, 'Hits@10': 0.63}
=== Test ranking metrics ===
{'MRR': 0.401, 'MR': 81.5, 'Hits@1': 0.27, 'Hits@3': 0.44, 'Hits@10': 0.61}
```

Results are written to `experiment_metrics.json` by default; override with `--output /path/to/file.json` to save elsewhere.

Note: The current benchmark treats link prediction as binary classification over (head, relation, tail) with negative sampling. Ranking metrics are computed by scoring candidate tails with the binary classifier and applying filtered evaluation.

### Evaluation modes

- **Ranking evaluation (default)**: MRR/MR/Hits@K for head/tail prediction (filtered).
- **Classification evaluation**: binary accuracy/F1/ROC-AUC on the generated triple classification set (`--classification-metrics`).
  Use `--classification-threshold` to tune the decision boundary (e.g., per-relation tuning if desired).

## Ranking-only evaluation

To train a binary model and run a dedicated ranking evaluation pass:

```bash
uv run python src/run_ranking_eval.py --model tabicl --max-train 1000 --max-valid 500 --max-test 500
```

## TabDPT integration (optional)

TabDPT lives at https://github.com/layer6ai-labs/TabDPT-inference. Install it and add to `PYTHONPATH`,
or pass `--tabdpt-path` to point at the repo.

Example:

```bash
uv run python main.py --model tabdpt \
  --tabdpt-path /path/to/TabDPT-inference \
  --max-train 1000 --max-valid 500 --max-test 500
```

## SAINT integration (optional)

SAINT lives at https://github.com/somepago/saint. Clone it and point `--saint-path`
to the repo root.

Example:

```bash
uv run python main.py --model saint \
  --saint-path /path/to/saint \
  --max-train 1000 --max-valid 500 --max-test 500
```

## KG-BERT integration (optional)

KG-BERT uses a BERT sequence classifier over serialized triples in the format
`[HEAD] <head> [REL] <relation> [TAIL] <tail>`. You can pass a different
Hugging Face model id via `--kgbert-model`.

Example:

```bash
uv run python main.py --model kgbert \
  --kgbert-model bert-base-uncased \
  --max-train 1000 --max-valid 500 --max-test 500
```

## RotatE (PyKEEN) baseline

RotatE runs through PyKEEN but is evaluated through the same binary pipeline by scoring triples and applying filtered ranking metrics.

Example:

```bash
uv run python main.py --model rotatee --rotatee-epochs 100 --rotatee-dim 200 --rotatee-batchsize 1024
```

## ComplEx (PyKEEN) baseline

ComplEx can run in two modes:

- `--model complex` uses the binary triple-scoring wrapper (same pipeline as the tabular models).
- `--model complex-kge` trains/evaluates ComplEx as a standard KGE model using PyKEEN's
  rank-based evaluator.

Example:

```bash
uv run python main.py --model complex --complex-epochs 100 --complex-dim 200 --complex-batchsize 1024
```

```bash
uv run python main.py --model complex-kge --complex-epochs 100 --complex-dim 200 --complex-batchsize 1024
```

## LimiX integration (optional)

To use `--model limix`, install LimiX from https://github.com/limix-ldm/LimiX and ensure
`inference.predictor.LimiXPredictor` is importable. The default checkpoint is downloaded from
`stableai-org/LimiX-16M` via Hugging Face; set `HF_TOKEN` if needed.

Example:

```bash
uv run python main.py --model limix \
  --limix-config config/cls_default_noretrieval.json \
  --max-train 1000 --max-valid 500 --max-test 500
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

### Render a job YAML (recommended)

Use the generator to avoid copying multiple YAML files:

```bash
python kube/render_job.py --model tabicl --max-train 1000 --max-valid 500 --max-test 500
kubectl apply -f kube/job_rendered.yaml
```

For other models:

```bash
python kube/render_job.py --model tabpfn --max-train 1000 --max-valid 500 --max-test 500
python kube/render_job.py --model limix --max-train 5000 --max-valid 1000 --max-test 1000
python kube/render_job.py --model tabdpt --max-train 1000 --max-valid 500 --max-test 500
```

### Batch experiment presets (kubejobs)

- Generate YAMLs for a couple of canned experiments (TabICL small and TabPFN small). This writes to `kube/experiments/` and can optionally submit them:

  ```bash
  python kube/experiments.py \
    --image <REGISTRY>/tab-fm-link-pred:latest \
    --user-email you@university.edu \
    --namespace <k8s-namespace> \
    --queue informatics-user-queue \
    --gpu-product NVIDIA-A100-SXM4-40GB \
    --gpu-limit 1 \
    --apply    # drop this flag if you only want YAMLs
  ```

  By default this emits `tabfm-tabicl-small` and `tabfm-tabpfn-small` jobs. Override the prefix with `--job-prefix` or edit `kube/experiments.py` to add more presets.

### Step-by-step tutorial (kubejobs)

- Follow along with `kube/tutorial_example.py`, mirroring the kubejobs simple tutorial:

  ```bash
  pip install kubejobs
  kubectl config current-context          # confirm cluster/namespace
  docker build -t <REGISTRY>/tab-fm-link-pred:latest .
  docker push <REGISTRY>/tab-fm-link-pred:latest

  # Dry-run: print YAML for a small TabICL job
  python kube/tutorial_example.py \
    --image <REGISTRY>/tab-fm-link-pred:latest \
    --user-email you@university.edu \
    --namespace <k8s-namespace> \
    --gpu-product NVIDIA-A100-SXM4-40GB \
    --gpu-limit 1 \
    --dry-run

  # Apply directly (streams YAML to kubectl apply -f -)
  python kube/tutorial_example.py \
    --image <REGISTRY>/tab-fm-link-pred:latest \
    --user-email you@university.edu \
    --namespace <k8s-namespace> \
    --gpu-product NVIDIA-A100-SXM4-40GB \
    --gpu-limit 1

  kubectl get jobs
  kubectl logs job/tabfm-tutorial
  ```

  Switch to TabPFN with `--model tabpfn --device cuda`. Add `--env HF_TOKEN=...` if HF auth is required. Use `--pvc-name`/`--pvc-mount` to attach a shared PVC.

### Zero-CLI preset runner

- If you prefer not to pass flags, edit the CONFIG block in `kube/tutorial_preset.py` (image, user_email, namespace, PVC, etc.), then run:

  ```bash
  python kube/tutorial_preset.py
  ```

  It writes `kube/tutorial_preset.yaml` using those baked-in values. Set `apply=True` in the CONFIG block to automatically submit with `kubectl apply -f kube/tutorial_preset.yaml`.
