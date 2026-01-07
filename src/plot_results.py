"""Plot comparison charts from experiment metrics JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_metrics(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _average_head_tail(metrics: dict, prefix: str) -> dict | None:
    head_key = f"{prefix}_head"
    tail_key = f"{prefix}_tail"
    if head_key not in metrics or tail_key not in metrics:
        return None
    head = metrics[head_key]
    tail = metrics[tail_key]
    combined = {}
    for key in head:
        if key in tail:
            combined[key] = (head[key] + tail[key]) / 2.0
    return combined


def _extract_lp(metrics: dict, split: str) -> dict | None:
    kge_key = f"{split}_link_prediction"
    if kge_key in metrics:
        return metrics[kge_key]
    avg = _average_head_tail(metrics, f"{split}_link_prediction")
    if avg is not None:
        return avg
    return None


def _collect_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        metrics = _load_metrics(path)
        model = metrics.get("model", path.stem)
        val_lp = _extract_lp(metrics, "val")
        test_lp = _extract_lp(metrics, "test")
        rows.append({
            "model": model,
            "val": val_lp,
            "test": test_lp,
            "elapsed_seconds": metrics.get("elapsed_seconds"),
            "path": str(path),
        })
    return rows


def _plot_bar(ax, rows: list[dict], split: str, metric: str) -> None:
    labels = []
    values = []
    for row in rows:
        lp = row.get(split)
        if not lp or metric not in lp:
            continue
        labels.append(row["model"])
        values.append(lp[metric])
    ax.bar(labels, values, color="#4C78A8")
    ax.set_title(f"{split.capitalize()} {metric}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)


def _plot_runtime(ax, rows: list[dict]) -> None:
    labels = []
    values = []
    for row in rows:
        elapsed = row.get("elapsed_seconds")
        if elapsed is None:
            continue
        labels.append(row["model"])
        values.append(elapsed / 60.0)
    ax.bar(labels, values, color="#F58518")
    ax.set_title("Elapsed Minutes")
    ax.set_ylabel("Minutes")
    ax.tick_params(axis="x", rotation=30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Metrics JSON files or glob patterns.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to write plot images.",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for item in args.inputs:
        if any(ch in item for ch in "*?[]"):
            paths.extend(Path().glob(item))
        else:
            paths.append(Path(item))
    paths = [path for path in paths if path.exists()]
    if not paths:
        raise SystemExit("No metrics files found.")

    rows = _collect_rows(paths)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
    for split in ("val", "test"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for idx, metric in enumerate(metrics):
            _plot_bar(axes[idx], rows, split, metric)
        fig.tight_layout()
        out_path = output_dir / f"{split}_ranking_metrics.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_runtime(ax, rows)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_minutes.png", dpi=150)
    plt.close(fig)

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
