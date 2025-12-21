#!/usr/bin/env python3
"""Tutorial runner with baked-in parameters (no CLI flags needed).

Edit the CONFIG block once, then run:

    python kube/tutorial_preset.py

It will write a single YAML (default: kube/tutorial_preset.yaml) and
optionally apply it to the current Kubernetes context if APPLY=True.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import subprocess

from kubejobs.jobs import GPU_PRODUCT, KueueQueue, KubernetesJob

from submit_job import _build_training_command


# ---- User-editable config -------------------------------------------------

CONFIG: Dict[str, object] = {
    # Container image with this repo inside. REQUIRED.
    "image": "<REGISTRY>/tab-fm-link-pred:latest",
    # Used for cluster annotations/accounting. REQUIRED.
    "user_email": "you@university.edu",
    # Namespace to submit into (None uses the context default).
    "namespace": None,
    # Kueue queue name.
    "queue": KueueQueue.INFORMATICS,
    # Kubernetes job name.
    "job_name": "tabfm-tutorial-preset",
    # GPU settings.
    "gpu_product": GPU_PRODUCT.NVIDIA_A100_SXM4_40GB,
    "gpu_limit": 1,
    # CPU/RAM/shm requests.
    "cpu_request": "12",
    "ram_request": "64Gi",
    "shm_size": "32Gi",
    # Optional PVC mount.
    "pvc_name": None,
    "pvc_mount": "/workspace/shared",
    # Environment variables inside the job container.
    "env": {
        "PYTHONPATH": "/workspace/tab_fm_link_pred/src",
        # "HF_TOKEN": "xxx",  # uncomment if needed for dataset auth
    },
    # Training arguments.
    "model": "tabicl",  # or "tabpfn"
    "device": "cuda",
    "max_train": 2000,
    "max_valid": 500,
    "max_test": 500,
    # Output YAML location.
    "output_path": Path("kube/tutorial_preset.yaml"),
    # Set to True to run `kubectl apply -f <output_path>` automatically.
    "apply": False,
}


# ---- Script logic ---------------------------------------------------------

REQUIRED_FIELDS = ("image", "user_email")


def _assert_required(cfg: Dict[str, object]) -> None:
    missing = [k for k in REQUIRED_FIELDS if not cfg.get(k) or "<REGISTRY>" in str(cfg[k])]
    if missing:
        sys.exit(
            f"Please set required config values (current missing/placeholder: {missing}). "
            "Edit CONFIG at the top of kube/tutorial_preset.py."
        )


def _build_job(cfg: Dict[str, object]) -> KubernetesJob:
    training_ns = argparse.Namespace(
        model=cfg["model"],
        device=cfg["device"],
        max_train=cfg["max_train"],
        max_valid=cfg["max_valid"],
        max_test=cfg["max_test"],
    )

    volume_mounts: Optional[Dict[str, Dict[str, str]]] = None
    if cfg["pvc_name"]:
        volume_mounts = {
            "shared-data": {
                "mountPath": cfg["pvc_mount"],
                "pvc": cfg["pvc_name"],
            }
        }

    return KubernetesJob(
        name=str(cfg["job_name"]),
        image=str(cfg["image"]),
        kueue_queue_name=str(cfg["queue"]),
        command=["/bin/bash"],
        args=["-lc", _build_training_command(training_ns)],
        gpu_type="nvidia.com/gpu",
        gpu_product=cfg["gpu_product"],
        gpu_limit=int(cfg["gpu_limit"]),
        cpu_request=str(cfg["cpu_request"]),
        ram_request=str(cfg["ram_request"]),
        backoff_limit=1,
        shm_size=str(cfg["shm_size"]),
        env_vars={k: str(v) for k, v in cfg["env"].items()},
        volume_mounts=volume_mounts,
        user_email=str(cfg["user_email"]),
        namespace=cfg["namespace"],
    )


def main() -> None:
    cfg = CONFIG.copy()
    _assert_required(cfg)

    output_path = Path(cfg["output_path"])
    job = _build_job(cfg)

    yaml_text = job.generate_yaml()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_text)
    print(f"Wrote {output_path}")

    if cfg.get("apply"):
        subprocess.run(["kubectl", "apply", "-f", str(output_path)], check=True)
        print(f"Applied job {cfg['job_name']} to the current context.")
    else:
        print("Dry-run mode (apply=False). To submit automatically, set CONFIG['apply']=True.")


if __name__ == "__main__":
    main()
