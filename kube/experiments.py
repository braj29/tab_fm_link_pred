#!/usr/bin/env python3
"""Generate a few kubejobs experiment YAMLs for TabICL/TabPFN."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from kubejobs.jobs import GPU_PRODUCT, KueueQueue, KubernetesJob

from submit_job import _build_training_command, _parse_env


ExperimentDef = Tuple[str, Dict[str, object]]


def _default_experiments() -> List[ExperimentDef]:
    """Named experiment presets (aligns with repo quick runs)."""
    return [
        (
            "tabicl-small",
            {
                "model": "tabicl",
                "device": None,
                "max_train": 2000,
                "max_valid": 500,
                "max_test": 500,
            },
        ),
        (
            "tabpfn-small",
            {
                "model": "tabpfn",
                "device": "cuda",
                "max_train": 2000,
                "max_valid": 500,
                "max_test": 500,
            },
        ),
    ]


def _build_job(
    job_name: str,
    image: str,
    queue: str,
    namespace: str | None,
    user_email: str,
    gpu_product: GPU_PRODUCT,
    gpu_limit: int,
    cpu_request: str,
    ram_request: str,
    shm_size: str,
    volume_mounts: Dict[str, Dict[str, str]] | None,
    env_vars: Dict[str, str],
    training_args: Dict[str, object],
) -> KubernetesJob:
    cmd_ns = argparse.Namespace(**training_args)
    return KubernetesJob(
        name=job_name,
        image=image,
        kueue_queue_name=queue,
        command=["/bin/bash"],
        args=["-lc", _build_training_command(cmd_ns)],
        gpu_type="nvidia.com/gpu",
        gpu_product=gpu_product,
        gpu_limit=gpu_limit,
        cpu_request=cpu_request,
        ram_request=ram_request,
        backoff_limit=1,
        shm_size=shm_size,
        env_vars=env_vars,
        volume_mounts=volume_mounts,
        user_email=user_email,
        namespace=namespace,
    )


def _write_jobs(jobs: Iterable[Tuple[str, KubernetesJob]], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for name, job in jobs:
        path = output_dir / f"{name}.yaml"
        path.write_text(job.generate_yaml())
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate kubejobs YAMLs for a few experiment presets."
    )
    parser.add_argument("--image", required=True, help="Container image containing the repo.")
    parser.add_argument(
        "--job-prefix",
        default="tabfm",
        help="Prefix for job names (experiment name is appended).",
    )
    parser.add_argument(
        "--queue",
        default=KueueQueue.INFORMATICS,
        help="Kueue queue name.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Namespace to submit into (falls back to current context default).",
    )
    parser.add_argument(
        "--user-email",
        required=True,
        help="User email for cluster annotations/accounting.",
    )
    parser.add_argument(
        "--image-pull-secret",
        default=None,
        help="Image pull secret name if your registry is private.",
    )
    parser.add_argument("--gpu-limit", type=int, default=1)
    parser.add_argument(
        "--gpu-product",
        default=GPU_PRODUCT.NVIDIA_A100_SXM4_40GB,
        choices=[
            GPU_PRODUCT.NVIDIA_A100_SXM4_80GB,
            GPU_PRODUCT.NVIDIA_A100_SXM4_40GB,
            GPU_PRODUCT.NVIDIA_A100_SXM4_40GB_MIG_3G_20GB,
            GPU_PRODUCT.NVIDIA_A100_SXM4_40GB_MIG_1G_5GB,
            GPU_PRODUCT.NVIDIA_H100_80GB,
        ],
    )
    parser.add_argument("--cpu-request", default="12")
    parser.add_argument("--ram-request", default="64Gi")
    parser.add_argument("--shm-size", default="32Gi")

    parser.add_argument(
        "--pvc-name",
        default=None,
        help="Existing PVC to mount (optional).",
    )
    parser.add_argument(
        "--pvc-mount",
        default="/workspace/shared",
        help="Path inside the container for the PVC mount.",
    )
    parser.add_argument(
        "--env",
        nargs="*",
        default=[],
        help="Extra env vars as KEY=VALUE pairs (e.g., HF_TOKEN=...).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kube/experiments"),
        help="Directory to write experiment YAMLs.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="If set, apply each generated YAML via kubectl.",
    )

    args = parser.parse_args()

    env_vars = {"PYTHONPATH": "/workspace/tab_fm_link_pred/src"}
    try:
        env_vars.update(_parse_env(args.env))
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    volume_mounts = None
    if args.pvc_name:
        volume_mounts = {
            "shared-data": {
                "mountPath": args.pvc_mount,
                "pvc": args.pvc_name,
            }
        }

    experiments = _default_experiments()
    jobs: List[Tuple[str, KubernetesJob]] = []
    for exp_name, training_args in experiments:
        job_name = f"{args.job_prefix}-{exp_name}"
        job = _build_job(
            job_name=job_name,
            image=args.image,
            queue=args.queue,
            namespace=args.namespace,
            user_email=args.user_email,
            gpu_product=args.gpu_product,
            gpu_limit=args.gpu_limit,
            cpu_request=args.cpu_request,
            ram_request=args.ram_request,
            shm_size=args.shm_size,
            volume_mounts=volume_mounts,
            env_vars=env_vars,
            training_args=training_args,
        )
        jobs.append((exp_name, job))

    paths = _write_jobs(jobs, args.output_dir)
    for path in paths:
        print(f"Wrote {path}")

    if args.apply:
        for path in paths:
            subprocess.run(["kubectl", "apply", "-f", str(path)], check=True)
        print("Submitted jobs with kubectl apply.")


if __name__ == "__main__":
    main()
