#!/usr/bin/env python3
"""Minimal kubejobs tutorial for this repo.

Mirrors the kubejobs simple tutorial: build one job spec, optionally apply it.
Defaults to a small TabICL run; override flags to change the experiment.
"""

from __future__ import annotations

import argparse
import subprocess
from typing import Dict

from kubejobs.jobs import GPU_PRODUCT, KueueQueue, KubernetesJob

from submit_job import _build_training_command, _parse_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tutorial: generate/apply a single kubejobs spec."
    )
    parser.add_argument("--image", required=True, help="Training image with this repo.")
    parser.add_argument("--user-email", required=True, help="User email for annotations.")
    parser.add_argument("--job-name", default="tabfm-tutorial", help="Kubernetes job name.")
    parser.add_argument("--queue", default=KueueQueue.INFORMATICS, help="Kueue queue name.")
    parser.add_argument("--namespace", default=None, help="Namespace to submit into.")
    parser.add_argument("--gpu-product", default=GPU_PRODUCT.NVIDIA_A100_SXM4_40GB)
    parser.add_argument("--gpu-limit", type=int, default=1)
    parser.add_argument("--cpu-request", default="12")
    parser.add_argument("--ram-request", default="64Gi")
    parser.add_argument("--shm-size", default="32Gi")
    parser.add_argument("--pvc-name", default=None, help="Optional PVC to mount.")
    parser.add_argument("--pvc-mount", default="/workspace/shared")
    parser.add_argument(
        "--env",
        nargs="*",
        default=[],
        help="Extra env vars KEY=VALUE (e.g., HF_TOKEN=...).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print YAML.")

    # Experiment knobs (small defaults to keep it light).
    parser.add_argument("--model", default="tabicl", choices=["tabicl", "tabpfn"])
    parser.add_argument("--device", default="cuda", help="TabPFN device (ignored for TabICL).")
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-valid", type=int, default=500)
    parser.add_argument("--max-test", type=int, default=500)

    args = parser.parse_args()

    env_vars: Dict[str, str] = {"PYTHONPATH": "/workspace/tab_fm_link_pred/src"}
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

    training_ns = argparse.Namespace(
        model=args.model,
        device=args.device,
        max_train=args.max_train,
        max_valid=args.max_valid,
        max_test=args.max_test,
    )

    job = KubernetesJob(
        name=args.job_name,
        image=args.image,
        kueue_queue_name=args.queue,
        command=["/bin/bash"],
        args=["-lc", _build_training_command(training_ns)],
        gpu_type="nvidia.com/gpu",
        gpu_product=args.gpu_product,
        gpu_limit=args.gpu_limit,
        cpu_request=args.cpu_request,
        ram_request=args.ram_request,
        backoff_limit=1,
        shm_size=args.shm_size,
        env_vars=env_vars,
        volume_mounts=volume_mounts,
        user_email=args.user_email,
        namespace=args.namespace,
    )

    yaml_spec = job.generate_yaml()
    print(yaml_spec)

    if args.dry_run:
        print("Dry-run: not applying (pass --dry-run to suppress apply).")
        return

    subprocess.run(["kubectl", "apply", "-f", "-"], input=yaml_spec.encode(), check=True)
    print(f"Submitted job {args.job_name}")


if __name__ == "__main__":
    main()
