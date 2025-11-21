#!/usr/bin/env python3
"""Submit Tabular FM experiments to Kubernetes using kubejobs."""

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

from kubejobs.jobs import GPU_PRODUCT, KueueQueue, KubernetesJob


def _parse_env(env_pairs: List[str]) -> Dict[str, str]:
    env_vars: Dict[str, str] = {}
    for pair in env_pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(
                f"Environment variables must look like KEY=VALUE, got '{pair}'"
            )
        key, value = pair.split("=", 1)
        env_vars[key] = value
    return env_vars


def _build_training_command(args: argparse.Namespace) -> str:
    cmd = f"PYTHONPATH=src python main.py --model {args.model}"
    if args.device:
        cmd += f" --device {args.device}"
    if args.max_train is not None:
        cmd += f" --max-train {args.max_train}"
    if args.max_valid is not None:
        cmd += f" --max-valid {args.max_valid}"
    if args.max_test is not None:
        cmd += f" --max-test {args.max_test}"

    script_lines = [
        "set -euo pipefail",
        "cd /workspace/tab_fm_link_pred",
        cmd,
    ]
    return " && ".join(script_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and submit a Kubernetes Job for this repo."
    )
    parser.add_argument("--image", required=True, help="Container image with the repo.")
    parser.add_argument(
        "--job-name",
        default="tab-fm-link-pred",
        help="Prefix for the Kubernetes Job name.",
    )
    parser.add_argument(
        "--queue",
        default=KueueQueue.INFORMATICS,
        help="Kueue queue name (default matches kubejobs examples).",
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

    # Resources
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
    parser.add_argument(
        "--cpu-request",
        default="12",
        help="CPU request/limit for the container.",
    )
    parser.add_argument(
        "--ram-request",
        default="64Gi",
        help="RAM request/limit for the container.",
    )
    parser.add_argument(
        "--shm-size",
        default="32Gi",
        help="Size of /dev/shm via emptyDir.",
    )
    parser.add_argument("--backoff-limit", type=int, default=1)

    # PVC mounting
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

    # Training options forwarded to src/run.py
    parser.add_argument("--model", choices=["tabicl", "tabpfn"], default="tabicl")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for TabPFN (ignored for TabICL). Use 'cuda' for GPUs.",
    )
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-valid", type=int, default=500)
    parser.add_argument("--max-test", type=int, default=500)

    # Misc
    parser.add_argument(
        "--env",
        nargs="*",
        default=[],
        help="Extra env vars as KEY=VALUE pairs (e.g., HF_TOKEN=...).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kube_job.yaml"),
        help="Where to write the generated YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write the YAML; do not call kubectl.",
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

    job = KubernetesJob(
        name=args.job_name,
        image=args.image,
        kueue_queue_name=args.queue,
        command=["/bin/bash"],
        args=["-lc", _build_training_command(args)],
        gpu_type="nvidia.com/gpu",
        gpu_product=args.gpu_product,
        gpu_limit=args.gpu_limit,
        cpu_request=args.cpu_request,
        ram_request=args.ram_request,
        backoff_limit=args.backoff_limit,
        shm_size=args.shm_size,
        env_vars=env_vars,
        volume_mounts=volume_mounts,
        user_email=args.user_email,
        namespace=args.namespace,
        image_pull_secret=args.image_pull_secret,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    job_yaml = job.generate_yaml()
    args.output.write_text(job_yaml)
    print(f"Wrote job spec to {args.output}")

    if args.dry_run:
        print("Dry-run mode: not submitting to the cluster.")
        return

    subprocess.run(["kubectl", "apply", "-f", str(args.output)], check=True)
    print("Submitted job with kubectl apply.")


if __name__ == "__main__":
    main()
