#!/usr/bin/env python3
"""Render a Kubernetes Job YAML for tab_fm_link_pred experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_command(model: str, args: argparse.Namespace) -> str:
    steps = [
        "set -euxo pipefail",
        "echo \"Step 1: apt-get update\"",
        "apt-get update",
        "echo \"Step 2: install deps\"",
        "apt-get install -y git curl python3",
        "echo \"Step 3: install uv\"",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "echo \"Step 4: clone repo\"",
        f"git clone {args.repo} /workspace/tab_fm_link_pred",
        "echo \"Step 5: uv sync\"",
        "cd /workspace/tab_fm_link_pred",
        "/root/.local/bin/uv sync --frozen",
        "/root/.local/bin/uv pip install huggingface_hub",
    ]

    if model == "limix":
        steps.extend(
            [
                "echo \"Step 6: clone LimiX\"",
                f"git clone {args.limix_repo} /workspace/LimiX",
                "echo \"Step 7: install LimiX deps\"",
                "/root/.local/bin/uv pip install torch torchvision torchaudio scikit-learn "
                "einops matplotlib networkx numpy pandas scipy tqdm typing_extensions "
                "xgboost kditransform hyperopt",
                "echo \"Step 8: run experiment\"",
                "/root/.local/bin/uv run python -u main.py --model limix "
                f"--limix-path /workspace/LimiX "
                f"--limix-config /workspace/LimiX/config/cls_default_noretrieval.json "
                f"--max-train {args.max_train} --max-valid {args.max_valid} "
                f"--max-test {args.max_test} --output {args.output}",
            ]
        )
    elif model == "tabdpt":
        steps.extend(
            [
                "echo \"Step 6: clone TabDPT\"",
                f"git clone {args.tabdpt_repo} /workspace/TabDPT-inference",
                "echo \"Step 7: install TabDPT deps\"",
                "cd /workspace/TabDPT-inference",
                "/root/.local/bin/uv sync",
                "/root/.local/bin/uv pip install faiss-cpu",
                "echo \"Step 8: run experiment\"",
                "cd /workspace/tab_fm_link_pred",
                "/root/.local/bin/uv run python -u main.py --model tabdpt "
                f"--tabdpt-path /workspace/TabDPT-inference "
                f"--max-train {args.max_train} --max-valid {args.max_valid} "
                f"--max-test {args.max_test} --output {args.output}",
            ]
        )
    else:
        steps.extend(
            [
                "echo \"Step 6: run experiment\"",
                f"/root/.local/bin/uv run python -u main.py --model {model} "
                f"--max-train {args.max_train} --max-valid {args.max_valid} "
                f"--max-test {args.max_test} --output {args.output}",
            ]
        )

    return "\n".join(steps)


def render_yaml(args: argparse.Namespace) -> str:
    env_lines = [
        "            - name: PYTHONUNBUFFERED",
        "              value: \"1\"",
        "            - name: HF_HOME",
        f"              value: {args.hf_home}",
    ]
    if not args.skip_hf_token:
        env_lines.extend(
            [
                "            - name: HF_TOKEN",
                "              valueFrom:",
                "                secretKeyRef:",
                f"                  name: {args.hf_secret}",
                f"                  key: {args.hf_key}",
            ]
        )
    if args.model in {"limix", "tabdpt"}:
        env_lines.extend(
            [
                "            - name: PYTHONPATH",
                f"              value: {args.pythonpath}",
            ]
        )

    command = build_command(args.model, args)
    env_block = "\n".join(env_lines)
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  annotations: &id001
    eidf/user: {args.user}
    groups: {args.groups}
    home: {args.home}
    login_user: {args.user}
    shell: /bin/bash
  labels: &id002
    eidf/user: {args.user}
    kueue.x-k8s.io/queue-name: {args.queue}
  generateName: {args.job_prefix}-{args.model}-
  namespace: {args.namespace}
spec:
  backoffLimit: 4
  template:
    metadata:
      annotations: *id001
      labels: *id002
    spec:
      containers:
        - name: {args.job_prefix}-{args.model}
          image: {args.image}
          env:
{env_block}
          command:
            - /bin/bash
            - -lc
            - |
{indent(command, 14)}
          resources:
            limits:
              cpu: {args.cpu}
              memory: {args.memory}
              nvidia.com/gpu: {args.gpu}
            requests:
              cpu: {args.cpu}
              memory: {args.memory}
      restartPolicy: Never
"""


def indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(f"{pad}{line}" for line in text.splitlines())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tabicl", "tabpfn", "limix", "tabdpt"], required=True)
    parser.add_argument("--max-train", type=int, default=1000)
    parser.add_argument("--max-valid", type=int, default=500)
    parser.add_argument("--max-test", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--image", type=str, default="ubuntu:22.04")
    parser.add_argument("--cpu", type=str, default="1")
    parser.add_argument("--memory", type=str, default="4Gi")
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--namespace", type=str, default="eidf097ns")
    parser.add_argument("--queue", type=str, default="eidf097ns-user-queue")
    parser.add_argument("--user", type=str, default="pminervini")
    parser.add_argument("--groups", type=str, default="eidf097 eidf097-host1-login eidf-gateway-login")
    parser.add_argument("--home", type=str, default="/home/eidf097/eidf097/pminervini")
    parser.add_argument("--job-prefix", type=str, default="raj")
    parser.add_argument("--hf-secret", type=str, default="hf-token")
    parser.add_argument("--hf-key", type=str, default="token")
    parser.add_argument("--hf-home", type=str, default="/workspace/.cache/huggingface")
    parser.add_argument("--skip-hf-token", action="store_true")
    parser.add_argument("--repo", type=str, default="https://github.com/braj29/tab_fm_link_pred")
    parser.add_argument("--limix-repo", type=str, default="https://github.com/limix-ldm/LimiX")
    parser.add_argument("--tabdpt-repo", type=str, default="https://github.com/layer6ai-labs/TabDPT-inference")
    parser.add_argument("--pythonpath", type=str, default="/workspace/LimiX:/workspace/TabDPT-inference/src:/workspace/tab_fm_link_pred/src")
    parser.add_argument("--out", type=str, default="kube/job_rendered.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = f"/workspace/tab_fm_link_pred/experiment_metrics_{args.model}.json"
    yaml_text = render_yaml(args)
    out_path = Path(args.out)
    out_path.write_text(yaml_text)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
