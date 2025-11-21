PYTHON ?= python
UV ?= uv
IMAGE ?= tab-fm-link-pred:latest
REGISTRY ?= ""
FULL_IMAGE := $(if $(REGISTRY),$(REGISTRY)/$(IMAGE),$(IMAGE))

.PHONY: help uv-sync tabicl tabpfn docker-build kubesubmit test format lint

help:
	@echo "Common commands:"
	@echo "  make uv-sync        # create .venv and install deps via uv"
	@echo "  make tabicl         # run TabICL experiment with defaults"
	@echo "  make tabpfn         # run TabPFN experiment on cuda"
	@echo "  make docker-build   # build Docker image (override IMAGE/REGISTRY)"
	@echo "  make kubesubmit     # submit kube job (requires kubectl + kubejobs)"
	@echo "  make test           # run unit tests"
	@echo "  make format         # ruff format"
	@echo "  make lint           # ruff check"

uv-sync:
	$(UV) sync

tabicl:
	PYTHONPATH=src $(UV) run main.py --model tabicl --max-train 2000 --max-valid 500 --max-test 500

tabpfn:
	PYTHONPATH=src $(UV) run main.py --model tabpfn --device cuda --max-train 2000 --max-valid 500 --max-test 500

docker-build:
	docker build -t $(FULL_IMAGE) .

kubesubmit:
	$(UV) run python kube/submit_job.py --image $(FULL_IMAGE) --user-email you@university.edu --gpu-product NVIDIA-A100-SXM4-40GB --gpu-limit 1

test:
	PYTHONPATH=src $(UV) run -m unittest discover -s tests -v

format:
	$(UV) run ruff format

lint:
	$(UV) run ruff check
