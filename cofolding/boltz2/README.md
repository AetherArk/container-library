# Boltz2

- Boltz 2.2.1 (`cb04aec`); Python 3.11; PyTorch 2.7.1/cu128; Linux x86-64.

## Prerequisites

- NVIDIA GPU and CUDA-12.8-compatible driver.
- [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), configured for GPU access.
- Git; host Python 3.12+ with pip/venv; Docker access for the current user.
- GPU UUID: `nvidia-smi --query-gpu=uuid,name --format=csv`.

## Setup

- Clone: `git clone https://github.com/AetherArk/container-library.git`.
- Directory: `cd container-library/cofolding/boltz2`.
- Build: `docker build -t container-library/boltz2:2.2.1 .` (no registry image yet).
- Host environment: `python3 -m venv .venv && .venv/bin/pip install -r requirements-host.txt`.
- Assets: `python3 fetch_assets.py --cache /path/to/cache --download`.
- [Weights/data](https://huggingface.co/boltz-community/boltz-2): public, no login; downloader verifies pinned files and extracts chemical-component data. Downloads 6.2 GB, including affinity weights; allow additional space for extraction, image, and outputs.
- Inference: offline; external weights; trusted checkpoints/output files only.

## Examples

- CYP: `.venv/bin/python run_test.py --case cyp --output /path/to/new-run --cache /path/to/cache --checkpoint /path/to/cache/boltz2_conf.ckpt --gpu GPU-UUID`.
- PXR: replace `--case cyp` with `--case pxr`; use a separate output directory.
- [Inputs/MSAs/settings](../test-cases/); [verified execution and output checks](TESTING.md).
- Results: `OUTPUT/result.json`; structures, confidence arrays, full tensors, and verification under `OUTPUT/prediction/seed-11/`.

## Custom inputs

- Run: `.venv/bin/python batch.py --input examples --output /path/to/new-run --cache /path/to/cache --checkpoint /path/to/cache/boltz2_conf.ckpt --gpu GPU-UUID --seeds 0 --diffusion_samples 1 --no_kernels`.
- YAML directory: one protein chain per input, SMILES ligands, explicit `msa: empty` or local A3M. MSA generation is external.
- Templates, CCD inputs, constraints, affinity: native CLI via [Compose](compose.yaml).
- `--no_kernels`: tested Blackwell route; packaged examples use their configured kernels.
- `--resume`: verify/reuse completed runs; `--dry-run`: stage inputs and print the request.
- Options/resource limits: `batch.py --help`. CPU tests: `.venv/bin/python -m unittest discover -s tests -v`.
- Code/weights: [MIT](licenses/Boltz-MIT.txt).
