# Boltz2

- Boltz 2.2.1 (`cb04aec`); Python 3.11; PyTorch 2.7.1/cu128; Linux x86-64.
- Requires Docker, NVIDIA Container Toolkit, CUDA-12.8-compatible driver, and host Python 3.12+.
- Weights stay external. Batch inference is offline; use trusted checkpoints/output files.

## Setup

```sh
cd cofolding/boltz2
docker build -t container-library/boltz2:2.2.1 .
python3 -m venv .venv
.venv/bin/pip install -r requirements-host.txt
python3 fetch_assets.py --cache /path/to/cache --download
```

- Source, base image, Python packages, and model assets are pinned; OS packages remain repository-resolved.
- Asset extraction is atomic; incomplete/unmarked caches are refused. Omit `--download` to verify an existing cache.

## Packaged examples

```sh
.venv/bin/python run_test.py --case cyp \
  --output /path/to/new-cyp-test --cache /path/to/cache \
  --checkpoint /path/to/cache/boltz2_conf.ckpt --gpu GPU-UUID
# Repeat with --case pxr and a separate output directory.
```

- Canonical inputs, original MSAs, method settings, and native output inventories: [test-cases](../test-cases/).
- Pass requires the batch native structure, ligand/cofactor chemistry, and confidence verification.
- Summary: `OUTPUT/result.json`; native outputs and `verified.json`: `OUTPUT/prediction/seed-11/`.
- `--resume` rechecks the run's own outputs with identical inputs/settings; `--dry-run` stages inputs and prints the batch request without inference or a PASS report.
- PXR and CYP examples passed GPU execution and native output verification; see [verification](TESTING.md).

## Other inputs

```sh
.venv/bin/python batch.py --input examples --output /path/to/new-run \
  --cache /path/to/cache --checkpoint /path/to/cache/boltz2_conf.ckpt \
  --gpu GPU-UUID --seeds 0 --diffusion_samples 1 --no_kernels
```

- YAML directory: one protein chain, SMILES ligands, explicit empty/local A3M alignment.
- One model load per seed. Saves native structures/confidence, full tensors, and verification receipts.
- Templates, CCD inputs, constraints, and affinity require the native CLI; optional Compose configuration is supplied.
- `--no_kernels`: tested Blackwell route. Packaged examples retain their native kernel setting.
- Options: `batch.py --help`. Tests: `.venv/bin/python -m unittest discover -s tests -v`.
- [Tested scope](TESTING.md). Code/weights: [Boltz MIT](licenses/Boltz-MIT.txt). No registry image published yet.
