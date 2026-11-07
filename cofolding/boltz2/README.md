# Boltz2

- Boltz 2.2.1 (`cb04aec`); Python 3.11; PyTorch 2.7.1/cu128; Linux x86-64.
- Persistent worker: one checkpoint load across every input × external seed in one invocation.
- Container release candidate: `2.2.1-worker-v2-cloud-rc1`; not published yet.

## Prerequisites

- NVIDIA GPU, CUDA-12.8-compatible driver, [Docker](https://docs.docker.com/engine/install/), [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Git; host Python 3.12+ with pip/venv; Docker access.
- GPU UUID: `nvidia-smi --query-gpu=uuid,name --format=csv`.

## Setup

- Clone: `git clone https://github.com/AetherArk/container-library.git`; enter `container-library` and check out the commit containing this worker.
- Host environment: `python3 -m venv .venv && .venv/bin/pip install -r cofolding/boltz2/requirements-host.txt`.
- Build: `docker build --build-arg CONTAINER_REVISION="$(git rev-parse HEAD)" -t container-library/boltz2:2.2.1-worker-v2-cloud-rc1 cofolding/boltz2`.
- Assets: `.venv/bin/python cofolding/boltz2/fetch_assets.py --cache /path/to/cache --download`.
- [Weights/data](https://huggingface.co/boltz-community/boltz-2): public; downloader verifies pinned files and extracts chemical-component data. Download: 6.2 GB, including affinity weights; allow additional extraction/image/output space.
- Inference is offline. Weights remain external; use trusted checkpoints/output files.

## Run

- `.venv/bin/python run_cofold.py --image container-library/boltz2:2.2.1-worker-v2-cloud-rc1 --inputs /path/to/native-inputs --assets /path/to/cache --checkpoint /path/to/cache/boltz2_conf.ckpt --output /path/to/new-results --gpu GPU-UUID --seeds 11 12 --samples 5 --no-kernels`.
- Example inputs: `cofolding/boltz2/tests/fixtures/resident_inputs` (Trp-cage with two small ligands).
- Native YAML unchanged; every listed seed applies to every input. MSA generation is external; local references must be inside the mounted input/assets directories.
- Tested input shape: one canonical protein chain, SMILES ligands, explicit empty/local A3M alignment. Resident worker is structure-only; affinity requires the native CLI via [Compose](compose.yaml).
- Options: `--samples`, `--recycles`, `--steps`, `--max-parallel-samples`, `--use-potentials`; `--no-kernels` is the tested Blackwell route.
- `--resume`: verify/reuse completed jobs; `--dry-run`: print the resolved launch command. Limits and interruption grace: `run_cofold.py --help`.

## Cloud transport

- [Prepare per-job inputs in R2 and run `cofold-cloud-run`](CLOUD_JOBS.md): an additive wrapper around the same resident worker; no automatic Salad deployment.

## Outputs and tests

- Raw native predictions, confidence, embeddings, full tensor captures and receipts: `OUTPUT/jobs/`.
- Final metadata: `OUTPUT/run.json` (provisional format); live progress: `OUTPUT/progress.json`.
- [Local CPU/image/GPU tests](../../LOCAL_TESTING.md); [tested scope](TESTING.md); [CYP/PXR example inputs](../test-cases/).
- Legacy batch examples remain available through `run_test.py`/`batch.py`; they are separate from the resident-worker route.
- [GHCR publication](PUBLISHING.md). Code/weights: [MIT](licenses/Boltz-MIT.txt).
