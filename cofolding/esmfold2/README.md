# ESMFold2

- ESM 3.4.1.post1 (`esm[fused]`, PyPI wheel; upstream commit `43b4548`); Python 3.12; PyTorch 2.11.0/cu130; Linux x86-64.
- Model: `biohub/ESMFold2`, revision `69869f737beffec5294845ede23db5fc0b4f509e`; weights stay outside the image.
- Persistent worker: one checkpoint load across every input × external seed in one invocation.
- Inputs and outputs use the same native YAML format and job/receipt layout as the [Boltz2 container](../boltz2/). [Interface specification](SPEC.md).
- Container release candidate: `3.4.1.post1-worker-rc1`; not published yet.

## Prerequisites

- NVIDIA GPU with a CUDA-13-compatible driver, [Docker](https://docs.docker.com/engine/install/), [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Git; host Python 3.12+ with pip/venv for the launcher.
- GPU UUID: `nvidia-smi --query-gpu=uuid,name --format=csv`.

## Setup

- Clone: `git clone https://github.com/AetherArk/container-library.git`; enter `container-library`.
- Host environment: `python3 -m venv .venv && .venv/bin/pip install -r cofolding/boltz2/requirements-host.txt` (the launcher is shared).
- Build: `docker build --build-arg CONTAINER_REVISION="$(git rev-parse HEAD)" -t container-library/esmfold2:3.4.1.post1-worker-rc1 cofolding/esmfold2`.
- Assets: `.venv/bin/python cofolding/esmfold2/fetch_assets.py --assets /path/to/cache --download`.
- Download is about 26.7 GB including weights and `ccd.pkl`; inference itself is offline.

## Run

- `.venv/bin/python run_cofold.py --image container-library/esmfold2:3.4.1.post1-worker-rc1 --inputs cofolding/esmfold2/examples --assets /path/to/cache --checkpoint /path/to/cache --output /path/to/new-results --gpu GPU-UUID --seeds 11 --samples 1 --steps 100 --recycles 20`
- `--checkpoint` is the directory that directly contains `config.json` and `ccd.pkl`; `--assets`
  may be that same directory. Under a Hugging Face cache this is
  `huggingface/hub/models--biohub--ESMFold2/snapshots/69869f73...`. The directory is hashed once per invocation.
- Native YAML unchanged; every listed seed applies to every input. Inputs are the same files the Boltz2 image consumes.

## Engine capability

| Entity | Supported |
|---|---|
| Protein | exactly one chain, canonical sequence |
| Ligand | one SMILES ligand |
| Heme | one optional ligand containing iron (chain H) |
| MSA | `empty` only |
| Templates, constraints, affinity, modified polymers, CCD ligands | rejected with the capability name |

`examples/trpcage-ethanol.yaml` and `examples/cyp.yaml` cover the supported shapes.
Multi-chain inputs such as the shared Zika panel are refused; use the Boltz2 image for those.

## Options

`--samples` (independent N=1 calls per seed, default 1), `--steps` (num_sampling_steps, default
100), `--recycles` (num_loops, default 20), `--backend fused|reference`, `--no-kernels` (alias for
reference), `--resume`, `--dry-run`. `--max-parallel-samples` must be 1: the engine samples one
trajectory per call. `cofold-throughput sweep` defaults to the same numbers and refuses levels
above 1.

Seed mapping is the frozen native one: base seed `S`, one-based sample index `i` →
`S*1000 + i - 1`.

## Outputs and tests

- Raw mmCIF, confidence JSON, PAE/pLDDT/embedding arrays, consumed features, complete native result tensors and receipts: `OUTPUT/jobs/`.
- Final metadata: `OUTPUT/run.json`; live progress: `OUTPUT/progress.json`; append-only `predictions.jsonl` and `events.jsonl`.
- [Local CPU tests](TESTING.md); [interface specification](SPEC.md); [worker details](resident/README.md).
- Native tensors and coordinates are retained; no fitting, calibration or filtering is applied.
