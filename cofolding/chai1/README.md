# Chai-1

- chai_lab 0.6.1 (PyPI wheel); Python 3.12; PyTorch 2.11.0/cu130; Linux x86-64.
- Model artifacts: the eight files in `assets.json` from `chaiassets.com/chai1-inference-depencencies`; weights stay outside the image.
- Persistent worker: one engine initialization across every input × external seed in one invocation.
- Inputs and outputs use the same native YAML format and job/receipt layout as the [ESMFold2 container](../esmfold2/). [Interface specification](SPEC.md).
- Container release candidate: `0.6.1-worker-rc1` (**not published yet**).

## Container image

No image has been built or published for this container. The Dockerfile is the source of truth;
see [PUBLISHING.md](PUBLISHING.md) for the workflow that will produce the first tag. Do not use a
tag that is not listed in the root README table.

## Prerequisites

- NVIDIA GPU with a CUDA-13-compatible driver. Chai-1 documentation asks for 48 GB; the container
  targets a 16 GB RTX 5070 Ti with `low_memory=True` (the engine default) and the small examples.
  A 16 GB card is expected to fit 20-residue systems only; that expectation is **not yet measured**.
- [Docker](https://docs.docker.com/engine/install/),
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Git; host Python 3.12+ with pip/venv for the launcher.
- GPU UUID: `nvidia-smi --query-gpu=uuid,name --format=csv`.

## Setup

- Clone: `git clone https://github.com/AetherArk/container-library.git`; enter `container-library`.
- Host environment: `python3 -m venv .venv && .venv/bin/pip install -r cofolding/boltz2/requirements-host.txt` (the launcher is shared).
- Build: `docker build --build-context common=cofolding/common --build-arg CONTAINER_REVISION="$(git rev-parse HEAD)" -t container-library/chai1:0.6.1-worker-rc1 cofolding/chai1`. The named `common` context installs `cofolding/common` at `/opt/cofold/cofolding/common` so the worker can import the shared numpy-only interface-scoring module.
- Assets: `.venv/bin/python cofolding/chai1/fetch_assets.py --assets /path/to/cache --download`.
- The download is about 6.6 GB (5.7 GB is the traced ESM-2 3B); inference itself is offline.

## Run

- `.venv/bin/python run_cofold.py --image container-library/chai1:0.6.1-worker-rc1 --inputs cofolding/chai1/examples --assets /path/to/cache --checkpoint /path/to/cache --output /path/to/new-results --gpu GPU-UUID --seeds 11 --samples 1 --steps 200 --recycles 3 --memory 24g`
- `--checkpoint` is the same directory as `--assets`: the one that directly contains
  `models_v2/`, `conformers_v1.apkl` and `esm/`. It is hashed once per invocation.

## Engine capability

| Entity | Supported |
|---|---|
| Protein | one or more chains, canonical sequences, unique ids |
| Ligand | one or more SMILES ligands |
| Apo (protein-only) | accepted |
| Heme | accepted as an ordinary SMILES ligand |
| MSA | `empty` only |
| Templates, constraints, affinity, modified polymers, glycan strings, CCD ligands, RNA/DNA | rejected with the capability name |

`examples/trpcage-ethanol.yaml`, `examples/two-chain.yaml` and `examples/small-two-ligand.yaml`
cover the supported shapes.

## Options

`--samples` (independent N=1 fold calls per seed, default 1), `--steps` (num_diffn_timesteps,
default 200), `--recycles` (num_trunk_recycles, default 3), `--no-low-memory`, `--resume`,
`--derived none|tensors`, `--dry-run`. `--max-parallel-samples` must be 1. `cofold-throughput
sweep` defaults to the same numbers and refuses levels above 1.

Seed mapping is `base seed S`, zero-based sample index `i` → `S*1000 + i`.

## Outputs and tests

- Written mmCIF, confidence JSON, PAE/PDE/pLDDT arrays, the raw chai_lab sample directory, the
  explicit PAE token map and receipts: `OUTPUT/jobs/`.
- Interface scores: `OUTPUT/jobs/<key>/<attempt>/metrics/derived.json` (shared ipSAE module).
- Final metadata: `OUTPUT/run.json`; live progress: `OUTPUT/progress.json`; append-only
  `predictions.jsonl` and `events.jsonl`.
- [Local CPU tests](TESTING.md); [interface specification](SPEC.md).
- Native tensors and coordinates are retained; no fitting, calibration or filtering is applied.
