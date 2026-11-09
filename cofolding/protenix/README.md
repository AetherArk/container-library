# Protenix

- protenix 2.0.0 (PyPI wheel) with the released `protenix-v2` checkpoint; Python 3.12; PyTorch
  2.7.1/cu128; Linux x86-64.
- Model artifacts: the five files in `assets.json` — the checkpoint from
  `huggingface.co/TMF001/protenix-v2-weights` (revision `653edab`) and the four `common/` cache
  files from the Protenix object store. Weights stay outside the image.
- Persistent worker: one model initialization across every input × external seed in one
  invocation.
- Inputs use the shared native YAML format; Protenix's native input is AF3-style JSON, generated
  per job and retained. Outputs use the same job/receipt layout as the
  [ESMFold2 container](../esmfold2/) and [Chai-1](../chai1/). [Interface specification](SPEC.md).
- Container release candidate: `2.0.0-worker-rc1` (**not published yet**).

## Container image

No image has been built or published for this container. The Dockerfile is the source of truth;
see [PUBLISHING.md](PUBLISHING.md) for the workflow that will produce the first tag. Do not use a
tag that is not listed in the root README table.

## Prerequisites

- NVIDIA GPU with a driver for CUDA 12.8 (`sm_120` / RTX 50-series is the measured target).
  protenix-v2 refuses more than 2560 tokens upstream; the packaged examples are far smaller.
  Memory fit for those examples is **not yet measured** in this image.
- [Docker](https://docs.docker.com/engine/install/),
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Git; host Python 3.12+ with pip/venv for the launcher.
- GPU UUID: `nvidia-smi --query-gpu=uuid,name --format=csv`.

## Setup

- Clone: `git clone https://github.com/AetherArk/container-library.git`; enter `container-library`.
- Host environment: `python3 -m venv .venv && .venv/bin/pip install -r cofolding/boltz2/requirements-host.txt` (the launcher is shared).
- Build: `docker build --build-context common=cofolding/common --build-arg CONTAINER_REVISION="$(git rev-parse HEAD)" -t container-library/protenix:2.0.0-worker-rc1 cofolding/protenix`. The named `common` context installs `cofolding/common` at `/opt/cofold/cofolding/common` so the worker can import the shared numpy-only interface-scoring module.
- Assets: `.venv/bin/python cofolding/protenix/fetch_assets.py --assets /path/to/cache --download`.
- The download is about 2.6 GB (1.86 GB checkpoint, 0.63 GB CCD/cluster cache); inference itself is
  offline.

## Run

- `.venv/bin/python run_cofold.py --image container-library/protenix:2.0.0-worker-rc1 --inputs cofolding/protenix/examples --assets /path/to/cache --checkpoint /path/to/cache --output /path/to/new-results --gpu GPU-UUID --seeds 11 --samples 1 --steps 200 --recycles 10 --memory 32g`
- `--checkpoint` is the same directory as `--assets`: the `PROTENIX_ROOT_DIR` that directly
  contains `checkpoint/protenix-v2.pt` and `common/`. It is hashed once per invocation, and the
  worker refuses to run if the two paths differ.

## Engine capability

| Entity | Supported |
|---|---|
| Protein | one or more chains, canonical sequences, unique ids |
| Ligand | one or more SMILES ligands |
| Apo (protein-only) | accepted |
| Heme | accepted as an ordinary SMILES ligand |
| Ion | rejected as an entity; pass a one-atom SMILES ligand instead |
| MSA | `empty` only (Protenix's single-sequence path) |
| Templates, constraints, affinity, modified polymers, CCD ligands, RNA/DNA | rejected with the capability name |

`examples/trpcage-ethanol.yaml`, `examples/two-chain.yaml` and `examples/small-two-ligand.yaml`
cover the supported shapes.

## Options

`--samples` (independent N=1 forwards per seed, default 1), `--steps`
(`sample_diffusion.N_step`, default 200), `--recycles` (`model.N_cycle`, default 10),
`--resume`, `--derived none|tensors`, `--dry-run`. `--max-parallel-samples` must be 1.
`cofold-throughput sweep` defaults to the same numbers and refuses levels above 1.

Seed mapping is `base seed S`, zero-based sample index `i` → `S*1000 + i`.

## Outputs and tests

- Written mmCIF, confidence JSON, PAE/PDE/per-atom pLDDT arrays, the raw Protenix writer directory,
  the explicit PAE token map and receipts: `OUTPUT/jobs/`.
- Protenix has no PAE writer: the matrix lives in the confidence head's
  `full_data["token_pair_pae"]`. The container enables the engine's full-data dump so the writer
  emits its own copy, and scores the unrounded tensor before it is dropped. See SPEC.md §5.1.
- Interface scores: `OUTPUT/jobs/<key>/<attempt>/metrics/derived.json` (shared ipSAE module).
- Final metadata: `OUTPUT/run.json`; live progress: `OUTPUT/progress.json`; append-only
  `predictions.jsonl` and `events.jsonl`.
- [Local CPU tests](TESTING.md); [interface specification](SPEC.md).
- Native tensors and coordinates are retained; no fitting, calibration, filtering or re-ranking is
  applied.
