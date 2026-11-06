# Boltz2

## Scope

- Boltz 2.2.1, upstream revision `cb04aeccdd480fd4db707f0bbafde538397fa2ac`.
- Python 3.11; PyTorch 2.7.1/cu128; Linux x86-64.
- Native `boltz` CLI in the image; host-side batch launcher with automatic full-output verification.
- Weights and chemical-component assets mounted separately.
- Build recipe pins the base digest and Python package versions; Ubuntu/PPA packages remain repository-resolved, not a byte-identical OS rebuild.
- Original CYP scientific settings are not supplied by the shared execution example.

## Requirements

- Docker Engine, Docker Compose v2 for the optional Compose route, NVIDIA Container Toolkit.
- NVIDIA driver compatible with CUDA 12.8.
- Host Python 3.12+ and PyYAML for `batch.py`.
- Writable output/cache directories; one allocated GPU.
- Trusted checkpoints and run-owned output files only: checkpoint/readback code deserializes Python pickle data.
- HTTPS during image build and explicit asset download; batch inference runs without network access.

## Build

```sh
cd cofolding/boltz2
docker build -t container-library/boltz2:2.2.1 .
python3 -m venv .venv
.venv/bin/pip install -r requirements-host.txt
```

- No GHCR or Docker Hub image has been published yet.
- Source revision is fetched and checked during build.
- `requirements.lock` pins the package versions from the retained working runtime.
- The container defaults to non-root UID/GID 1000; batch runs use the invoking user's IDs.

## External assets

```sh
python3 fetch_assets.py --cache /absolute/path/to/cache --download
python3 fetch_assets.py --cache /absolute/path/to/cache
```

- `--download`: explicit download/extraction from the revision in `assets.json`.
- Without that flag: verify existing files only.
- All three archives/checkpoints are size/hash checked; existing mismatches fail rather than being overwritten.
- Upstream Boltz 2.2.1 checks for the affinity checkpoint even for structure-only inference. The batch launcher replaces only that download step with offline checks, requiring the selected structure checkpoint and extracted `mols/` directory; the model/parser/writer calls are unchanged. The asset helper provisions the complete cache for both batch and native CLI use.
- Upstream [code and weights license](https://github.com/jwohlwend/boltz/tree/cb04aeccdd480fd4db707f0bbafde538397fa2ac): MIT.
- The image retains installed dependency metadata/licenses and upstream source LICENSE; see `licenses/Boltz-MIT.txt`.

## Shared protein–ligand test

```sh
.venv/bin/python batch.py \
  --input examples --output /absolute/path/to/new-output \
  --cache /absolute/path/to/cache \
  --checkpoint /absolute/path/to/cache/boltz2_conf.ckpt \
  --image container-library/boltz2:2.2.1 \
  --gpu GPU-REPLACE-WITH-ALLOCATED-UUID --seeds 0 \
  --diffusion_samples 1 --recycling_steps 3 --sampling_steps 200 \
  --max_parallel_samples 1 --no_kernels
```

- Canonical case: `../test-case/input.json`; native representation: `examples/shared.yaml`.
- Arbitrary Trp-cage/ethanol execution test, not a binding benchmark.
- `--no_kernels` is required for the tested Blackwell route; older supported GPUs may use native CUDA kernels.
- Re-run the identical command with `--resume` to verify and skip the completed seed.
- Interrupted seeds are preserved; use a new output directory for a fresh attempt.
- `--dry-run` validates local inputs/assets/image and prints commands without inference.
- Use `batch.py --help` for resource limits and lock-directory configuration.

## Batch inputs and outputs

- Input directory: native YAML files only; exactly one canonical protein chain per file, optional SMILES ligands.
- Explicit `msa: empty` or local `.a3m`; keep A3M files outside the YAML directory.
- Templates, constraints, CCD inputs, modified polymers, and affinity are outside this batch interface.
- Native Boltz CLI remains available for other upstream features; those routes do not inherit batch verification.
- One native process/model load per seed, processing every YAML in that directory.
- Output: request/settings and input hashes; native CIF/confidence files; consumed features and full model outputs; independent verification receipt per seed.
- Prediction coordinates are retained independently of biological quality.

## Optional native CLI with Compose

```sh
cp .env.example .env
# Set real directories, user/group IDs, and one GPU UUID in .env.
docker compose run --rm boltz predict /workspace/inputs/shared.yaml \
  --out_dir /workspace/outputs/native-test --cache /workspace/cache \
  --model boltz2 --seed 0 --diffusion_samples 1 --no_kernels
```

- Mount `examples/` as the input directory to use the same shared case.
- Compose exposes the native CLI; it does not add full-output capture/readback.
- Supply an existing local MSA or explicitly request upstream MSA service use; `--use_msa_server` sends sequence data externally.
- The legacy shell wrapper is omitted; no `.env` shell sourcing or readonly `UID` assignment.

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -v
```

- CPU tests: shared input consistency, command construction, paths, image resolution, verification, and error handling.
- GPU acceptance status is recorded in `TESTING.md` after execution.
