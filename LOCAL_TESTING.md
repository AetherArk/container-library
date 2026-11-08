# Local candidate testing

- Local branch/image only; no push or registry publication.
- Host Python with PyYAML: existing `cofolding/boltz2/.venv/bin/python` is suitable.
- Native YAML stays unchanged; the external seed list applies to every input.

## Build

- Boltz2: `docker build -t container-library/boltz2:single-load-dev cofolding/boltz2`.
- ESMFold2: `docker build -t container-library/esmfold2:dev cofolding/esmfold2` (CUDA 13 devel
  base; the transformer-engine build dominates the time).
- Development tags only; preserve the released image tags.

## CPU

- Boltz2: `cofolding/boltz2/.venv/bin/python scripts/test_local.py --suite cpu`.
- ESMFold2: `python3 -m unittest discover -s cofolding/esmfold2/tests` with an interpreter that
  has pyyaml, numpy, rdkit, gemmi and CPU torch.
- Input handling, one-load lifecycle, seed mapping, failure isolation, resume, signals, launcher ownership.
- Model/subprocess doubles are unit tests, not GPU evidence.

## Installed image

- Boltz2: `python3 scripts/test_local.py --suite image --image container-library/boltz2:single-load-dev`.
- Installed entry point and imports; no GPU or host implementation-code mounts.
- Cloud installed command/imports: `python3 scripts/test_local.py --suite image-cloud --image container-library/boltz2:single-load-dev`.
- CPU suite also discovers cloud transport tests (local HTTP + explicitly fake worker; no R2, GPU or inference). See [cloud jobs](cofolding/boltz2/CLOUD_JOBS.md).
- ESMFold2: `docker run --rm --entrypoint python <image> -m unittest discover -s /pkg/tests`
  with the package mounted read-only, plus `cofold-run --help`, `cofold-throughput --help` and
  the pinned-version/uid assertions in `.github/workflows/esmfold2-publish.yml`.

## GPU

- Boltz2: `python3 scripts/test_local.py --suite gpu --image container-library/boltz2:single-load-dev --inputs cofolding/boltz2/tests/fixtures/resident_inputs --assets /path/to/assets --checkpoint /path/to/assets/boltz2_conf.ckpt --output /path/to/new-test --gpu GPU-UUID --seeds 11 12 --no-kernels`.
- Two ligand-containing inputs, two seeds, default sampling schedule.
- Checks: one checkpoint load/transfer, stable CUDA model storage, exact requested seed coverage, native outputs and completion receipts.
- Use an allocated GPU and fresh output directory; input dependencies must be accessible under `/inputs` or `/models`. Relative references resolve from each YAML.
- ESMFold2: `python3 run_cofold.py --image container-library/esmfold2:dev --inputs cofolding/esmfold2/examples --assets /path/to/model --checkpoint /path/to/model --output /path/to/new-test --gpu GPU-UUID --seeds 11 --samples 1`. `--checkpoint` is the model directory here; the launcher accepts either a file or a directory.

## Normal launch

- `python3 run_cofold.py --image container-library/boltz2:single-load-dev --inputs /path/to/inputs --assets /path/to/assets --checkpoint /path/to/assets/boltz2_conf.ckpt --output /path/to/results --gpu GPU-UUID --seeds 11 12`.
- One offline container; inputs/assets/checkpoint read-only; no automatic restart.
- `--resume`: verify/reuse completed jobs. `--dry-run`: print the resolved command.
- Method options: `--samples`, `--recycles`, `--steps`, `--max-parallel-samples`, `--no-kernels`, `--use-potentials`, `--output-format`. For ESMFold2, `--recycles` means `num_loops`, `--steps` means `num_sampling_steps`, `--max-parallel-samples` must be 1 and `--output-format pdb` is refused by the worker.
- Limits: `--cpus 8 --memory 36g`; optional whole-run `--timeout`; interruption grace `--stop-grace 120` seconds.
- Shared GPU locks: `--lock-root` or `COFOLD_GPU_LOCK_ROOT`; active compute blocks launch. Serving management stays outside the launcher.
- Live status: `OUTPUT/progress.json`. Final inputs/seeds/outcomes: `OUTPUT/run.json` (provisional metadata format).
