# Local candidate testing

- Local branch/image only; no push or registry publication.
- Host Python with PyYAML: existing `cofolding/boltz2/.venv/bin/python` is suitable.
- Native YAML stays unchanged; the external seed list applies to every input.

## Build

- `docker build -t container-library/boltz2:single-load-dev cofolding/boltz2`.
- Development tag only; preserve the released image tag.

## CPU

- `cofolding/boltz2/.venv/bin/python scripts/test_local.py --suite cpu`.
- Input handling, one-load lifecycle, seed mapping, failure isolation, resume, signals, launcher ownership.
- Model/subprocess doubles are unit tests, not GPU evidence.

## Installed image

- `python3 scripts/test_local.py --suite image --image container-library/boltz2:single-load-dev`.
- Installed entry point and imports; no GPU or host implementation-code mounts.

## GPU

- `python3 scripts/test_local.py --suite gpu --image container-library/boltz2:single-load-dev --inputs cofolding/boltz2/tests/fixtures/resident_inputs --assets /path/to/assets --checkpoint /path/to/assets/boltz2_conf.ckpt --output /path/to/new-test --gpu GPU-UUID --seeds 11 12 --no-kernels`.
- Two ligand-containing inputs, two seeds, default sampling schedule.
- Checks: one checkpoint load/transfer, stable CUDA model storage, exact requested seed coverage, native outputs and completion receipts.
- Use an allocated GPU and fresh output directory; input dependencies must be accessible under `/inputs` or `/models`. Relative references resolve from each YAML.

## Normal launch

- `python3 run_cofold.py --image container-library/boltz2:single-load-dev --inputs /path/to/inputs --assets /path/to/assets --checkpoint /path/to/assets/boltz2_conf.ckpt --output /path/to/results --gpu GPU-UUID --seeds 11 12`.
- One offline container; inputs/assets/checkpoint read-only; no automatic restart.
- `--resume`: verify/reuse completed jobs. `--dry-run`: print the resolved command.
- Method options: `--samples`, `--recycles`, `--steps`, `--max-parallel-samples`, `--no-kernels`, `--use-potentials`, `--output-format`.
- Limits: `--cpus 8 --memory 36g`; optional whole-run `--timeout`; interruption grace `--stop-grace 120` seconds.
- Shared GPU locks: `--lock-root` or `COFOLD_GPU_LOCK_ROOT`; active compute blocks launch. Serving management stays outside the launcher.
- Live status: `OUTPUT/progress.json`. Final inputs/seeds/outcomes: `OUTPUT/run.json` (provisional metadata format).
