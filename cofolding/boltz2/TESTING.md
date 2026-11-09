# Verification

- Current CPU suite: 123 Boltz2 tests plus 24 repository tests passed (147 total).

## Local resident-worker candidate

- Image: `container-library/boltz2:single-load-dev`; no host implementation-code mounts.
- GPU: two Trp-cage/ligand inputs × seeds 11/12; four native predictions, one checkpoint load and one model CPU→GPU transfer.
- All 5,035 named model parameters remained on CUDA with unchanged storage across jobs.
- Recovery: completed-run resume loaded no checkpoint; invalid-input jobs did not stop valid jobs; SIGTERM stopped after two jobs and resume reused those two while completing the remaining two with one new load.
- Scope: RTX 5070 Ti, one sample/seed, three recycles, 200 steps, `--no-kernels`; no 10K endurance claim.

## Cloud transport candidate

- Source: `8a905cb`; local image `container-library/boltz2:cloud-jobs-reviewed`, digest `sha256:f2e06cde658d20154fede1b9a35dfa24ca17701f0a1d6d40fefe998f4731baf0`.
- Installed native/cloud CPU smoke suites passed without source mounts.
- Live R2 → installed GPU worker → R2: Trp-cage with ethanol/methanol × seeds 11/12, one sample each, three recycles, 200 steps, `--no-kernels`; four fresh predictions and one checkpoint load on RTX 5070 Ti.
- Full archive: 96 files, 13,905,920 bytes; independent R2 readback matched every local output hash and native per-job receipts. Archive SHA-256: `e1229e22492cb8650df303bba54188a82e4134411422a85b1bc3f99aa3e5ecb2`.
- Duplicate packaged attempt was refused before worker dispatch. Expiry, subprocess-launch SIGTERM, receipt corruption and credential isolation have focused regression tests; removing each review repair made its test fail.
- Existing verified model assets were mounted read-only. Cloud first-boot downloads, Salad execution/shutdown and larger single-PUT payloads remain untested. No image publication or paid Salad deployment.

## Provider-independent benchmark candidate

- Local image: `container-library/boltz2:2.2.1-worker-v3-benchmark-rc1`, image ID `sha256:b4cdc8b5008af583c119cffde4d9e59a3e4d2c40a2c9951e9a5cfbadbf546553`.
- Code-only offline rebuild from the previously verified dependency image; no dependency upgrades. Installed runtime hashes matched all six source modules. The uncached full Dockerfile required package downloads and was not used for this local qualification.
- Installed native/cloud smoke suites passed. All 19 focused benchmark regressions passed against installed runtime modules; only tests and the host preparation helper were mounted. The fake-worker tests require executable temporary storage.
- Mounted execution passed without network/provider metadata. Final-image live R2 execution passed with explicit local metadata and mounted model assets: ethanol/methanol, seeds 11/12, one sample, three recycles, 200 steps, no kernels, RTX 5070 Ti.
- Final-image comparison against the previous reviewed image: all 24 native prediction files across four predictions were byte-identical, including structures, confidence, embeddings, PAE/PDE and pLDDT.
- Final-image cloud run: one checkpoint load, 31 GPU samples; 105 archived files independently matched local bytes; eight live progress snapshots were independently retrieved from R2. Archive SHA-256: `6ffc29f43feaeacbd60e94070dacb43c4ec8923cce1f6161edce1db0bb72d436`.
- Regression coverage includes actual Salad identifier allowlisting, progress deadline/slow-drip/DNS/connect/kill/reap behavior, bounded shutdown, and interrupted-JSONL tail quarantine/resume. Native timing uses CPU doubles separately from real GPU verification.
- Model server restored and healthy; acceptance container absent. No benchmark-image registry publication or paid cloud execution. New cloud first-boot downloads and large-system performance remain unqualified.
- Evidence: `/archive/cofolding/container-library/boltz2-benchmark-instrumentation/FINAL_ACCEPTANCE.json`.

## Existing batch route
- Fresh packaged examples passed on RTX 5070 Ti: PXR produced five ligand-containing PDB structures and 25 native files; CYP3A4 produced five ligand/heme-containing mmCIF structures and 26 native files.
- `run_test.py` stages canonical entities, the original MSA, and method settings, then invokes `batch.py`.
- The batch native verifier checks consumed sequence/MSA, ligand/cofactor chemistry and atom coverage, finite tensors, native structure coordinates/elements, confidence and PAE/PDE/pLDDT pairing, and the complete native file inventory. Resume re-reads and verifies the run's own outputs.
- `result.json` summarizes the batch verification; `--dry-run` does not produce a PASS report. See the case READMEs for native output inventories.
- Image setup and Trp-cage–ethanol execution/resume checks also passed on RTX 5070 Ti.
- No binding/accuracy claim or registry publication.

## Capture opt-out (`--capture none`)

- Image: `container-library/boltz2:issue6-capture`; one Trp-cage/ethanol input, seed 11, one sample,
  three recycles, 200 steps, `--no-kernels`, RTX 5070 Ti.
- Both modes completed (`run.json` status `completed`) and wrote the same native inventory:
  `trpcage_model_0.cif`, `confidence_trpcage_model_0.json`, `pae/pde/plddt_trpcage_model_0.npz`,
  `embeddings_trpcage.npz`, and the processed inputs — every file byte-identical between modes.
- `--capture none` wrote no `capture/` directory and no `*_inventory.json`; it wrote
  `prediction_summary.pt` (coords, masks, PAE/PDE/pLDDT, pLDDT/IPTM scalars, `s`, `z`) instead.
  Attempt size fell from 3,908 KB to 2,400 KB; `capture/` alone was 1,820 KB of the full attempt.
- `benchmark_metrics.json`: 11 fields identical, 5 timing fields differ (forward/predict-step/output
  write); `capture_hook_write_seconds` is 0.0 with capture off. `predictions.jsonl` reports
  `completed`/seed 11 in both modes.
- Verification difference: with `capture none` the verifier cannot cross-check the consumed batch or
  the raw forward tensors; it still checks structure geometry against the saved coordinates, the
  confidence scalars, the PAE/PDE/pLDDT pairing, the embeddings, the rank bijection and the exact
  native file inventory.

## Observability contract acceptance (RTX 5070 Ti, local image)

Image `container-library/boltz2:obs-smoke` built from this branch, run through `run_cofold.py` on
GPU 1 with Trp-cage/ethanol, seed 11, one sample, three recycles, 200 steps, `--no-kernels`:

- Two inputs in one invocation: the valid one `completed`, the deliberately malformed YAML failed
  with `error_kind: native_error`, one checkpoint load for the whole run.
- The failed attempt wrote `failure.json` with the real diagnosis (YAML `ScannerError`, message and
  line/column, 35-line traceback tail) and its `receipt.json` carries `failure: 'failure.json'`.
- `run.json`, `progress.json` and both `receipt.json` carry `schema: 1`; `progress.json` carries
  `pid` and `updated_at_unix`.
- `--timeout 8` on the same input returned exit `124` and wrote `termination.json`
  (`reason: timeout`, container id present, `last_progress` showing the last counters).
- `--job-timeout 2` returned exit `1` with the job failed as `error_kind: timeout`, a `failure.json`
  reading `job exceeded --job-timeout of 2 seconds`, and no `termination.json`.

CPU: 131 tests pass, 2 skipped, including the new failure-redaction, liveness, schema, OOM-double and
launcher tests.

## Robustness should-fix batch (CPU only, 2026-11-09)

- `cofolding/boltz2/.venv/bin/python scripts/test_local.py --suite cpu`: the Boltz2 package suite
  ran 138 tests, 2 skipped, OK; the shared `tests/` suite ran 37 tests, OK. No GPU, checkpoint,
  image build or network use.
- CLI contract: `tests/test_cli_contract.py` reads `cofolding/cli-contract.json` and checks every
  table flag against the installed `resident.worker` parser (presence, requiredness, enum domain),
  and that every value the worker accepts is also offered by `run_cofold.py`. Unit evidence only.
- Resume inventory: tests cover an extra unrecorded file (reused, `inventory_extras: 1`), a missing
  recorded file (rejected) and a changed recorded file (rejected).
- Protocol: `prepare` now writes `protocol.json` (engine, resolved settings, base seed, seed
  formula, engine entry point); the test asserts the file is in `receipt.json`'s inventory and is
  referenced from the job as `protocol: protocol.json`.
- Image identity: unit tests cover label-first, `COFOLD_IMAGE_ID` second, and null plus
  `image_id_reason` when neither exists; `run.json.implementation.image_id` is asserted for the
  label case.
- Launcher (shared suite): second launch on the same output directory exits 2; preflight rejects a
  disk estimate above free space and `--memory` above host memory with exit 2; the estimated and
  measured values appear under `run.json.preflight`; without `--estimate-attempt-bytes` the free
  bytes are recorded, the comparison is skipped and the launcher says so.
- Not tested here: the Dockerfile image-label mirror is exercised only by an image build, which this
  batch does not perform.

## Derived interface metrics (`--derived`, 2026-11-09)

- CPU: `cofolding/boltz2/.venv/bin/python -m unittest discover -s cofolding/boltz2/tests`
  ran 142 tests, 2 skipped, OK (4 new in `tests/test_derived.py`); the shared `tests/` suite ran
  37 tests, OK. No GPU, checkpoint, build or network use.
- `tests/test_derived.py` uses a fake engine and a synthetic PAE + token map: `metrics/derived.json`
  carries schema 1, the three formula ids, cutoffs 10/15, one pair entry with `ipsae`, `iptm_d0chn`,
  `ipae12/21`, `n0` and `d0`; a missing or malformed token map is a recorded skip with `pairs: null`;
  `--derived none` writes no `metrics/` and records `disabled_by_--derived_none` in the receipt and
  `predictions.jsonl`.
- Image `container-library/boltz2:ipsae-derived-final`, ID
  `sha256:22e0da50d565f9b29af9fd86e7b10e016838387f27fb7cfb203de8f6fa896611`, built with
  `--build-context common=cofolding/common`; `cofolding.common` imports from
  `/opt/cofold/cofolding/common` without torch.
- GPU 1 (RTX 5070 Ti, `GPU-a3d96ffd-0387-fe9f-1c5c-9003ef4fbb47`), Trp-cage/ethanol, seed 11, one
  sample, three recycles, 200 steps, `--no-kernels`, through `run_cofold.py`:
  - `--capture full --derived tensors`: `/home/dan/cofold-runs/ipsae-final2/boltz-full/out`,
    completed, one checkpoint load, `metrics/derived.json` in the receipt inventory; PAE source
    `capture/full_prediction.pt`; token map `tokens.json` (23 tokens: A protein 20, L ligand 3,
    atom names C8/C9/O7) plus an identical `capture/tokens.json`; pair A-L, cutoff 10 ipsae
    0.5616587516, n0 3/19, d0 2.0/1.0.
  - `--capture none --derived tensors`:
    `/home/dan/cofold-runs/ipsae-final2/boltz-none/out`, completed, one load, no `capture/`,
    PAE source `prediction_summary.pt`, same derived values as the capture-full run.
  - `--derived none`: `/home/dan/cofold-runs/ipsae-final2/boltz-derived-none/out`, completed, one
    load, no `metrics/derived.json`, receipt and `predictions.jsonl` say
    `disabled_by_--derived_none`.
- No registry publication; the image is local and the change is uncommitted.
