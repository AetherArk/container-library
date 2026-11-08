# Container development requirements

Applies to every method container added under this repository. Both existing containers
(`cofolding/boltz2/`, `cofolding/esmfold2/`) are the reference: read their `SPEC.md`, `README.md`,
`TESTING.md` and `PUBLISHING.md` before starting, and copy that structure rather than inventing one.

The outcome is a container that the existing launcher (`run_cofold.py`), reader and cloud job can
drive exactly like the others, with method differences stated and justified in one place.

## 1. Files a new method delivers

    cofolding/<method>/
      Dockerfile                 # CUDA base, venv, non-root user 1000, ENTRYPOINT cofold-run
      requirements.lock          # pinned full set
      assets.json                # publisher file names, sizes and hashes
      fetch_assets.py            # downloads and verifies weights into a mounted cache
      SPEC.md                    # interface contract plus deviations table
      README.md                  # usage: build, pull, run, options, outputs
      TESTING.md                 # dated evidence, per mode and per device
      PUBLISHING.md              # GHCR release notes and verification order
      CLOUD_JOBS.md              # only if the method gets cloud transport
      resident/cofold-run        # worker entrypoint
      resident/{worker,verify,benchmark}.py plus the engine module and any helper modules
      examples/                  # native YAML inputs
      tests/                     # CPU suite, CLI contract, verifier regressions
    .github/workflows/<method>-publish.yml   # manual dispatch only

Method-specific modules are expected. Do not copy another method's engine code.

## 2. Input contract

- Native format of the underlying engine, unchanged. No repository-specific schema.
- One system per YAML file in an input directory, entities under `sequences`.
- Seeds never appear in the YAML; a `seed`/`seeds` key is rejected. The external seed list applies
  to every input in one invocation.
- Shared cases live in `cofolding/test-cases/` and are reused. A method that cannot support a case
  rejects it with a named rule, and that rejection is a test rather than a silent skip.
- Validate before preparation: entity counts, sequence alphabet, SMILES parsing, ligand slots, MSA
  mode, and anything the engine cannot honor (templates, constraints, affinity, modifications,
  multi-chain). Each rejection names the engine capability, never a generic parse error.
- Referenced files are hash-bound into the job identity.

## 3. Command line

    cofold-run --input DIR --output DIR --assets DIR --checkpoint PATH --seeds 11 12 \
               --samples N --steps N --recycles N [--max-parallel-samples N] \
               [--output-format mmcif|pdb] [--resume] [--metadata JSON] [--capture full|none]

- Flag names and meanings are shared across methods wherever the engine allows it; the same flag
  must mean the same thing in every container.
- Where an engine cannot honor a flag, reject the unsupported value with an explicit message instead
  of accepting and ignoring it. ESMFold2 rejects `--max-parallel-samples` above 1 and
  `--output-format pdb` for that reason.
- Defaults are the engine's own defaults, not tuned values.
- Every resolved setting, including inherited engine defaults, goes into `protocol.json`.

## 4. Seeds

- The launcher passes a base seed list; each entry is one job per input.
- How a base seed becomes a per-sample native seed is engine-specific and is recorded in
  `protocol.json` as a formula. ESMFold2 uses `base_seed*1000 + zero_based_sample_index`; Boltz2
  seeds the process once per job and lets the engine draw its samples.
- Repeated seeds inside one invocation are disambiguated by `seed_occurrence`.
- Determinism claims are engine-specific. Record what is bitwise reproducible and what is not:
  ESMFold2 reproduces PAE bitwise at a fixed seed while coordinates agree only to ~0.02 Å.

## 5. Output layout

Identical across methods so one launcher and one reader serve all of them:

    OUTPUT/
      run.json               inputs, seeds, settings, identity, job outcomes, load count, status
      progress.json          lightweight live status
      benchmark.json         stage timers, hardware, sampled GPU/memory
      predictions.jsonl      append-only per-sample rows
      events.jsonl           append-only structured events
      jobs/<identity-key>/<attempt-id>/
        receipt.json         status, identity, sha256 of every file in the attempt
        capture/             full tensors (see §6)
        native/              engine writer output plus confidence arrays
        inputs/, source/     the exact consumed input and implementation reference
        benchmark_metrics.json
        model_parameters.json (or _before/_after) with resolved model arguments

Per-method additions are allowed and must be named in `SPEC.md`: ESMFold2 writes
`chemistry.json`, `protocol.json` (resolved settings and seed formula) and
`checkpoint-lock.json` next to the files above.

- `predictions.jsonl` keeps the shared schema: `job_key`, `input`, `seed`, `sample_index`,
  `seed_occurrence`, `status`, `verified`, `error_kind`, `output`, `metrics`, `metrics_scope`.
- `metrics_scope` states what the timing covers, for example `input_seed_batch` or
  `input_seed_sample_not_batch`.

## 6. Heavy outputs

- Anything that writes full model tensors beside the deliverable must have an opt-out flag:
  `--capture {full,none}` in the worker CLI, forwarded unchanged by the launcher
  (`run_cofold.py`) and by any batch driver.
- The capture-producing mode stays the default.
- With capture off, write the structures, confidence JSON, per-model arrays, receipts, metrics and a
  small summary packet the verifier can read. The verifier must accept that mode and reject a stray
  capture directory instead of skipping verification.
- Prove the opt-out is output-neutral rather than assuming it: one input and seed in each mode, native
  files byte-identical, only timing fields differing. Then run once with the flag absent to show the
  default still captures. A default value in code is not a run.
- Record both modes and the measured size difference in `TESTING.md`.

## 7. Verification

Verification is an independent reader over the written files, not a trust statement about the engine:

- Structure parses; all coordinates finite.
- Chain set equals the requested entities; ligand presence is mandatory, never an apo fallback.
- Ligand heavy-atom and element counts equal the RDKit reference for the consumed SMILES.
- Protein residue count equals the input sequence length.
- Confidence arrays have the expected shapes and finite values.
- Captured tensors, saved arrays and written structures agree with each other.
- With capture off, state which cross-checks disappear (consumed batch, raw forward tensors) in the
  method README and in `TESTING.md`.

## 8. Identity, resume, exit codes

- Job identity covers input bytes and referenced file hashes, settings, checkpoint, assets,
  implementation and worker version.
- A completed job writes `receipt.json` with a sha256 inventory of the whole attempt.
- `--resume` reuses a completed job only when the identity matches and every hash still matches;
  otherwise the job gets a new attempt directory and the interrupted attempt stays in place.
- A directory checkpoint is hashed once with the worker's own sorted per-file scheme, so the sweep
  and the receipt carry the same digest.
- Exit codes: `0` all requested jobs completed or reused, `1` failure or fatal, `143` stopped at a
  job boundary.

## 9. Instrumentation and failure handling

- Stage timers: identity, preparation, model load, prediction, verification, receipt.
- Resource sampling: CPU, RAM, IO, cgroup, sampled GPU memory and utilization, with explicit
  unavailable reasons when a value cannot be read.
- CUDA OOM is classified across the whole engine call scope, the affected system is skipped, the
  model is not reloaded, and independent systems continue. An OOM on a device is a result to record,
  not a defect to hide.
- One checkpoint load per invocation, across every input × seed × sample; record the load count and
  verify it from two completed jobs.
- A sweep tool rejects settings the engine cannot express instead of measuring the same work twice
  under a different label.

## 10. Tests

- CPU suite with no GPU and no weights: input handling, capability matrix, identity, receipts,
  resume, signals, launcher ownership, OOM classification with doubles.
- CLI contract tests: every shared flag reaches the container argv unchanged, and unsupported
  values are rejected.
- Installed-image checks: entrypoint and imports work from the image alone, as a non-root user.
- GPU acceptance with a real checkpoint: one small system and one system at the intended scale,
  finite coordinates, ligand retained, one resume, capture-mode behaviour, and the device's memory
  peak recorded.
- Model and subprocess doubles are unit tests, not GPU evidence; say which is which in `TESTING.md`.

## 11. Image and assets

- Non-root user 1000, `ENTRYPOINT ["cofold-run"]`, offline at run time, no provider SDK or
  credentials inside the image.
- Labels mirror the existing images: `org.opencontainers.image.*`, plus method source/revision and
  model-revision labels.
- Weights and asset caches stay outside the image; `fetch_assets.py --download` verifies every file
  against `assets.json`. A Hugging Face cache layout is supported and hashed by blob bytes.
- Record the tested image tag and digest in the root `README.md` table and in the method README.
- Audit the image before publication: no host paths, keys, `.env` files or build-machine users.

## 12. Publication

- Every image is built and pushed by the repository's publish workflow (`.github/workflows/<method>-publish.yml`)
  on a GitHub-hosted runner. No package is built or pushed from a developer machine, including as a
  workaround when a workflow step fails.
- A local `docker push` bypasses the workflow's CPU tests and installed-image checks, and GHCR does
  not link a package to the repository for pushes that do not come from that repository's workflow,
  so the package stays out of the repository's package list.
- When the workflow fails at the push step with `permission_denied: write_package`, fix the package's
  Actions access (Organization → Packages → `<package>` → Package settings → Manage Actions access,
  add the repository with Write) or grant the repository connection, then re-dispatch. Do not fall
  back to pushing from a machine.
- Confirm the package is connected to the repository after the first successful workflow publish:
  the repository's Packages section lists it.
- Manual workflow dispatch only; never publish on push or PR.
- One explicit tag per release, named for upstream version and worker build. Never overwrite a
  published tag; no `latest`.
- Order: CPU tests → image build → installed-image checks → push → pull by registry digest →
  installed-image checks and label verification.
- New packages are private by default; visibility is changed by hand, and anonymous pull is
  confirmed after the change.
- Quote the digest, not the tag, when a build must be fixed.

## 13. Cloud transport

- If provided, it is an additive wrapper around the same worker: inputs staged in object storage,
  the same image, the same receipts. The job id carries the method prefix.
- Attempts are single-use; a spent attempt needs a freshly prepared job.
- Verify the returned archive against the receipt before accepting the attempt.

## 14. Documentation split

- `SPEC.md`: interface contract and a deviations table with the engine reason for each difference.
- Method `README.md`: build, pull, run, options, outputs. No general lessons.
- `TESTING.md`: dated evidence, per mode and per device, including what was not tested.
- A lesson that applies beyond one method belongs in `docs/` and is linked from the method README.

## 15. Acceptance checklist

- [ ] Native YAML accepted; unsupported features rejected by name, with tests.
- [ ] Shared flags implemented with shared meanings; unsupported values rejected.
- [ ] Output layout and `predictions.jsonl` schema match the existing containers.
- [ ] Identity, receipt, resume and exit codes as in §8.
- [ ] Capture opt-out implemented and proven output-neutral; default still captures.
- [ ] Verifier independent, with the capture-off cross-check loss documented.
- [ ] One checkpoint load per invocation, observed on real runs.
- [ ] CPU suite and installed-image checks pass; GPU acceptance recorded in `TESTING.md`.
- [ ] `SPEC.md`, `README.md`, `TESTING.md`, `PUBLISHING.md` written; root README table updated.
- [ ] Image audited and published by the repository workflow with an explicit tag; digest recorded.
- [ ] Package connected to the repository and listed under the repository's packages.

## 16. Open questions

Not settled by the current work; decide each one explicitly for a new method.

- The capture opt-out exists in Boltz2. The ESMFold2 container as merged writes `capture/`
  unconditionally, so §6 is a requirement for new and revised containers rather than a description
  of both current images. Whether to retrofit ESMFold2 is undecided.
- Cloud transport exists in both current containers but was never stated as mandatory.
- `cofold-throughput` exists in both; whether every method needs a sweep tool is undecided.
- No numeric thresholds are fixed here: test counts, minimum GPU memory and acceptable timing are
  recorded per method in `TESTING.md`, not as repository-wide gates.
- The two existing attempt layouts are not identical file for file: Boltz2 writes
  `resolved_model_args.json` and `model_parameters.json`/`_after.json`, ESMFold2 adds
  `chemistry.json`, `protocol.json`, `checkpoint-lock.json` and a per-attempt
  `benchmark_failure.json`. Common readers depend on `receipt.json`, `native/` and
  `predictions.jsonl`. Whether new methods should converge on the ESMFold2 superset is undecided.
- GPU acceptance hardware is chosen by measured fit. The CYP3A4 + heme case exhausted a 16 GB card
  and completed on a 32 GB card; that is evidence, not a minimum requirement.
