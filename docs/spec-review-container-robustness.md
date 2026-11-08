# Specification review: container robustness and observability

## 1. Scope and what I read

`docs/container-development.md`; both resident implementations and Dockerfiles; `run_cofold.py`; `cofolding/esmfold2/SPEC.md`; both `TESTING.md`; the CLI, lifecycle and launcher tests; the last 25 commits (settled points #7, #10, #11, #15 excluded).

## 2. Verdict

Strong on the scientific core — identity, receipts, resume, single load, independent verification, capture neutrality — and it rightly refuses enterprise process. Thin where the goal is strongest: **where logs go, what a failure looks like afterwards, and how a run ends when interrupted.** Artifacts describe a successful job in detail and a killed one barely at all: `error_label` discards exception messages (`cofolding/boltz2/resident/benchmark.py:52`), `progress.json` has no timestamp, cancellation polls a flag between jobs.

## 3. Findings

### Blocking before a third container

**B1. No log destination or format.** §9 covers instrumentation, not logs. `events.jsonl` lives only inside `OUTPUT`; stdout is inherited from whatever shell ran the launcher, lost on a TTY-less rented box. Cloud invented its own answer (`cofolding/esmfold2/cloud_job.py:557`), so the third container will invent a third. *Rule:* stdout is one line-delimited JSON record per event, identical to the `events.jsonl` row, so `docker logs` or a redirect is a sink; `events.jsonl` stays authoritative; cloud returns `worker.log`; rotation and retention are the operator's business.

**B2. Exit codes are incomplete and already contradicted by the launcher.** §8 gives 0/1/143; `run_cofold.py` returns 2 for an invalid invocation (`:241`), 124 on wall timeout (`:174`), `128+signal` on interrupt (`:176`), and passes 137 through. A batch driver cannot tell "nothing ran" from "everything failed". *Rule:* publish the table — `0` all completed/reused, `1` a job failed or fatal, `2` rejected before any job, `124` wall timeout, `143` stopped at a boundary, `137` killed; `run.json.status` is authoritative.

**B3. No timeout, cancellation, or liveness contract.** SIGTERM only sets a flag (`worker.py:322`) polled between jobs, so a stuck CUDA call is SIGKILLed after `--stop-grace` (`run_cofold.py:159`) with no receipt and no record of the job in flight. `progress.json` has no `updated_at_unix` (boltz `worker.py:204`, esm `worker.py:220`), so a hung run looks alive. *Rule:* `progress.json` gets `updated_at_unix` and `pid`; add `--job-timeout SECONDS` with a `timeout` outcome; on kill the launcher writes `termination.json` (`signal`, `exit_code`, `container_id`, last progress) before returning 137/143.

**B4. Errors are structurally unobservable.** `error_label()` returns `'TypeName: details suppressed'` (`benchmark.py:52-54`) and no traceback is persisted, so a missing MSA, an unparseable SMILES and an OOM reduce to one string. *Rule:* every failed attempt writes a receipt-referenced `failure.json`: `{error_kind, phase, exception_type, message (≤4 KiB, redacted), traceback_tail, attempt_path}`. Redaction governs secrets, not the diagnosis.

**B5. The OOM contract is universal in the spec, absent from the reference container.** §9/§10 require whole-call-scope classification and an OOM test double; Boltz2 has no classification code and `worker.py:270-278` treats every exception alike, unflagged in §16. *Rule:* non-OOM failures continue to the next job; after any failure the worker makes one recovery attempt (drop references, gc, `empty_cache`, reload) and aborts `fatal` if the snapshot or health check differs; remaining jobs are `skipped` with `blocked_by`. Require that test in every CPU suite.

### Should-fix

**S1. No schema versioning** (§5). `SCHEMA = 1` is stamped into JSONL rows and `benchmark.json` (`benchmark.py:17,195`), not `run.json`, `progress.json`, `receipt.json`. *Rule:* integer `schema` everywhere; required keys frozen; readers ignore unknown keys, reject a higher version.

**S2. `metrics_scope` has three spellings** (§5). Spec: `input_seed_batch` / `input_seed_sample_not_batch`. Code: `input_seed_batch_not_individual_sample` (boltz `worker.py:194`, esm `worker.py:210`). `cofolding/esmfold2/SPEC.md:129`: the third. *Rule:* fix the enum to the two code values.

**S3. `--metadata JSON` is unreachable and undefined** (§3). Workers take `--benchmark-metadata PATH` (boltz `worker.py:319`, esm `worker.py:402`); the launcher has no such flag, so nothing forwards it. *Rule:* rename in §3, state the allowlist, forward it.

**S4. The "shared CLI" is not the shared set** (§3, §10). §3 omits `--backend`, `--no-kernels`, `--use-potentials`, `--loops`, metadata; `--output-format` accepts `mmcif|cif` in ESMFold2 (`worker.py:400`) while the launcher allows `mmcif|pdb` (`run_cofold.py:34`), so `cif` is unreachable. "Every shared flag reaches argv unchanged" is unverifiable; each `test_cli_contract.py` hardcodes a list. *Rule:* one `cofolding/cli-contract.json` (flag, required, domain, rejection) read by every container's test.

**S5. GPU exclusivity and single-writer locking are met by the launcher, omitted from the spec** (§9/§10): `run_cofold.py:219-222` takes a per-GPU flock plus an output lock, refusing a busy GPU (`:116-122`); the worker takes `OUTPUT/.worker.lock` (`worker.py:124`). *Rule:* state both; a second worker on one GPU or output dir exits 2.

**S6. No disk or memory preflight** (§9). `--memory` defaults to 36g unchecked; nothing compares free space to attempt size, and a mid-attempt `ENOSPC` leaves an un-receipted directory resume re-runs. *Rule:* preflight memory and free disk ≥ attempt size × jobs, exit 2 before dispatch, record both.

**S7. Retry is never distinguished from resume** (§8, §13). *Rule:* no in-invocation retry; a retry is `--resume`, which creates a new attempt and leaves the failed one in place.

**S8. Secret handling is asserted only for the image** (§11). The metadata allowlist and no-URL rule exist in code (`benchmark.py:22-38`, `cloud_job.py:3`), not the spec. *Rule:* size-bounded allowlist; no environment dumps; no presigned URLs or tokens in artifacts or logs.

**S9. Image identity is optional** (§8). `implementation.image_id` comes only from `COFOLD_IMAGE_ID`, which only the launcher sets; a direct `docker run` records `null`. *Rule:* take the digest from the image label or env var; `null` with a reason otherwise.

**S10. Inventory comparison is over-strict** (§8). `completed()` compares the full dict (`worker.py:100`), so a file added by inspection invalidates reuse; §8 says "every hash still matches". *Rule:* recorded paths present and matching; extras tolerated.

### Nice-to-have

**N1.** Referenced-file binding is a key-name allowlist (`msa`, `cif`, `pdb`, `path`; `worker.py:73-85`), not engine-declared; a novel key escapes identity.
**N2.** `gpu_sample` every 2s makes `events.jsonl` ~100 MB on a long sweep; keep samples in `benchmark.json`.
**N3.** Record attempt size in `receipt.json` so §6's cost claim is queryable.

### Explicitly waive

**W1.** Log rotation, retention, shipping.
**W2.** Retries with backoff — `--resume` is the retry.
**W3.** Metrics backends — the files are the interface.
**W4.** Multi-GPU and in-container parallelism.
**W5.** Secret managers, SBOMs, signing, provenance, alerting.

## 4. Spec versus implementation

1. §9/§10 OOM — none in Boltz2 (`cofolding/boltz2/resident/worker.py:270-278`); §16 does not flag it.
2. §5 `error_kind` — absent in Boltz2 (`worker.py:188-197`), present in ESMFold2 (`worker.py:206`).
3. §3 "every resolved setting in `protocol.json`" — only `cofolding/esmfold2/resident/engine.py:217` writes one, so Boltz2's seed formula (§4) is recorded nowhere.
4. §6/§7 — ESMFold2 has no `--capture` (`worker.py:388-402`), as §16 admits; less visibly `cofolding/esmfold2/resident/verify.py:116-117` requires the capture packet unconditionally, so capture-off verification is proven only by Boltz2.
5. `cofolding/esmfold2/SPEC.md:113-124`, the layout §1 tells new methods to copy, lists `native/prediction.cif` and `capture/full_result.pt`; code writes `native/<rid>_sample_N.cif` and `capture/full_result_sample_N.pt` (`verify.py:69,116`) and omits `chemistry.json`, `protocol.json`, `checkpoint-lock.json`.
6. §3 `--metadata JSON` vs `--benchmark-metadata PATH`, not forwarded (`run_cofold.py:19-37`).
7. §8 exit codes vs `run_cofold.py:174,177,241`.

## 5. What I would not add

No service: no log daemon, metrics endpoint, scheduler, run-state database, or distributed lock. The artifacts are the interface; the only additions worth making are a `progress.json` timestamp and a `failure.json`.

## 6. Not determinable from the repository

- Whether the published Boltz2 digest (`README.md:11`) came from the workflow.
- `--stop-grace` behaviour against a hung CUDA call; no test forces a stuck kernel.
- Whether ESMFold2's cloud `worker.log` was ever inspected for a failure.
- The "worker version" bump rule: hardcoded `1` (`worker.py:168` / `:181`), so an identity-breaking change may not invalidate old receipts.

## 7. Coordinator verification of this review (2026-11-08)

Checked against the code before acting on the findings:

- **B1 is already satisfied and should be withdrawn.** Both containers call
  `Benchmark.append(..., stdout=True)` from `event()`, so stdout already receives one JSON line per
  event, identical to the `events.jsonl` row (`cofolding/boltz2/resident/benchmark.py:209-211`,
  `cofolding/esmfold2/resident/benchmark.py:243-245`). The launcher additionally prints one human
  summary line and the resolved `docker run` command. What remains true is that the spec does not
  say any of this.
- **B3 line number:** SIGTERM handler is `cofolding/esmfold2/resident/worker.py:405`, not 322.
- **B2, B4, B5, S1, S2, S3 confirmed** at the cited locations; `run.json` and `progress.json` carry
  no `schema` key, Boltz2 has no OOM classification, and `metrics_scope` is emitted only as
  `input_seed_batch_not_individual_sample` in both workers.
