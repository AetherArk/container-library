# Boltz2 benchmark observations (schema 1)

The `2.2.1-worker-v3-benchmark-rc1` candidate adds observations to the existing
resident worker, not another inference implementation. It retains native YAML,
settings, external seed order and duplicate seed occurrences, per-cell seed reset,
BF16, all capture tensors, the native writer and scientific verification. The
checkpoint still loads/transfers once per invocation with pending work, and zero
times for an entirely verified resume. GPU numerical equivalence and packaged
execution must be qualified separately; CPU doubles are not prediction evidence.

## Mounted/local route

The existing `cofold-run --input ... --output ... --assets ... --checkpoint ...
--seeds ...` command needs no provider variables, network or metadata file.
Optional `--benchmark-metadata /path/to/nonsecret.json` is accepted by both worker
and cloud wrapper. This is a local JSON object, at most 16 KiB. Only these keys
are retained:

- `provider`, `region`, `zone`, `machine_type`, `gpu_class`, `node_id`, `currency`;
- numeric finite nonnegative `price_per_hour` (caller assertion, **not** measured cost);
- `SALAD_ORGANIZATION_NAME`, `SALAD_ORGANIZATION_ID`, `SALAD_PROJECT_NAME`,
  `SALAD_PROJECT_ID`, `SALAD_CONTAINER_GROUP_NAME`, `SALAD_CONTAINER_GROUP_ID`,
  `SALAD_MACHINE_ID`, `SALAD_INSTANCE_ID`.

Identifier strings must be short single-line identifiers, not URLs. Unknown keys
are dropped. Do not place secrets under even an allowed key. The cloud wrapper
reads exactly those eight Salad identifiers **before** making the credential-free
child environment, overlays them on optional caller metadata, then writes a local
sanitized `runtime/benchmark-metadata.json` and passes its path as a worker argument.
No parent environment dump, provider auth key, signed URL or metadata credential
variable is passed to the child. Native inputs/output tensors remain scientific
payloads: callers must not put credentials in them.

## Files and identifiers

All records use `schema: 1`. A fresh random `invocation_id` distinguishes each
process invocation, including resume invocations in the same output directory.

| File | Scope and update behavior |
|---|---|
| `benchmark.json` | Atomic startup snapshot, then final summary for the **latest invocation**; not a live aggregate. Hardware/software, optional metadata, stage observations, requested/verified denominator and load count. |
| `events.jsonl` | Append-only, flushed/fsynced structured events, also emitted to stdout. Stage start/finish, device telemetry and terminal events. Earlier resume invocation events remain. |
| `predictions.jsonl` | Append-only, flushed/fsynced rows at cell terminal boundaries. One row per requested sample per input/seed occurrence, including failed, reused and orderly-stop pending rows. Earlier invocations remain. |
| `jobs/.../benchmark_metrics.json` | Native per-cell measurements, included in existing completion receipt/hash inventory. Written before verification/sealing, not modified afterward. |
| `run.json`, `progress.json`, per-cell receipts | Existing lifecycle/verification authority, not replaced by benchmark files. |

The JSONL newline is a record commit boundary. Before the first append to each
file in an invocation, under the observation lock (and existing exclusive worker
ownership lock on resume), any unterminated tail is copied verbatim to a unique
`<filename>.partial-<id>` file, mode 0600. Evidence and its directory entry are
fsynced before removing only that tail from the JSONL stream. Complete historical
newline-terminated records remain byte-for-byte unchanged; new rows remain
independently parseable. Quarantined bytes are **uncommitted evidence**, even if
they happen to parse as JSON: never count them as verified predictions. A failure
to preserve evidence aborts the append rather than discarding it. This repairs
interrupted tails, not arbitrary corruption of earlier complete records.

Join prediction records using invocation ID, job key and sample index. Job keys
include duplicate seed occurrence; `seed_occurrence` and `job_index` make this
explicit. `verified` means the whole cell passed the existing verifier/receipt
path, not scientific accuracy. Failed rows may have partial native outputs; they
are **not** claimed verified. Reused rows point to the original output and contain
no new model timings. `model_was_loaded_before_cell` separates the cell that
triggered load from subsequent warm cells (null when not attempted/reused).

Native `predict_step` may generate multiple samples in one call. Every sample row
explicitly says `metrics_scope: input_seed_batch_not_individual_sample`; its
shared metrics are repeated references to the same cell observation, **not**
independent per-sample measurements. Deduplicate by invocation/job key before
summing times or peaks. No duration is divided by sample count or fabricated for
an unobserved sample. Requested/verified totals include duplicate seed occurrences;
verified totals include receipt-verified reuse, while load/timing counts do not.

## Timer scopes

Times are monotonic `perf_counter` elapsed **wall seconds**; event timestamps are
Unix wall-clock seconds for correlation, not subtraction. Stage rows include
`started_elapsed_seconds`, `seconds` and `status`; stages can nest/overlap and are
not an additive partition of invocation wall time.

Worker stages:

- `identity_verification`: checkpoint and assets hashes; no model load.
- `resume_verification`: existing receipt/inventory verification (also a minimal
  observed stage when resume is off).
- `initialization`: lazy torch/Boltz imports and engine initialization, once only.
- `preparation`: unchanged native parser/featurizer and input staging for a cell.
- `load`: original CPU checkpoint load plus eval/device transfer, synchronized at
  the end; one observed stage, not a per-cell reload.
- `prediction`: full engine predict call, including batch transfer, captures,
  native output writing and cleanup; **not compute-only**.
- `verification`: existing scientific output checks.
- `receipt_verification_and_write`: complete per-cell file hashing and receipt
  write. This can be substantial storage time for full tensors.

Per-cell native metrics:

- `forward_seconds`: one entry per original `model.forward` call. Synchronize
  immediately before starting and after the original forward returns/raises.
  Wrap the callable **inside** module post-hooks; stop this timer **before** the
  full-forward capture hook serializes any tensors. Includes Python/launch and
  synchronization overhead; this is synchronized model-forward wall time, not
  CUDA-event kernel time or storage time.
- `capture_hook_write_seconds`: CPU transfer, serialization, finite/inventory
  checks and writes inside that existing full-forward post-hook, separately.
- `predict_step_seconds`: synchronized whole native `predict_step` call, including
  forward capture-hook writes and native post-forward work. It overlaps forward
  and capture timings; do not add them again. A failed call can still have a
  measured elapsed time; status remains failed.
- `batch_transfer_seconds`: synchronized native input transfer, before predict.
- `output_write_seconds`: consumed-batch/full-prediction packet CPU copies,
  serialization/inventories, native structure/confidence/embedding writer and
  model-argument/parameter JSON. Excludes the separately counted forward hook,
  benchmark JSON's own write, preparation writes and receipt hash/seal time.
- Allocator baseline: allocated/reserved bytes after model load and preparation,
  before input transfer. Reset peak statistics once per attempted native predict;
  report absolute max allocated/reserved bytes across transfer, capture, predict
  and output writing. Baseline includes resident model/cache, is not subtracted
  from peaks, and no allocator cache is emptied for benchmarking.

Cloud wrapper stages: `download_and_input_verification`, `assets`, `worker_total`,
`outputs_archive` (tar creation plus local hash), `upload` (result PUT), and
`upload_verification` (fresh result GET/hash). Mounted assets still have a measured
validation stage, not a fictional download. Worker-local download/upload times
are absent, **not zero**: they do not occur on the mounted route. The final archive
contains `outputs/cloud_transport/` as a **pre-archive/upload snapshot**. The remote
receipt's `benchmark.stages` contains subsequent archive, PUT and GET timing;
`wrapper_wall_seconds_before_receipt` stops before publishing the receipt. A
receipt cannot contain its own eventual PUT/GET duration. Complete local wrapper
wall time and terminal events remain in `work/transport/` and stdout after exit.
Scheduling, image pull and provider billing are outside this process and are not
measured or invented.

## Hardware, telemetry and nulls

Hardware records host CPU model/count and physical RAM, process CPU affinity,
accessible cgroup-v2 `cpu.max`, effective cpuset, memory limit/high/current/peak and
swap limit, plus Python/platform and installed torch/Boltz/NumPy/Lightning versions.
These cgroup strings are observed interfaces (`max` means unlimited at that
interface); they are not a reconstruction of hidden ancestor limits. Host RAM/CPU
are not claimed as the container allocation. Missing cgroup fields have explicit
null reasons. Actual CUDA logical device, name, UUID if exposed by torch, VRAM,
CUDA runtime and cuDNN versions are also captured in native per-cell metrics.

The worker samples `nvidia-smi` initially and about every two seconds, with a
five-second query timeout. Events retain device UUID/name/driver/VRAM, GPU
utilization, used VRAM, temperature, SM/memory clocks and drawn/limit power. Units
are in `benchmark.json.gpu_units`. These are **sampled whole-device** observations
of NVML-visible devices, not per-process GPU utilization, allocator peaks or an
integrated energy measurement. CUDA per-cell identity disambiguates which logical
device executed the model when NVML lists additional visible devices. The wrapper
does not launch a duplicate GPU sampler; it relays safe structured worker events.

Observations use `{value, null_reason}`: a present value has null reason `null`;
missing nvidia-smi gives `devices: null, null_reason: nvidia_smi_missing`, query
failure gives `nvidia_smi_query_failed`, unsupported fields give
`unsupported_or_unavailable`, and absent packages/cgroup fields have named reasons.
Telemetry absence does not prevent inference. Explicit CPU engines/wrapper-only
processes use `not_sampled_in_this_process`. `predict_step_seconds: null` has
`predict_step_null_reason`; an empty forward list has `forward_null_reason`.
Missing native metrics have `metrics_null_reason: native_metrics_unavailable`,
while reused cells use `reused_not_measured`. Timers never substitute zero for
unavailable measurements; zero accumulators mean no completed write time accrued.

## Optional durable cloud progress (no server or provider SDK in image)

On the trusted host, add e.g. `--progress-slots 120 --progress-interval 60` to
`prepare_cloud_job.py`. Defaults are zero slots (disabled) and a 60-second cadence;
limits are 0–256 slots and 10–3600 seconds. Public `job.json.progress_keys` lists
fixed keys under `results/<attempt>/progress/0000.json`, etc. Private launch data
has matching `progress_urls: [{put, get}, ...]` capabilities. Generic callers can
supply these optional capabilities to the same wrapper with any compatible
presigned HTTPS object store; no R2/Salad SDK/auth is required in the image.

The wrapper attempts a first small snapshot immediately, another each interval
while the invocation runs, and a final snapshot before archiving **only if slots
remain**. A snapshot is at most 64 KiB, contains job/attempt/time, observed progress,
hardware/allowlisted metadata, the latest sampled GPU event, latest available
prediction row and recent stage event, not tensors, logs, URLs or resume state.
Files are read in bounded windows; fields may be null before the worker starts or
when no complete recent record is visible. No always-on upload server is needed.

Each PUT is create-only (`If-None-Match: *`) and consumes one slot even on failure;
there are no overwrite retries or unbounded new keys. Only a matching separate
GET SHA-256/byte count produces `progress_upload_verified`. Missing capabilities
are explicitly disabled. Expired, failed or corrupt-readback slots produce
`progress_upload_unverified`, never a durable-success assertion, and do not stop
inference or the final independent result/receipt path. Tiny progress PUT and GET
operations each have a ten-second monotonic **end-to-end deadline**, including
isolated interpreter startup, DNS/connect, response headers and slow-drip reads.
They run in a short-lived stdlib subprocess; capabilities and snapshot bytes go
through stdin, never command-line arguments or logs. Deadline expiry kills and
reaps the process before returning an unverified outcome. Progress child processes
have an empty environment and no file-transfer path arguments.

Shutdown joins the publisher with a finite timeout (22 seconds for an in-flight
PUT plus GET and cleanup); an unexpectedly live writer refuses archival rather
than freezing observations it could still modify. A final available slot can
then take up to another PUT/GET pair of deadlines before the publisher is closed.
No progress background writes continue after the pre-archive snapshot is frozen.
Reserve termination grace for this bounded progress shutdown, plus the unchanged
finish-current-job and final archive/upload behavior. Generic potentially multi-GB
input/result transfers keep socket-inactivity timeouts and existing byte/expiry
checks; they do **not** acquire an arbitrary 120-second total transfer deadline.

The final receipt reports attempted, readback-verified and unverified slots.
After interruption, only snapshots actually stored remotely survive node loss;
operators should GET known public keys and validate job/attempt/slot/timestamp.
A PUT might have stored bytes even when its readback failed: that is **unverified**,
not absent or successful. Exhausted slots, expiry, hard kill before the first PUT,
network loss or eviction can leave no fresh snapshot. A snapshot is never proof of
full native output completion and cannot resume the worker. Full captures/receipts
remain local until the ordinary full archive upload/readback succeeds. Existing
create-only attempt claims, cloud no-resume behavior and explicit group shutdown
responsibility remain unchanged.

Wrapper child logs are redacted in memory before writing: URLs and credential-like
assignments are suppressed and oversized lines discarded. Only selected schema-1
worker events are relayed to stdout; raw native model logs remain private result
logs. Exception text is suppressed in worker metadata and transport errors. This
is not a license to place secrets in native scientific inputs or caller metadata.

## CPU verification

Run `python scripts/test_local.py --suite cpu` from a host environment with the
existing host/test dependencies. Focused `test_benchmark.py` exercises missing
nvidia-smi/unsupported fields, no-provider metadata, repeated seeds/resume/failures,
actual native predict method with CPU doubles and a deterministic clock, hook
synchronization/timer boundaries, log redaction, metadata environment stripping,
periodic bounded HTTP snapshots and failed/expired/corrupt readback. Regressions
also cover the eight real Salad metadata names with secret/unknown exclusions,
completed FakeEngine runs resumed after interrupted event/prediction tails,
large/fragment-only tail quarantine and preservation failure, actual localhost
slow-drip headers/bodies (GET and PUT), isolated blocking DNS/connect doubles,
kill/reap cleanup, bounded publisher shutdown, no late event writes and refusal
to archive with a live publisher. The existing
local HTTP store and fake subprocess remain explicitly test-only; no GPU or cloud
provider is used. Real checkpoint numerical comparison, packaged smoke checks,
mounted GPU execution and live object-store acceptance are separate gates.
