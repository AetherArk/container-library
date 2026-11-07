# Boltz2 cloud jobs (R2 transport, explicit deployment)

`cofold-cloud-run` is an **additional command**, not a replacement model or a
Salad API client. It downloads inputs, calls the installed, unchanged `cofold-run`
once for all input × seed jobs, and uploads its complete output tree. Default
image entrypoint remains `cofold-run --help`; local Docker mounts and native
`boltz` remain available.

## Prepare on the trusted host

Install host-only dependencies (the SDK is not added to the image):

```sh
python3 -m venv .venv
.venv/bin/pip install -r cofolding/boltz2/requirements-cloud-host.txt
.venv/bin/python prepare_cloud_job.py \
  --inputs /path/to/dedicated-inputs --output cloud-jobs/new-submission \
  --bucket YOUR_BUCKET --r2-config ~/.r2 \
  --image YOUR_REGISTRY/boltz2@sha256:YOUR_64_HEX_DIGEST \
  --project salad-benchmarking --experiment test \
  --seeds 11 12 --samples 5 --no-kernels --expires-in 86400
```

The digest placeholder must be replaced with the actual immutable image digest.
`--r2-config` defaults to `~/.r2`. It accepts a flat JSON object or non-executable
`KEY=value` lines (optional `export`, quoted values and comments). Required keys:
`AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY` (or `AWS_ACCESS_KEY_ID`),
`AWS_SECRET_ACCESS_KEY`. Configuration is read **only on the host**. Do not paste
credentials into commands, tickets, logs or the node's environment.

- A new job ID is `boltz2-<UTC timestamp>-<unique suffix>`; every preparation
  assigns a fresh attempt ID. Existing local output directories are refused.
- Top-level `.yaml`/`.yml` files and their referenced relative MSA/template
  dependencies are bundled byte-for-byte. Unreferenced files are not uploaded.
  Use a dedicated input tree and keep secrets out of native YAML/dependencies.
- Native YAML is never rewritten. References using `msa`, `cif`, `pdb`, `path`
  must resolve to existing files within the input tree; absolute references,
  symlinks and special files are refused. The resident worker's structure-only
  semantics and external seed list are unchanged. Duplicate seed occurrences
  remain duplicate requested jobs. Sampling/settings flags match `cofold-run`.
- Host uploads the bundle and sanitized `job.json` with create-only writes and
  hashes their exact R2 readback before generating launch capabilities.
- `--expires-in` is 60–604800 seconds. Include image scheduling/pull time, public
  asset download, inference, archive upload **and** hash readback. URLs cannot be
  refreshed on a node; an expired attempt needs a fresh preparation. Expiration
  is rechecked before asset preparation and inference dispatch.

Local output includes:

| File | Contents |
|---|---|
| `job.json` | Nonsecret object keys, image digest, input/bundle checksums, seeds/settings, expiration |
| `inputs.tar.gz` | Unchanged native inputs and dependencies |
| `launch.json` | Sanitized command description; explicitly `deploys: false` |
| `runtime-env.secret.json` | **Private** environment mapping with scoped signed URLs |
| `deployment.secret.json` | **Private** image, full command override and environment fields |

The preparation directory is mode 0700; secret JSON files are exclusively
created mode 0600 and ignored by Git (`*.secret.json`, also `cloud-jobs/`). No
credentials or signed URLs are written to public metadata or printed. Protect
these capability files like passwords; do not use `cat`, shell tracing, public
CI artifacts or `docker inspect` output to share their contents. The deployment
file supplies only the transport/container fields, **not** a complete Salad group
request (GPU/resources, registry authorization and lifecycle are caller-owned).

## Start a new Salad group — separate, potentially billed action

Preparation does **not** deploy or start anything. With separate authorization,
create a group using the prepared image digest, one replica, and a **full command
override**:

```json
{"command": ["cofold-cloud-run"]}
```

Copy `environment_variables` from `deployment.secret.json` through the private
API/UI channel, without logging it. This passes `COFOLD_CLOUD_SPEC`, not the
parent AWS/R2 credentials. Do not supply only arguments to the default
`cofold-run` entrypoint. Set writable disk space for public assets/extraction,
full worker tensors and a second full uncompressed results archive. Allow outbound
HTTPS to R2 and public Hugging Face asset storage. Safe phase/status messages go
to stdout for Salad logs; detailed model/download logs are retained in the result
archive. The wrapper uses the existing
pinned/hash-verified downloader with `--structure-only` (no unused affinity
checkpoint); `HF_HUB_OFFLINE=1` is set only for inference. `HOME`, `TMPDIR` and
library caches are writable under the attempt directory. Worker/download children
receive an environment allowlist excluding parent credentials and signed URLs.

For a packaged local test with mounted, already-prepared assets, use the full
command `['cofold-cloud-run', '--assets', '/models', '--checkpoint', '/checkpoint']`
and provide those mounts; both options are required together. No asset downloads
occur in this mode. It is the caller's responsibility to supply trusted assets;
the unchanged native worker records their identities and verifies scientific
outputs. This mount mode is for local testing, not an automatic Salad mount plan.

## Results and completion

```text
<project>/<experiment>/<job-id>/
  inputs/inputs.tar.gz
  inputs/job.json
  results/<attempt-id>/started.json
  results/<attempt-id>/results.tar
  results/<attempt-id>/receipt.json
```

1. A signed `If-None-Match: *` PUT creates `started.json` **before** downloading or
   running science. Reusing the same payload (restart or second replica) fails
   at the claim, before invoking the worker. Result/receipt PUTs are also
   create-only; there are no overwrite retries.
2. Bundle SHA-256 and the exact per-file inventory must match before dispatch.
   Input extraction rejects absolute/traversal/alias paths, duplicates, links,
   devices, FIFOs, sparse files and extended headers before metadata allocation.
   Host bundles use USTAR (ordinary USTAR filename limits apply). Compressed input
   is limited to 1 GiB, expanded files to 4 GiB/10,000 members, with a 512 MiB
   disk reserve. Archive/downloader failures never launch science.
3. `results.tar` preserves **all** worker files, including native tensor captures,
   native per-job receipts, `run.json`, `progress.json`, logs and partial failed
   outputs, beneath `outputs/`. No fabricated scientific outputs are added.
4. The wrapper PUTs that archive, streams a fresh signed GET into a SHA-256/byte
   count (without storing a duplicate readback), then publishes `receipt.json`
   and verifies its exact stored bytes through a separate GET before returning
   success.
   `status: completed` requires worker exit 0 and terminal native
   `run.json.status: completed`. Worker failure/SIGTERM produces a failed/stopped
   receipt and partial archive when transport is available. Upload/readback
   failure before receipt publication means **no new receipt**. Receipt readback
   failure returns nonzero; the operator must independently validate the stored
   receipt and archive before accepting the attempt.
5. The host/operator must fetch the exact receipt key, match job/attempt/image and
   input bundle identity to public `job.json`, require `status: completed`, then
   independently hash `results.tar` against `result_sha256`/`result_bytes` before
   accepting it. Inspect native `outputs/run.json` and per-job receipts for the
   requested input × seed and sample coverage; transport completion is not a new
   scientific qualification. Download through the trusted R2 client, not by
   publishing signed links. Do not blindly extract untrusted result archives.

Single PUT uploads (no multipart/retry manager) are used, so provider single-object
PUT limits apply. Large output tensors need sufficient disk and expiry; partial
archives remain local after transport failure where the node still exists. A
hard kill, eviction or full disk can prevent any archive/receipt from arriving.
SIGTERM is forwarded to the resident worker, which finishes the current job and
stops at the next boundary; configure enough termination grace for that job and
upload. There is no durable-node recovery or cloud resume in this wrapper.

**The operator must monitor the exact receipt and stop/delete the Salad group
on success, failure or expiry, and verify billing has stopped.** Process exit does
not stop billing or the group's restart policy. Do not leave deterministic
restarts spinning. To retry, prepare a fresh payload/new attempt; never reuse the
old signed URLs. This repository does not implement Salad lifecycle/monitoring.

## Tests (no paid deployment)

```sh
.venv/bin/python scripts/test_local.py --suite cpu
.venv/bin/python scripts/test_local.py --suite image --image YOUR_LOCAL_CANDIDATE
.venv/bin/python scripts/test_local.py --suite image-cloud --image YOUR_LOCAL_CANDIDATE
```

CPU tests use a local HTTP object-store double and an explicitly fake subprocess
worker: real download → dispatch → complete archive PUT → GET/hash → receipt,
failures/collisions/expiry, safe extraction, SIGTERM and credential isolation.
They do **not** run Boltz inference or contact R2. Existing native CPU tests are
retained. Both image smoke suites run installed entrypoints/imports without
network/GPU/source mounts. Real R2 behavior and real packaged GPU output checks
are separate acceptance steps; no Salad deployment is required for either.
