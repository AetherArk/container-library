# ESMFold2 cloud jobs (R2 transport, explicit deployment)

`cofold-cloud-run` is an additional command in the image, not a replacement model or a provider
client. It downloads one prepared input bundle, calls the installed, unchanged `cofold-run` once
for all input × seed jobs, and uploads the complete output tree to the prepared R2 prefix. The
default entrypoint remains `cofold-run --help`; local Docker mounts and native `esm` stay usable.

Transport, receipt semantics and safety rules are the same as the [Boltz2 job contract](../boltz2/CLOUD_JOBS.md);
the engine-specific differences are listed at the end.

## Prepare on the trusted host

```sh
python3 -m venv .venv
.venv/bin/pip install -r cofolding/esmfold2/requirements-cloud-host.txt
.venv/bin/python cofolding/esmfold2/prepare_cloud_job.py \
  --inputs /path/to/dedicated-inputs --output cloud-jobs/esmfold2-submission \
  --bucket YOUR_BUCKET --r2-config ~/.r2 \
  --image ghcr.io/aetherark/esmfold2@sha256:YOUR_64_HEX_DIGEST \
  --project esmfold2 --experiment acceptance \
  --seeds 11 --samples 1 --expires-in 21600
```

- Credentials are read only on the host, never written to the node or printed. `--r2-config`
  defaults to `~/.r2` and accepts a flat JSON object or non-executable `KEY=value` lines.
- Local output: `job.json`, `inputs.tar.gz`, `launch.json`, and the private
  `runtime-env.secret.json` / `deployment.secret.json` (mode 0600, git-ignored).
- Expiration covers image pull, the ~25 GiB weight download, inference, archive upload and
  readback. An expired attempt needs a fresh preparation.

## Run on the node

```sh
docker run --rm --gpus all \
  --env COFOLD_CLOUD_SPEC="$(python3 -c 'import json;print(json.load(open("runtime-env.secret.json"))["COFOLD_CLOUD_SPEC"])')" \
  ghcr.io/aetherark/esmfold2@sha256:YOUR_64_HEX_DIGEST cofold-cloud-run
```

- Only the presigned capabilities reach the node; R2 credentials stay on the host.
- `--work` defaults to `/work/cloud-attempt`; weights land in `<work>/assets` and are verified
  file by file against `assets.json` before inference.
- Local mount mode for testing: `cofold-cloud-run --assets /models --checkpoint /models`, where
  `/models` is the flat directory produced by `fetch_assets.py --materialize`.

## Results

```text
<project>/<experiment>/<job-id>/
  inputs/inputs.tar.gz
  inputs/job.json
  results/<attempt-id>/started.json
  results/<attempt-id>/results.tar
  results/<attempt-id>/receipt.json
```

`results.tar` holds everything under `outputs/`: native mmCIF, confidence JSON, PAE/pLDDT/embedding
arrays, complete tensor captures, per-job receipts, `run.json`, `progress.json`, `benchmark.json`
and both JSONL logs. The wrapper verifies its own upload by streaming a readback before publishing
the receipt. Completion requires worker exit 0 and native `run.json.status: completed`.

## Differences from the Boltz2 contract

| Area | Boltz2 | ESMFold2 |
|---|---|---|
| Checkpoint | single `boltz2_conf.ckpt` file | model directory (`config.json`, `ccd.pkl`, six safetensors shards) |
| Asset preparation | `fetch_assets.py --download --structure-only` | `fetch_assets.py --download`, ~24.9 GiB |
| Settings sent to the worker | `--samples --steps --max-parallel-samples --recycles --use-potentials --no-kernels --output-format` | `--samples --steps --max-parallel-samples --recycles (num_loops) --backend --output-format` |
| Rejected settings | — | `max_parallel_samples != 1`, `backend` outside `fused/reference`, `output_format != mmcif` |
| Job id prefix | `boltz2-` | `esmfold2-` |

## Provider pitfalls learned on Vast.ai

- `--env` takes docker syntax: `-e KEY=VALUE`. A bare `KEY=VALUE` is parsed as a port mapping and
  the worker starts with no spec, failing with "cloud attempt failed".
- An attempt is single-use. The create-only claim (`started.json`) is written before the assets
  stage, so any aborted run spends the attempt; `prepare` a new job instead of rerunning the old one.
- Image pull dominates first-run time. On a host that pulled the 24 GB image in 214-366 s the whole
  job finished in ~6 minutes; one 17 Gbps host never finished the pull in 32 minutes. Probe a host
  for a bounded pull window and destroy it when the workload has not started.
- Rent in args/entrypoint mode (`--entrypoint /opt/cofold/cofold-cloud-run`); the ssh/jupyter mode
  injects sshd into an image that has none.

## Operator duties

- Monitor the exact receipt; stop the paid instance on success, failure or expiry and verify
  billing stopped. Process exit does not stop a provider's billing.
- Fetch `results.tar` through the trusted R2 client, check `result_sha256` and `result_bytes`,
  then inspect native `run.json` and receipts before accepting the attempt.
- Transport completion is not scientific acceptance. The GPU evidence recorded in `TESTING.md`
  applies to the image digest named there.
