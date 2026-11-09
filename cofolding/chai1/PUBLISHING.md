# GHCR release

## Status

No Chai-1 image has been built, pushed or published. There is no tag and no digest. The root
README table marks this method as not yet published.

## Prepared workflow

- `.github/workflows/chai1-publish.yml`: manual dispatch only; no automatic publication on push/PR.
- Order: CPU capability tests → image build → installed-image checks → push → pull by registry
  digest → installed-image checks and source-label verification.
- Credentials: built-in `GITHUB_TOKEN`, job-scoped `packages: write`; no PAT/secret required.
- Labels distinguish container-library revision/release from the chai_lab version and from the
  asset revision (`ai.aetherark.chai1.model_revision`).
- No `latest` tag. One explicit tag per release; never overwrite a published tag.
- Hosted CPU checks do not run GPU inference and never download weights; GPU evidence is recorded
  separately in TESTING.md.
- Model artifacts are **not** in the image. `fetch_assets.py` pulls the eight files in
  `assets.json` (about 6.6 GB) into the mounted asset directory and verifies size and sha256.

## GitHub setup

- Merge the reviewed source/workflow into the default branch before manual dispatch appears.
- Repository Actions are enabled; the publish job explicitly requests package writes.
- First publication: package is private by default. Organization Packages → chai1 → Package
  settings → Change visibility → Public. Confirm access is inherited from
  `AetherArk/container-library`; grant that repository Actions access if inheritance is disabled.

## Publish and verify

- Actions → Publish Chai-1 candidate to GHCR → Run workflow → select the reviewed ref and the
  explicit release tag.
- Installed-image checks asserted by the workflow: `cofold-run --help`,
  `cofold-throughput --help`, uid 1000, `WORKDIR /work`, `chai_lab.__version__ == '0.6.1'`,
  `torch.__version__` starts with `2.11.0`, and fail-closed `fetch_assets.py --offline` on an
  empty asset directory.
- The workflow does **not** run `pip check`: chai_lab 0.6.1's metadata pins `torch<2.7`, while the
  image intentionally installs torch 2.11 for `sm_120`. See SPEC.md §2.
- After public visibility is set, verify an anonymous pull with an empty Docker credential
  directory: `DOCKER_CONFIG=$(mktemp -d) docker pull ghcr.io/aetherark/chai1@sha256:...`
- GPU acceptance (one small cell, memory peak, one resume, `--levels 1` throughput) must be
  recorded before the tag is recommended for use.

## Before the first build

1. Confirm `chai_lab 0.6.1` imports and the eight TorchScript artifacts load under torch 2.11 on
   an `sm_120` card.
2. Verify `fetch_assets.py --download` then `--offline` on the same directory.
3. Record the measured memory peak for the packaged examples in TESTING.md.

## 0.6.1-worker-rc1

Digest `sha256:42c9f2ebab68555daaa2b5d5604c42cfc14a6476000c6cdc566dd5895573d08e`, published by workflow run 42617982261 from the merge of
PR #27 (`b3f6ce7`). First publication of this container, and the first to exercise chai_lab 0.6.1
under torch 2.11.0+cu130 with the shared `cofolding/common` build context.

Accepted locally from the published digest on a 16 GB RTX 5070 Ti: pulled anonymously, run through
`run_cofold.py` on the Trp-cage + ethanol example at seed 11, one sample. `completed`;
`run.json.implementation.image_id` reports the published digest from the image label; the job wrote
`metrics/derived.json` with A/B `ipsae` 0.365472, `ipsae21` 0.012205, `n0` 3 / 7 — inside the
run-to-run tolerance recorded in TESTING.md.
