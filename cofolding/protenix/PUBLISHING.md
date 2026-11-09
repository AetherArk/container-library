# GHCR release

## Status

No Protenix image has been built, pushed or published. There is no tag and no digest. The root
README table marks this method as not yet published.

## Prepared workflow

- `.github/workflows/protenix-publish.yml`: manual dispatch only; no automatic publication on
  push/PR.
- Order: CPU capability tests → image build → installed-image checks → push → pull by registry
  digest → installed-image checks and source-label verification.
- Credentials: built-in `GITHUB_TOKEN`, job-scoped `packages: write`; no PAT/secret required.
- Labels distinguish container-library revision/release from the protenix version and from the
  checkpoint revision (`ai.aetherark.protenix.model_revision`).
- No `latest` tag. One explicit tag per release; never overwrite a published tag.
- Hosted CPU checks do not run GPU inference and never download weights; GPU evidence is recorded
  separately in TESTING.md.
- Model artifacts are **not** in the image. `fetch_assets.py` pulls the five files in
  `assets.json` (about 2.6 GB) into the mounted asset directory and verifies size and sha256.

## GitHub setup

- Merge the reviewed source/workflow into the default branch before manual dispatch appears.
- Repository Actions are enabled; the publish job explicitly requests package writes.
- First publication: package is private by default. Organization Packages → protenix → Package
  settings → Change visibility → Public. Confirm access is inherited from
  `AetherArk/container-library`; grant that repository Actions access if inheritance is disabled.

## Publish and verify

- Actions → Publish Protenix candidate to GHCR → Run workflow → select the reviewed ref and the
  explicit release tag.
- Installed-image checks asserted by the workflow: `cofold-run --help`,
  `cofold-throughput --help`, uid 1000, `WORKDIR /work`, `protenix.__version__ == '2.0.0'`,
  `torch.__version__` starts with `2.7.1`, the precompiled fast-layer-norm extension imports, and
  fail-closed `fetch_assets.py --offline` on an empty asset directory.
- After public visibility is set, verify an anonymous pull with an empty Docker credential
  directory: `DOCKER_CONFIG=$(mktemp -d) docker pull ghcr.io/aetherark/protenix@sha256:...`
- GPU acceptance (one small cell, memory peak, one resume, `--levels 1` throughput) must be
  recorded before the tag is recommended for use.

## Before the first build

1. Confirm the image builds on the `nvidia/cuda:12.8.1-devel-ubuntu24.04` base with Python 3.12:
   torch 2.7.1+cu128, the `requirements.lock` freeze and `protenix==2.0.0 --no-deps`.
2. Confirm `build_probe.py` succeeds, in particular that the fast-layer-norm extension compiles at
   build time and imports from site-packages as a non-root user.
3. Confirm `protenix.model.protenix.Protenix(configs)` accepts the checkpoint and that
   `configs.load_strict` is `True` for `protenix-v2` (the released checkpoint matches the v2
   architecture exactly).
4. Verify `fetch_assets.py --download` then `--offline` on the same directory.
5. Record the measured memory peak for the packaged examples in TESTING.md.
