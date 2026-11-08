# GHCR release

## Prepared workflow

- `.github/workflows/esmfold2-publish.yml`: manual dispatch only; no automatic publication on push/PR.
- Target: `ghcr.io/aetherark/esmfold2:3.4.1.post1-worker-rc1`; Linux amd64. Choose a new explicit tag for every later release; never overwrite a published tag.
- Order: CPU capability tests → image build → installed-image checks → push → pull by registry digest → installed-image checks and source-label verification.
- Credentials: built-in `GITHUB_TOKEN`, job-scoped `packages: write`; no PAT/secret required.
- Labels distinguish container-library revision/release from upstream `esm` revision and from the Hugging Face model revision (`ai.aetherark.esmfold2.model_revision`).
- No `latest` or bare `3.4.1.post1` tag updated. Use the reported digest for exact deployment.
- Hosted CPU checks do not run GPU inference and never download weights; GPU evidence is recorded separately.
- Model weights are **not** in the image. `fetch_assets.py` pulls `biohub/ESMFold2` at revision
  `69869f737beffec5294845ede23db5fc0b4f509e` (11 files, ~24.9 GiB) into the mounted asset
  directory and verifies every file against `assets.json`.

## GitHub setup

- Merge the reviewed source/workflow into the default branch before manual dispatch appears in Actions.
- Repository Actions are enabled; the current read-only token default can remain unchanged because the publish job explicitly requests package writes.
- If organization policy restricts package creation, allow the publishing actor to create the package.
- First publication: package is private by default. Organization Packages → esmfold2 → Package settings → Change visibility → Public. Confirm access is inherited from `AetherArk/container-library`; grant that repository Actions access if inheritance is disabled.
- Making the package public permits anonymous downloads. Do not share credentials in chat.

## Publish and verify

- Actions → Publish ESMFold2 candidate to GHCR → Run workflow → select the reviewed ref and the explicit release tag.
- The successful run summary records `ghcr.io/aetherark/esmfold2@sha256:...` and the source revision.
- Installed-image checks asserted by the workflow: `cofold-run --help`, `cofold-throughput --help`,
  uid 1000, `WORKDIR /work`, pinned `esm`/`torch`/`transformer-engine`/`transformers` versions,
  and fail-closed `fetch_assets.py --offline` on an empty asset directory.
- After public visibility is set, separately verify an anonymous pull with an empty Docker credential directory:
  `DOCKER_CONFIG=$(mktemp -d) docker pull ghcr.io/aetherark/esmfold2@sha256:...`
- Published use: `docker pull IMAGE@DIGEST`; pass that exact reference as the cofolding image.
  Mount the verified asset directory and pass `--assets` and `--checkpoint` explicitly
  (`--assets /models --checkpoint /models/esmfold2`).
- Offline asset preparation for a host cache:
  `python cofolding/esmfold2/fetch_assets.py --assets /models --download`, then re-verify with
  `--offline`. Both modes refuse symlinked or size/checksum-mismatched files.
- GPU acceptance (one CYP cell and one Trp-cage cell, finite coordinates, ligand retention,
  tensor inventory, one resume, `--levels 1` throughput) has not been run for this tag.
- Registry publication and anonymous-pull verification remain pending until the workflow actually
  runs and visibility is confirmed.
