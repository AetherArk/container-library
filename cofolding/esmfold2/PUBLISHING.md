# GHCR release

## Prepared workflow

- `.github/workflows/esmfold2-publish.yml`: manual dispatch only; no automatic publication on push/PR.
- Latest release: `ghcr.io/aetherark/esmfold2:3.4.1.post1-worker-rc3`, digest `sha256:3cd68646f8af34e209a1a6801f58d8a16d14fbc19771c7c2d7a8893163cb92e6`, built and pushed by the repository workflow from commit `279658f` (multi-chain input). Earlier tag `3.4.1.post1-worker-rc1` remains published and is single-chain only. Linux amd64; choose a new explicit tag for every later release; never overwrite a published tag.
- Order: CPU capability tests → image build → installed-image checks → push → pull by registry digest → installed-image checks and source-label verification.
- Credentials: built-in `GITHUB_TOKEN`, job-scoped `packages: write`; no PAT/secret required.
- Labels distinguish container-library revision/release from upstream `esm` revision and from the Hugging Face model revision (`ai.aetherark.esmfold2.model_revision`).
- No `latest` or bare `3.4.1.post1` tag updated. Use the reported digest for exact deployment.
- Hosted CPU checks do not run GPU inference and never download weights; GPU evidence is recorded separately.
- Model weights are **not** in the image. `fetch_assets.py` pulls `biohub/ESMFold2` at revision
  `69869f737beffec5294845ede23db5fc0b4f509e` (11 files, ~24.9 GiB) into the mounted asset
  directory and verifies every file against `assets.json`.
- `--assets` may be a flat directory of verified files, or an existing Hugging Face cache root.
  In the cache case the tool locates `models--biohub--ESMFold2/snapshots/<revision>`, follows the
  snapshot symlinks and hashes the blob bytes; the recorded checksum is still the acceptance rule.
  `--checkpoint` must always be the directory that directly contains `config.json` and `ccd.pkl`,
  which under a cache is that snapshot directory.

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
  (`--assets /models --checkpoint /models/esmfold2`), where `/models/esmfold2` is the directory
  holding `config.json` and `ccd.pkl`.
- Offline asset preparation for a host cache:
  `python cofolding/esmfold2/fetch_assets.py --assets /models --download`, then re-verify with
  `--offline`. Both modes reject size or checksum mismatches and report the path that failed.
- GPU acceptance (one CYP cell and one Trp-cage cell, finite coordinates, ligand retention,
  tensor inventory, one resume, `--levels 1` throughput) has not been run for this tag.
- Registry publication and anonymous-pull verification remain pending until the workflow actually
  runs and visibility is confirmed.

## Release record

- `3.4.1.post1-worker-rc3` — digest `sha256:3cd68646f8af34e209a1a6801f58d8a16d14fbc19771c7c2d7a8893163cb92e6`, source revision `279658fd19720ff7d95f90c5541cfe0824d804f1`,
  published by the repository workflow (run 42504382303). Current recommended tag.
- `3.4.1.post1-worker-rc2` — digest `sha256:b547f416c70be20b4dcb7dcf9728028061d507e73eb3b7100ee27158e231a421`, same source revision, pushed from a developer machine
  before Actions write access existed; superseded by rc3 and kept only for the record.
- Workflow dispatch of `.github/workflows/esmfold2-publish.yml` (run 42499925109) built the image and passed
  the installed-image checks, then failed at the push step with `permission_denied: write_package`: the
  package did not grant the repository's Actions token write access. rc2 was therefore pushed from a
  developer machine, which is not the accepted path — see the rule below and
  `docs/container-development.md` §12.
- The repository now has package Actions write access and the package is connected to it, so the
  workflow path works: run 42504382303 published `3.4.1.post1-worker-rc3`, digest
  `sha256:3cd68646f8af34e209a1a6801f58d8a16d14fbc19771c7c2d7a8893163cb92e6`, from commit `279658f`, and the package now appears under the repository's
  packages. This is the only accepted way to publish.
- Packages are built and pushed by the repository workflow on a GitHub-hosted runner; never from a
  developer machine. A local push skips the workflow checks and does not connect the package to the
  repository, so the package is missing from the repository's package list.
- Published tags are public; `DOCKER_CONFIG=<empty dir> docker pull ghcr.io/aetherark/esmfold2@<digest>`
  and an anonymous manifest request both returned success for rc2.
- `3.4.1.post1-worker-rc4` — digest
  `sha256:1ef048b21711965e72a729d5d39647db3de57911c05a94d0b3b7d13a10b82be3`, published by run
  42530589045 from the observability-contract merge. Adds `failure.json`, `progress.json` liveness
  fields, `schema` on every JSON artifact and `--job-timeout`. Current recommended tag.
