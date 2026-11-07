# GHCR release

## Prepared workflow

- `.github/workflows/boltz2-publish.yml`: manual dispatch only; no automatic publication on push/PR.
- Target: `ghcr.io/aetherark/boltz2:2.2.1-worker-v3-benchmark-rc1`; Linux amd64. Preserve the previously published `2.2.1-worker-v1-rc1` image and the reviewed `2.2.1-worker-v2-cloud-rc1` tag/image; never overwrite either.
- Order: CPU tests → image build → installed-image checks → push → pull by registry digest → installed-image checks and source-label verification.
- Credentials: built-in `GITHUB_TOKEN`, job-scoped `packages: write`; no PAT/secret required.
- Labels distinguish container-library revision/release from upstream Boltz revision.
- No `latest` or bare `2.2.1` tag updated. Choose a new explicit tag for each release; use the reported digest for exact deployment.
- Hosted CPU checks do not run GPU inference; local GPU evidence remains in `TESTING.md`.

## GitHub setup

- Merge the reviewed source/workflow into the default branch before manual dispatch appears in Actions.
- Repository Actions are enabled; current read-only token default can remain unchanged because the publish job explicitly requests package writes.
- If organization policy restricts package creation, allow the publishing actor to create the package.
- First publication: package is private by default. Organization Packages → boltz2 → Package settings → Change visibility → Public. Confirm access is inherited from `AetherArk/container-library`; grant that repository Actions access if inheritance is disabled.
- Making the package public permits anonymous downloads. Do not share credentials in chat.

## Publish and verify

- Actions → Publish Boltz2 candidate to GHCR → Run workflow → select the reviewed ref and explicit release tag.
- Successful run summary records `ghcr.io/aetherark/boltz2@sha256:...` and the source revision.
- Authenticated pull/image checks run automatically. After public visibility is set, separately verify anonymous pull with an empty Docker credential directory and rerun `scripts/test_local.py --suite image --image IMAGE@DIGEST`.
- Published use: `docker pull IMAGE@DIGEST`; pass that exact reference to `run_cofold.py --image`.
- Registry publication and anonymous-pull verification remain pending until the workflow actually runs and visibility is confirmed.
