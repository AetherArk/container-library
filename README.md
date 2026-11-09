# AetherArk Container Library

- Reusable container recipes, launchers, and focused tests for commonly used workflows.
- In many cases, the underlying code has been optimized for use on NVIDIA Blackwell architecture, particularly consumer-grade RTX50-- hardware.
- Repository code: MIT; bundled upstream and dependency notices retain their own terms.

## Containers

| Method | Image | Digest | Guide |
|---|---|---|---|
| Boltz2 2.2.1 | `ghcr.io/aetherark/boltz2:2.2.1-worker-v5-ipsae-rc1` | `sha256:7821f9c67f4a61d34afad4764f1ca6d070856a01685e825e32cf56ec29dcfc54` | [usage](cofolding/boltz2/README.md), [evidence](cofolding/boltz2/TESTING.md) |
| ESMFold2 3.4.1.post1 | `ghcr.io/aetherark/esmfold2:3.4.1.post1-worker-rc5` | `sha256:2c18d66e843138eb5cfd7c0c7a902f7a5ce344052262ab2183e591f3ea5b037f` | [usage](cofolding/esmfold2/README.md), [evidence](cofolding/esmfold2/TESTING.md) |
| Chai-1 0.6.1 | not published (built and GPU-accepted locally) | — | [usage](cofolding/chai1/README.md), [evidence](cofolding/chai1/TESTING.md) |

- The Boltz2 and ESMFold2 packages are public; anonymous `docker pull` works. All tags: [boltz2](https://github.com/orgs/AetherArk/packages/container/package/boltz2), [esmfold2](https://github.com/orgs/AetherArk/packages/container/package/esmfold2). Chai-1 has no image yet.
- Model weights and asset caches stay outside the images; each method's `fetch_assets.py --download` populates a cache you mount.
- Tags name the upstream version and worker build; no `latest` tag is published. Quote the digest when a build must be fixed.

Requirements for adding a method container: [docs/container-development.md](docs/container-development.md).

Cloud/provider notes that apply to every container job: [docs/cloud-provider-notes.md](docs/cloud-provider-notes.md).

## Cofolding methods: [cofolding/](cofolding/).
  - Boltz2: [local](cofolding/boltz2/README.md), [cloud input/output preparation](cofolding/boltz2/CLOUD_JOBS.md)
