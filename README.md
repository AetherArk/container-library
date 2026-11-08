# AetherArk Container Library

- Reusable container recipes, launchers, and focused tests for commonly used workflows.
- In many cases, the underlying code has been optimized for use on NVIDIA Blackwell architecture, particularly consumer-grade RTX50-- hardware.
- Repository code: MIT; bundled upstream and dependency notices retain their own terms.

## Containers

| Method | Image | Digest | Guide |
|---|---|---|---|
| Boltz2 2.2.1 | `ghcr.io/aetherark/boltz2:2.2.1-worker-v3-benchmark-rc1` | `sha256:b2625dfa1e73766cbae90270bcdc4a9f6af5ff9278acf8ad2c99b7a261cdcac8` | [usage](cofolding/boltz2/README.md), [evidence](cofolding/boltz2/TESTING.md) |
| ESMFold2 3.4.1.post1 | `ghcr.io/aetherark/esmfold2:3.4.1.post1-worker-rc3` | `sha256:3cd68646f8af34e209a1a6801f58d8a16d14fbc19771c7c2d7a8893163cb92e6` | [usage](cofolding/esmfold2/README.md), [evidence](cofolding/esmfold2/TESTING.md) |

- Both packages are public; anonymous `docker pull` works. All tags: [boltz2](https://github.com/orgs/AetherArk/packages/container/package/boltz2), [esmfold2](https://github.com/orgs/AetherArk/packages/container/package/esmfold2).
- Model weights and asset caches stay outside the images; each method's `fetch_assets.py --download` populates a cache you mount.
- Tags name the upstream version and worker build; no `latest` tag is published. Quote the digest when a build must be fixed.

Requirements for adding a method container: [docs/container-development.md](docs/container-development.md).

Cloud/provider notes that apply to every container job: [docs/cloud-provider-notes.md](docs/cloud-provider-notes.md).

## Cofolding methods: [cofolding/](cofolding/).
  - Boltz2: [local](cofolding/boltz2/README.md), [cloud input/output preparation](cofolding/boltz2/CLOUD_JOBS.md)
