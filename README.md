# AetherArk Container Library

- Reusable container recipes, launchers, and focused tests for commonly used workflows.
- In many cases, the underlying code has been optimized for use on NVIDIA Blackwell architecture, particularly consumer-grade RTX50-- hardware.
- Repository code: MIT; bundled upstream and dependency notices retain their own terms.

## Containers

| Method | Image | Digest | Guide |
|---|---|---|---|
| Boltz2 2.2.1 | `ghcr.io/aetherark/boltz2:2.2.1-worker-v4-observability-rc1` | `sha256:9c1c72bbc8f5b7ac7278523ee45c50c351a275579df3158217837a2d34c29320` | [usage](cofolding/boltz2/README.md), [evidence](cofolding/boltz2/TESTING.md) |
| ESMFold2 3.4.1.post1 | `ghcr.io/aetherark/esmfold2:3.4.1.post1-worker-rc4` | `sha256:1ef048b21711965e72a729d5d39647db3de57911c05a94d0b3b7d13a10b82be3` | [usage](cofolding/esmfold2/README.md), [evidence](cofolding/esmfold2/TESTING.md) |

- Both packages are public; anonymous `docker pull` works. All tags: [boltz2](https://github.com/orgs/AetherArk/packages/container/package/boltz2), [esmfold2](https://github.com/orgs/AetherArk/packages/container/package/esmfold2).
- Model weights and asset caches stay outside the images; each method's `fetch_assets.py --download` populates a cache you mount.
- Tags name the upstream version and worker build; no `latest` tag is published. Quote the digest when a build must be fixed.

Requirements for adding a method container: [docs/container-development.md](docs/container-development.md).

Cloud/provider notes that apply to every container job: [docs/cloud-provider-notes.md](docs/cloud-provider-notes.md).

## Cofolding methods: [cofolding/](cofolding/).
  - Boltz2: [local](cofolding/boltz2/README.md), [cloud input/output preparation](cofolding/boltz2/CLOUD_JOBS.md)
