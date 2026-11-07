# Verification

- CPU suite: 104 Boltz2 tests plus 24 repository tests passed (128 total).

## Local resident-worker candidate

- Image: `container-library/boltz2:single-load-dev`; no host implementation-code mounts.
- GPU: two Trp-cage/ligand inputs × seeds 11/12; four native predictions, one checkpoint load and one model CPU→GPU transfer.
- All 5,035 named model parameters remained on CUDA with unchanged storage across jobs.
- Recovery: completed-run resume loaded no checkpoint; invalid-input jobs did not stop valid jobs; SIGTERM stopped after two jobs and resume reused those two while completing the remaining two with one new load.
- Scope: RTX 5070 Ti, one sample/seed, three recycles, 200 steps, `--no-kernels`; no 10K endurance claim.

## Cloud transport candidate

- Source: `8a905cb`; local image `container-library/boltz2:cloud-jobs-reviewed`, digest `sha256:f2e06cde658d20154fede1b9a35dfa24ca17701f0a1d6d40fefe998f4731baf0`.
- Installed native/cloud CPU smoke suites passed without source mounts.
- Live R2 → installed GPU worker → R2: Trp-cage with ethanol/methanol × seeds 11/12, one sample each, three recycles, 200 steps, `--no-kernels`; four fresh predictions and one checkpoint load on RTX 5070 Ti.
- Full archive: 96 files, 13,905,920 bytes; independent R2 readback matched every local output hash and native per-job receipts. Archive SHA-256: `e1229e22492cb8650df303bba54188a82e4134411422a85b1bc3f99aa3e5ecb2`.
- Duplicate packaged attempt was refused before worker dispatch. Expiry, subprocess-launch SIGTERM, receipt corruption and credential isolation have focused regression tests; removing each review repair made its test fail.
- Existing verified model assets were mounted read-only. Cloud first-boot downloads, Salad execution/shutdown and larger single-PUT payloads remain untested. No image publication or paid Salad deployment.

## Existing batch route
- Fresh packaged examples passed on RTX 5070 Ti: PXR produced five ligand-containing PDB structures and 25 native files; CYP3A4 produced five ligand/heme-containing mmCIF structures and 26 native files.
- `run_test.py` stages canonical entities, the original MSA, and method settings, then invokes `batch.py`.
- The batch native verifier checks consumed sequence/MSA, ligand/cofactor chemistry and atom coverage, finite tensors, native structure coordinates/elements, confidence and PAE/PDE/pLDDT pairing, and the complete native file inventory. Resume re-reads and verifies the run's own outputs.
- `result.json` summarizes the batch verification; `--dry-run` does not produce a PASS report. See the case READMEs for native output inventories.
- Image setup and Trp-cage–ethanol execution/resume checks also passed on RTX 5070 Ti.
- No binding/accuracy claim or registry publication.
