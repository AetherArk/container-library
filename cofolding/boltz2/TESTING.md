# Verification

- CPU suite: 69 Boltz tests plus 17 launcher/harness tests passed.

## Local resident-worker candidate

- Image: `container-library/boltz2:single-load-dev`; no host implementation-code mounts.
- GPU: two Trp-cage/ligand inputs × seeds 11/12; four native predictions, one checkpoint load and one model CPU→GPU transfer.
- All 5,035 named model parameters remained on CUDA with unchanged storage across jobs.
- Recovery: completed-run resume loaded no checkpoint; invalid-input jobs did not stop valid jobs; SIGTERM stopped after two jobs and resume reused those two while completing the remaining two with one new load.
- Scope: RTX 5070 Ti, one sample/seed, three recycles, 200 steps, `--no-kernels`; no 10K endurance claim.

## Existing batch route
- Fresh packaged examples passed on RTX 5070 Ti: PXR produced five ligand-containing PDB structures and 25 native files; CYP3A4 produced five ligand/heme-containing mmCIF structures and 26 native files.
- `run_test.py` stages canonical entities, the original MSA, and method settings, then invokes `batch.py`.
- The batch native verifier checks consumed sequence/MSA, ligand/cofactor chemistry and atom coverage, finite tensors, native structure coordinates/elements, confidence and PAE/PDE/pLDDT pairing, and the complete native file inventory. Resume re-reads and verifies the run's own outputs.
- `result.json` summarizes the batch verification; `--dry-run` does not produce a PASS report. See the case READMEs for native output inventories.
- Image setup and Trp-cage–ethanol execution/resume checks also passed on RTX 5070 Ti.
- No binding/accuracy claim or registry publication.
