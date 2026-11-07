# Verification

- CPU suite: 48 tests passed.
- Fresh packaged examples passed on RTX 5070 Ti: PXR produced five ligand-containing PDB structures and 25 native files; CYP3A4 produced five ligand/heme-containing mmCIF structures and 26 native files.
- `run_test.py` stages canonical entities, the original MSA, and method settings, then invokes `batch.py`.
- The batch native verifier checks consumed sequence/MSA, ligand/cofactor chemistry and atom coverage, finite tensors, native structure coordinates/elements, confidence and PAE/PDE/pLDDT pairing, and the complete native file inventory. Resume re-reads and verifies the run's own outputs.
- `result.json` summarizes the batch verification; `--dry-run` does not produce a PASS report. See the case READMEs for native output inventories.
- Image setup and Trp-cage–ethanol execution/resume checks also passed on RTX 5070 Ti.
- No binding/accuracy claim or registry publication.
