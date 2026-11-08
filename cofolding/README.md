# Cofolding

- [Boltz2](boltz2/)
- [ESMFold2](esmfold2/): same native YAML inputs and job/receipt layout; one protein chain plus one ligand and optional heme.
- Shared [test cases](test-cases/): Trp-cage–ethanol, PXR–ligand, CYP3A4–ligand/heme.
- Each method uses the same case inputs in its native format.
- New methods follow [container development requirements](../docs/container-development.md).
- Weights, caches, and generated outputs stay outside Git.
