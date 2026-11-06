# Cofolding containers

- [Boltz2](boltz2/): first packaged method.
- One shared execution test: [test-case/input.json](test-case/input.json).
- Other methods will use the same protein sequence, ligand, seed, and sample count in their native formats.
- Inference settings remain method-specific and must be recorded.
- Model weights, alignment files, caches, and prediction outputs stay outside Git.
- CYP-specific protocols and results stay in the CYP project.
