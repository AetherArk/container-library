# Shared execution test

- Canonical input: `input.json`.
- Protein: 20-residue Trp-cage sequence, chain A.
- Ligand: ethanol (`CCO`), chain L; neutral, no specified stereocentres.
- Seed: 0; one predicted sample; no MSA or structural template.
- Purpose: exercise protein–ligand input, inference, entity retention, finite coordinates, and output readback.
- Arbitrary test pairing; not evidence that ethanol binds Trp-cage or that a predicted pose is accurate.
- Every method uses this one case, translated to its native format.
- Resume checks reuse the same prediction. Invalid-input and corrupted-output tests are software checks, not additional prediction cases.
