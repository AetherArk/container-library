# PXR–x00242-1

- Original 293-residue PXR construct; prepared ligand charge −1, exact consumed SMILES preserved.
- Nonempty historical MSA included; prepared-top-state check passed.
- Boltz2: seed 11, five samples, three recycles, 200 steps, potentials enabled, PDB output.
- Baseline: `structure184_boltz2_msa_seed11_preparedtop_20260719T114922Z/smoke/seed_11_x00242_preparedtop_d5`.
- Expected checksums cover all 25 surviving native prediction files; exact replay currently fails (see [verification](../../boltz2/TESTING.md)).
- Released checkpoint is pinned here; its checksum was not recorded by the historical launcher.
- Ligand source: [OpenADMET PXR dataset](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test), CC BY 4.0; source acid prepared as carboxylate. Model outputs: Boltz MIT.
