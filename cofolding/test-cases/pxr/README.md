# PXR–x00242-1

- Original 293-residue PXR construct; prepared ligand charge −1, exact consumed SMILES preserved.
- Original nonempty MSA included, losslessly gzip-compressed.
- Boltz2: seed 11, five samples, three recycles, 200 steps, potentials enabled, PDB output.
- Native outputs: five PDB structures, five confidence JSON files, and five each of PAE/PDE/pLDDT NPZ files (25 files).
- Packaged-example GPU execution and native output verification passed; see [verification](../../boltz2/TESTING.md).
- Ligand source: [OpenADMET PXR dataset](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test), CC BY 4.0; source acid prepared as carboxylate. Model outputs: Boltz MIT.
