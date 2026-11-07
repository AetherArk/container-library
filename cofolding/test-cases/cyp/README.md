# CYP3A4–OCNT-2395436

- Human CYP3A4, UniProt P08684, 503 residues; source ligand SMILES and FeII-heme preserved.
- Original local MSA included, gzip-compressed without sequence changes.
- Boltz2: seed 11, five samples, three recycles, 200 steps, parallel samples 5.
- Native outputs: five mmCIF structures, five confidence JSON files, five each of PAE/PDE/pLDDT NPZ files, and one embeddings NPZ (26 files).
- Packaged-example GPU execution and native output verification passed; see [verification](../../boltz2/TESTING.md).
- Ligand source: [OpenADMET CYP dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test/tree/146282d2ce1532f1803679a834a9bec9dac1e482), Apache-2.0; model-generated outputs: Boltz MIT.
