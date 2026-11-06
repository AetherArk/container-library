# Tested scope

- CPU suite: 31 tests passed with Python 3.12 / PyYAML 6.0.2.
- Image: clean recipe build passed; native CLI help, Boltz2 model import, non-root UID 1000, and `pip check` passed.
- GPU: shared Trp-cage/ethanol case, seed 0, one sample; 3 recycling steps, 200 sampling steps, `--no_kernels`; RTX 5070 Ti.
- One checkpoint load; one prediction passed independent saved-output readback.
- Checks: protein sequence, ligand graph/charges, entity/atom coverage, finite tensors, native CIF/tensor coordinate pairing, confidence-array pairing, output inventory.
- Identical resume passed without another inference; receipt/payload hashes remained unchanged.
- Compose configuration validated; Compose native inference was not separately run.
- Local-MSA path staging is CPU-tested; this release's GPU test uses no MSA.
- Multi-input/seed dispatch is CPU-tested here; this release uses one shared molecular case and one seed for GPU acceptance.
- No experimental binding or structural-accuracy claim; no numerical equivalence claim to the historical CYP campaign.
- No registry image publication is included in this source release.
