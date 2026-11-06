# Verification

- CPU suite: 46 tests passed.
- Image build, imports, CLI, non-root execution, dependency check, and asset setup passed.
- Trp-cage–ethanol: GPU prediction and unchanged-output resume passed.
- CYP3A4–OCNT-2395436: all 26 native prediction files match the historical run byte-for-byte, including sample ranking.
- PXR–x00242-1: exact historical comparison fails, including with the original image/native command. Two identical native reruns differ in seven files. Repeated nonzero physical-guidance gradients vary in default mode (200/200 repeats); deterministic mode removes that local variation but does not reproduce historical outputs.
- Real runs used RTX 5070 Ti. No binding/accuracy claim or registry publication.
