# Verification

## CPU suite (no GPU, no model weights)

Run from this directory:

```sh
/home/dan/swr/miniconda3/envs/cheminf/bin/python -m unittest discover -s tests -v
```

`cheminf` is only an example interpreter: the suite needs `pyyaml`, `numpy`, `rdkit`, `gemmi`
and `torch` (CPU). The container itself has all of them.

| File | Tests | What is exercised |
|---|---|---|
| `tests/test_capability.py` | 17 | YAML entity mapping, heme detection, every rejection rule, packaged examples |
| `tests/test_resident.py` | 21 | lifecycle: one load, seed order, resume, receipts, corruption, fatal stop, SIGTERM, output lock |
| `tests/test_cli_contract.py` | 10 | flag mapping (`--recycles`→loops, `--no-kernels`→reference), rejected settings, shared panel read |
| `tests/test_verify.py` | 9 | independent verifier on synthetic artifacts: ligand loss, truncation, NaN, coordinate mismatch, PAE shape |

Status: 57 tests pass on CPU. `FakeEngine` stands in for the model; no ESMFold2 weights were
loaded and no CUDA call was made.

## Not yet verified

- Real checkpoint inference, ligand retention on generated coordinates, and tensor-to-CIF
  agreement on GPU output.
- Image build of `cofolding/esmfold2/Dockerfile` (dependencies reproduced from the qualified
  `local/esmfold2:3.4.1.post1-cu130` image; the build itself has not been run).
- `cofold-throughput` against this worker (only `--levels 1` is meaningful here).

## GPU acceptance plan (tomorrow)

1. `cofold-run` on `examples/trpcage-ethanol.yaml`, one seed, one sample: expect chains A+L,
   finite coordinates, ligand heavy atoms equal to RDKit ethanol, one checkpoint load.
2. Same on `examples/cyp.yaml`: chains A+L+H, heme element counts match the input SMILES.
3. `--resume`: no reload, all jobs reused.
4. One seed with `--samples 3`: three native calls, seeds `S*1000`, `S*1000+1`, `S*1000+2`.
5. `cofold-throughput sweep --levels 1` over the examples for the benchmark record.
