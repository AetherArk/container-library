# Verification

## CPU suite (no GPU, no model weights)

Run from this directory:

```sh
/home/dan/swr/miniconda3/envs/cheminf/bin/python -m unittest discover -s tests -v
```

`cheminf` is only an example interpreter: the suite needs `pyyaml`, `numpy`, `rdkit`, `gemmi`
and `torch` (CPU). The container itself has all of them.

The same 68 tests pass inside an image built from this directory (entrypoint override, code
mounted read-only, no network):

```sh
docker run --rm --network=none --user 1000:1000 --entrypoint python \
  --tmpfs /tmp:rw,exec,mode=1777,size=2g --env TMPDIR=/tmp --env HOME=/tmp \
  --mount type=bind,src="$PWD",dst=/pkg,readonly --workdir /pkg \
  container-library/esmfold2:3.4.1.post1-worker-rc1 -m unittest discover -s /pkg/tests
```

| File | Tests | What is exercised |
|---|---|---|
| `tests/test_capability.py` | 17 | YAML entity mapping, heme detection, every rejection rule, packaged examples |
| `tests/test_resident.py` | 21 | lifecycle: one load, seed order, resume, receipts, corruption, fatal stop, SIGTERM, output lock |
| `tests/test_cli_contract.py` | 10 | flag mapping (`--recycles`→loops, `--no-kernels`→reference), rejected settings, shared panel read |
| `tests/test_verify.py` | 9 | independent verifier on synthetic artifacts: ligand loss, truncation, NaN, coordinate mismatch, PAE shape |
| `tests/test_sweep_contract.py` | 7 | directory checkpoint hashing, engine defaults, level/`--use-potentials` rejection, worker argv |

Status: 68 tests pass on CPU. `FakeEngine` stands in for the model; no ESMFold2 weights were
loaded and no CUDA call was made.

## Not yet verified

- Real checkpoint inference, ligand retention on generated coordinates, and tensor-to-CIF
  agreement on GPU output.
- Image build of `cofolding/esmfold2/Dockerfile` (dependencies reproduced from the qualified
  `local/esmfold2:3.4.1.post1-cu130` image; the build itself has not been run).
- `cofold-throughput sweep` end to end against a real GPU worker; its argument and hashing
  contract is covered on CPU, but no sweep has executed model inference.

## GPU acceptance plan (tomorrow)

1. `cofold-run` on `examples/trpcage-ethanol.yaml`, one seed, one sample: expect chains A+L,
   finite coordinates, ligand heavy atoms equal to RDKit ethanol, one checkpoint load.
2. Same on `examples/cyp.yaml`: chains A+L+H, heme element counts match the input SMILES.
3. `--resume`: no reload, all jobs reused.
4. One seed with `--samples 3`: three native calls, seeds `S*1000`, `S*1000+1`, `S*1000+2`.
5. `cofold-throughput sweep --levels 1` over the examples for the benchmark record.
