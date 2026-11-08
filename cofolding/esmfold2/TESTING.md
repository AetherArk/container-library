# Verification

## CPU suite (no GPU, no model weights)

Run from this directory:

```sh
python -m unittest discover -s tests -v  # any Python with the package deps
```

`cheminf` is only an example interpreter: the suite needs `pyyaml`, `numpy`, `rdkit`, `gemmi`
and `torch` (CPU). The container itself has all of them.

The same 75 tests pass inside an image built from this directory (one is skipped without the `esm` package) (entrypoint override, code
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
| `tests/test_assets.py` | 5 | flat and Hugging Face cache layouts, corrupted blob behind a symlink, absent directory |
| `tests/test_protocol.py` | 2 | `fold()` override list vs inherited native defaults; `protocol.json` carries the call record |

Status: 75 tests pass on CPU (74 run locally, one skips without `esm`). `FakeEngine` stands in for the model; no ESMFold2 weights were
loaded and no CUDA call was made.

## GPU results (2026-11-08, RTX 5070 Ti 16 GB, revision 69869f73)

- Trp-cage + ethanol, one seed, one sample: completed. Chains A (20 residues, `NLYIQWLKDGGPSSGRPPPS`)
  and L (LIG, 3 atoms C/C/O); all coordinates finite; pLDDT mean 0.926; PAE 23x23; one model load
  (7.7 s) and 3.5 s per native call.
- CYP3A4 + ligand + heme, 200 steps or 20 steps: `cuda_out_of_memory` in both cases, peak allocated
  15.6 GB of 16.6 GB. The 13.8 GB of weights leave ~2.8 GB, so the limit is card capacity, not the
  sampling schedule. This system needs a 24 GB or larger card.
- Determinism at a fixed seed: PAE bitwise identical; coordinates within 0.02 Angstrom but not
  bitwise identical (fused, reference, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` all tested).
- A failed system does not disturb the others: with the two examples in one invocation, Trp-cage
  completed and CYP failed, with `run.json` status `failed` and one checkpoint load in total.
- `--resume` on a completed output directory: second invocation reported
  `checkpoint_load_count: 0`, the job was `reused: true` and independently verified. Reuse is keyed
  by the model image id as well, so results produced by different image builds are never reused
  (observed when a rebuild changed the id between two otherwise identical runs).
- `--samples 3` with `--seeds 11`: three native calls with seeds 11000, 11001, 11002 and
  pLDDT 0.927, 0.925, 0.928.

## Not yet verified

- Real checkpoint inference, ligand retention on generated coordinates, and tensor-to-CIF
  agreement on GPU output.
- Image build of `cofolding/esmfold2/Dockerfile` (dependencies reproduced from the qualified
  `local/esmfold2:3.4.1.post1-cu130` image; the build itself has not been run).
- `cofold-throughput sweep` end to end against a real GPU worker; its argument and hashing
  contract is covered on CPU, but no sweep has executed model inference.

## GPU acceptance plan (tomorrow)

1. `cofold-run` on `examples/trpcage-ethanol.yaml`, one seed, one sample: done, results above.
2. Same on `examples/cyp.yaml`: chains A+L+H, heme element counts match the input SMILES.
   Blocked on hardware: 16 GB is insufficient.
3. `--resume`: no reload, all jobs reused.
4. One seed with `--samples 3`: three native calls, seeds `S*1000`, `S*1000+1`, `S*1000+2`.
5. `cofold-throughput sweep --levels 1` over the examples for the benchmark record.

## Cloud acceptance on a 32 GB GPU (Vast.ai, RTX 5090)

- Image `ghcr.io/aetherark/esmfold2@sha256:6776870886482fb2b53dadc55005799c0c518cbbde026d10a30d7fcb7954fb6d`,
  pulled anonymously from GHCR in 214 s; instance `59724992` (offer 59535046), $0.96/h.
- Job `esmfold2-20261108T144730Z-b1a75916c6bf`, attempt `attempt-0312633798c446988a1c57b91c13db0e`.
- Inputs downloaded and hash-verified in 0.3 s; assets (24.9 GiB) in 304.5 s (~92 MB/s); worker 57.2 s.
- CYP3A4 + heme + ligand and Trp-cage/ethanol both completed on seed 11, one sample, 200 steps,
  20 loops, `fused` backend; one checkpoint load, one CPU→GPU transfer.
- CYP forward 25.0 s; peak allocator 19.14 GB allocated / 20.62 GB reserved of 33.71 GB total. The same
  input failed on a 16 GB RTX 5070 Ti at 14.5 GB, so this is the first GPU that completes it.
- Archive 424,345,600 bytes, sha256 `5fc943ef1ae895b40146d8c3692619bd1268c7a2aa6dde8f2d5bd63423c1a63e`;
  `run.json` status `completed`, both `predictions.jsonl` entries `completed`.

## Multi-chain input (RTX 5070 Ti, local image)

- `examples/two-chain.yaml`: chains `A` (Trp-cage) and `B` (20-residue synthetic), ethanol ligand;
  seed 11, one sample, default steps/loops, `fused` backend.
- Image `container-library/esmfold2:multichain`; assets and checkpoint mounted read-only from the
  local flat cache; non-root container user.
- `run.json` status `completed`; the written CIF carries chains `A`, `B`, `L`; 40 protein residues
  and 43 confidence tokens (40 residues + 3 ligand heavy atoms); PAE 43×43, pLDDT length 43.
- Forward 5.04 s; peak allocator reserved 13.14 GiB; receipt lists 20 hashed files.
- Verification passed the independent reader: chain set, per-chain residue counts, total residue
  count, ligand heavy-atom and element counts, PAE/pLDDT shapes and the captured-coordinate
  comparison against the CIF.
- Chain ids in the written CIF equal the input ids; duplicate protein ids, and protein ids that
  collide with ligand ids, are rejected before preparation (unit tests).
- Not tested: chains beyond two, identical sequences on several chains, and a two-chain input on
  the 32 GB cloud route.
