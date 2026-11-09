# Verification

## CPU suite (no GPU, no model weights)

Run from the repository root:

```sh
/home/dan/swr/miniconda3/envs/cheminf/bin/python -m unittest discover -s cofolding/chai1/tests
```

The suite needs `pyyaml`, `numpy`, `rdkit`, `gemmi` and `torch` (CPU). It never imports
`chai_lab`, never opens the network and never reads a weight file.

| File | Tests | What is exercised |
|---|---|---|
| `tests/test_capability.py` | 18 | YAML entity mapping, multi-ligand and apo acceptance, heme as a ligand, every rejection rule, FASTA text, chain letters, packaged examples |
| `tests/test_resident.py` | 28 | lifecycle: one load, seed order, resume, receipts, corruption, fatal stop, SIGTERM, output lock |
| `tests/test_cli_contract.py` | 14 | shared-flag table for `chai1`, defaults (`recycles=3`, `low_memory=True`), rejected engine-specific flags, settings validation |
| `tests/test_verify.py` | 14 | independent verifier on synthetic artifacts: ligand loss, element mismatch, truncation, NaN, PAE/PDE/pLDDT shapes, chain set |
| `tests/test_derived.py` | 6 | engine `derive()` against a synthetic PAE + token map, malformed map skip, worker forwarding of `--derived none`, receipt fields |
| `tests/test_token_map.py` | 9 | token map from collated-feature arrays, JSON round-trip through the shared reader, PAE/entity/residue/ligand refusal cases |
| `tests/test_protocol.py` | 3 | fold-call override list vs inherited defaults, `protocol.json`, generated FASTA, seed formula |
| `tests/test_assets.py` | 6 | pinned flat layout, corrupted bytes, same-size mismatch, missing file, unpinned sha256, absent directory |
| `tests/test_sweep_contract.py` | 9 | directory checkpoint digest, engine defaults, level/`--no-kernels`/`--use-potentials` refusal, worker argv |
| `tests/test_cloud.py` | 6 | cloud preparation against an in-memory object store, input bundle contents, private launch files, settings validation, worker command |

Status: 113 tests pass on CPU. `FakeEngine` stands in for the model; no Chai-1 weights were
loaded and no CUDA call was made.

## What is verified

- The native YAML subset accepted and rejected by the container, including the exact message for
  each refusal.
- Resident lifecycle: one `checkpoint_load_count`, external seed order, resume only on matching
  identity and hashes, fatal stop on model drift, SIGTERM at a job boundary.
- The token map is derived from tokenized context arrays and refuses PAE size, entity count,
  protein residue count, ligand heavy-atom count and non-contiguous chain mismatches.
- `metrics/derived.json` is produced from a synthetic PAE + token map through the shared
  `cofolding.common` module; `--derived none` writes nothing and records why.
- The independent verifier rejects ligand loss, element mismatch, truncation, NaN coordinates
  and wrong array shapes.
- Shared CLI contract: every flag marked `supported` or `rejected` for `chai1` in
  `cofolding/cli-contract.json` is checked against the installed parser.
- Cloud preparation packages only referenced inputs and writes 0600 launch files.
- Importing `resident.worker`, `resident.engine`, `resident.verify` and the shared derived module
  requires no GPU, no `chai_lab` and no weights.

## What is not verified

- **Image build.** The Dockerfile was not built. `torch 2.11` overriding chai_lab's `<2.7`
  metadata bound is recorded in SPEC.md; TorchScript module loading under torch 2.11 is untested.
- **GPU inference.** No checkpoint was downloaded through the container and no job was run.
  Memory fit on the 16 GB RTX 5070 Ti, numeric output, ligand retention on generated coordinates
  and tensor-to-CIF agreement are all unverified.
- **Determinism.** Not measured for Chai-1. No bitwise or tolerance claim is made.
- **Throughput sweep end to end.** Only its argument, digest and argv contract is on CPU.
- **Published image.** None exists; no digest, no anonymous pull, no installed-image check.

## GPU acceptance plan

1. Build the image with `--build-context common=cofolding/common`; run the CPU suite inside it.
2. `cofold-run` on `examples/trpcage-ethanol.yaml`, one seed, one sample; record peak memory.
3. Same on `examples/two-chain.yaml` and `examples/small-two-ligand.yaml`; check chain map and
   ligand element counts.
4. `--samples 2` with one seed: seeds `S*1000` and `S*1000+1`.
5. `--resume`: `checkpoint_load_count: 0`, jobs reused and independently verified.
6. `cofold-throughput sweep --levels 1` for the benchmark record.

## GPU acceptance on a 16 GB RTX 5070 Ti (sm_120)

Local image built from this tree with `--build-context common=cofolding/common`
(`container-library/chai1:test3`); torch 2.11.0+cu130, chai_lab 0.6.1; assets pre-populated at
`/nvme1/dan/folding/cache/chai1-assets` and mounted read-only; every run through `run_cofold.py`
on GPU 1 with seed 11, one sample, three recycles, 200 steps.

- **Trp-cage + ethanol**: `completed`, 44.0 s. `native/` holds the CIF, PAE, PDE, pLDDT, the token map
  (43 tokens: 40 protein residues, 3 ligand atoms `C1`/`C2`/`O1`) and the raw `chai_sample_1/` files.
  `metrics/derived.json` records the A/B pair at cutoff 10: `ipsae` 0.364598, `ipsae12` 0.364598,
  `ipsae21` 0.012211, `n0` 3 / 7.
- **Reference cross-check** on that run's own CIF and PAE (IPSAE2 0.1.10 `df6e46f2`): A/B `ipsae`
  0.364598, `ipsae12` 0.364598, `ipsae21` 0.012211, `n0` 3 / 7 — identical. On the homodimer run the
  B/C pair is 0.039059 in both container and reference, and A/C 0.022608 appears only in the
  container because the reference filters pairs with no 5 Angstrom heavy-atom contact.
- **Homodimer example**: `completed`, 80.2 s. Both directions of A/B score 0.0 with `n0 = 0`, and the
  reference reproduces that exactly: the PAE between the chains has a minimum of 13.25 Angstrom, so
  the model places each chain confidently but is not confident about their relative arrangement. The
  two chains are 4.97 Angstrom apart at the closest CA-CA contact, so this is a confidence property
  of the prediction, not a scoring fault. The example is therefore a poor interface test and is kept
  only as a two-chain smoke case.
- **Measured memory**: peak 8278 MiB of 16303 MiB during the Trp-cage fold, sampled every 0.5 s from
  outside the container. The container does not sample device memory itself, so this number is
  external evidence, not a recorded artifact.
- **`--derived none`**: `completed`, no `metrics/` directory, `derived: null` and a recorded reason.
- **Reproducibility**: a second run of the same input at seed 11 gave A/B `ipsae` 0.365803 against
  0.364598, so Chai-1 is reproducible to a tolerance, not bitwise, as recorded for ESMFold2.

Not verified here: an input large enough to exhaust 16 GB, so the CUDA out-of-memory classification
has not been exercised on this hardware; multi-sample seed derivation; the throughput sweep; the
cloud path.
