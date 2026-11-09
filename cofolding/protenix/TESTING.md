# Verification

## CPU suite (no GPU, no model weights)

Run from the repository root:

```sh
/home/dan/swr/miniconda3/envs/cheminf/bin/python -m unittest discover -s cofolding/protenix/tests
```

The suite needs `pyyaml`, `numpy`, `rdkit`, `gemmi` and `torch` (CPU). It never imports `protenix`,
`configs` or `runner`, never opens the network and never reads a weight file.

| File | Tests | What is exercised |
|---|---|---|
| `tests/test_capability.py` | 19 | YAML entity mapping, multi-ligand and apo acceptance, heme as a ligand, every rejection rule (RNA/DNA/ion/CCD/modifications/MSA), generated AF3-style JSON, chain letters, packaged examples |
| `tests/test_resident.py` | 28 | lifecycle: one load, seed order, resume, receipts, corruption, fatal stop, SIGTERM, output lock, redacted `failure.json`, job timeout |
| `tests/test_token_map.py` | 13 | token map from tokenized-feature arrays, ion vs ligand kind, shared-reader round trip, PAE/entity/residue/atom/order refusal cases |
| `tests/test_verify.py` | 17 | independent verifier on synthetic artifacts: ligand loss, element mismatch, truncation, NaN, PAE/PDE/pLDDT shapes, atom count, chain set, native full-data PAE agreement |
| `tests/test_cli_contract.py` | 13 | shared-flag table for `protenix`, defaults (`recycles=10`), rejected engine-specific flags, settings validation |
| `tests/test_sweep_contract.py` | 9 | directory checkpoint digest, engine defaults, level/`--no-kernels`/`--use-potentials` refusal, worker argv |
| `tests/test_derived.py` | 7 | engine `derive()` against a synthetic PAE + token map, malformed map skip, recorded placement failure, worker forwarding of `--derived none` |
| `tests/test_assets.py` | 6 | pinned `checkpoint/` + `common/` layout, corrupted bytes, same-size mismatch, missing file, unpinned sha256, absent directory |
| `tests/test_cloud.py` | 5 | cloud preparation against an in-memory object store, input bundle contents, private launch files, settings validation, worker command |
| `tests/test_protocol.py` | 3 | override list vs inherited defaults, `protocol.json`, generated JSON input, seed formula |

Status: 120 tests pass on CPU. `FakeEngine` stands in for the model; no Protenix weights were
loaded and no CUDA call was made.

## What is verified

- The native YAML subset accepted and rejected by the container, including the exact message for
  each refusal, and the exact AF3-style JSON generated for each packaged example.
- Resident lifecycle: one `checkpoint_load_count`, external seed order, resume only on matching
  identity and hashes, fatal stop on model drift, SIGTERM at a job boundary.
- The token map is derived from Protenix's tokenized feature columns and refuses PAE size, entity
  count, chain contiguity, token order, polymer residue count and ligand heavy-atom count
  mismatches; a placement failure becomes a recorded `derived_null_reason`.
- `metrics/derived.json` is produced from a synthetic PAE + token map through the shared
  `cofolding.common` module; `--derived none` writes nothing and records why.
- The independent verifier rejects ligand loss, element mismatch, truncation, NaN coordinates,
  wrong token/atom array shapes, a chain set that differs from the request, and a written PAE that
  disagrees with the writer's own `token_pair_pae`.
- Shared CLI contract: every flag marked `supported` or `rejected` for `protenix` in
  `cofolding/cli-contract.json` is checked against the installed parser, and every enum value the
  worker accepts is reachable through the shared launcher.
- Cloud preparation packages only referenced inputs and writes 0600 launch files.
- Importing `resident.worker`, `resident.engine`, `resident.verify` and the shared derived module
  requires no GPU, no `protenix` and no weights.

## Featurization check against real Protenix (CPU, no weights, 2026-11-09)

`protenix 2.0.0` was installed in a working local image (`local/protenix:2.0.0`), so the parts of
this container that do not need a GPU were run against the real engine instead of only against
test doubles: the YAML → AF3-style JSON conversion, `build_configs`, `InferenceDataset`
featurization and `engine.token_map`, with `PROTENIX_ROOT_DIR` mounted read-only and the network
disabled.

| Example | Tokens | Atoms | Chains (kind, units) | Ligand atoms | `entity_poly_type` |
|---|---|---|---|---|---|
| `trpcage-ethanol` | 23 | 157 | A protein 20, B ligand 3 | 3 | 1 polypeptide(L), 2 non-polymer |
| `two-chain` | 43 | 325 | A protein 20, B protein 20, C ligand 3 | 3 | 1/2 polypeptide(L), 3 non-polymer |
| `small-two-ligand` | 27 | 161 | A protein 20, B ligand 3, C ligand 4 | 7 | 1 polypeptide(L), 2/3 non-polymer |

`token_map` accepted all three real feature dictionaries: chains `A`,`B`,`C` in entity order,
protein `residue_index` 1..20, ligand tokens one per heavy atom with names `C1`,`C2`,`O1` and the
correct element symbols, and the atom array's `label_asym_id` equal to the container's chain
letters for every example. `small-two-ligand`'s acetate is 4 heavy atoms and the map placed 4
tokens, which is the ligand-heavy-atom check working on real Protenix output rather than a fixture.

## What is not verified

- **Image build.** The Dockerfile was not built. The Python 3.12 base and the build-time
  fast-layer-norm compile are the first things a build must prove; see SPEC.md §2.
- **GPU inference.** No checkpoint was loaded and no job was run in this image. Memory fit,
  numeric output, ligand retention on generated coordinates, token-map agreement with a real PAE
  and tensor-to-CIF agreement are all unverified.
- **Derived scores against a reference implementation.** The shared ipSAE module is exercised on
  synthetic PAE only; it has not been compared with a reference run for Protenix output.
- **Determinism.** Not measured. No bitwise or tolerance claim is made.
- **Throughput sweep end to end.** Only its argument, digest and argv contract is on CPU.
- **Published image.** None exists; no digest, no anonymous pull, no installed-image check.

## GPU acceptance plan

1. Build the image with `--build-context common=cofolding/common`; run the CPU suite inside it.
2. `cofold-run` on `examples/trpcage-ethanol.yaml`, one seed, one sample; record peak memory and
   the `protocol.json`/`benchmark_metrics.json` output.
3. Same on `examples/two-chain.yaml` and `examples/small-two-ligand.yaml`; check the chain map,
   ligand element counts and that each `metrics/derived.json` pair matches the token map.
4. `--samples 2` with one seed: seeds `S*1000` and `S*1000+1`, two independent CIFs.
5. `--resume`: `checkpoint_load_count: 0`, jobs reused and independently verified.
6. `--derived none`: no `metrics/` directory and a recorded reason.
7. `cofold-throughput sweep --levels 1` for the benchmark record.
8. A cell above 2560 tokens must fail with the upstream guard recorded as a job failure, not a
   crash of the resident process.

## GPU acceptance, run on a 16 GB RTX 5070 Ti (sm_120)

Local image built from this tree with `--build-context common=cofolding/common`
(`container-library/protenix:test5`); torch 2.7.1+cu128, protenix 2.0.0; assets mounted read-only
from `/nvme1/dan/folding/cache/protenix-root` (checkpoint plus the four `common/` files); runs
through `run_cofold.py` on GPU 1, seed 11, one sample, ten recycles, 200 steps.

- **Trp-cage + ethanol**: `completed`, 46.7 s, peak 3946 MiB of 16303 MiB, sampled every 0.5 s from
  outside the container. `native/` holds the CIF, PAE npz, PDE npz, pLDDT npz, token map and the
  Protenix `predictions/` tree; `metrics/derived.json` scores A/B at cutoff 10 with `ipsae`
  0.777462, `ipsae12` 0.777462, `ipsae21` 0.057346, `n0` 3 / 20.
- **Homodimer**: `completed`, 49.5 s, three scored pairs.
- **Reference cross-check** on the homodimer run's own CIF and PAE (IPSAE2 0.1.10 `df6e46f2`):
  A/B `ipsae` 0.124257 and `ipsae21` 0.075041 with `n0` 18/20; A/C 0.715645 / 0.068005 with
  `n0` 3/20; B/C 0.402425 / 0.043734 with `n0` 3/15. Every column equals the container's value,
  which is what validates the token map built from Protenix's own token features.
- **Kernel**: the build asserts the bundled `fast_layer_norm_cuda_v2.so` carries an sm_120 cubin
  (`cuobjdump -lelf` lists sm_70/80/86/89/90/100/120). The first GPU attempt died with "no kernel
  image is available for execution on the device" because the build host's architecture detection
  stopped at compute_100; that failure mode is now caught at build time.

Not verified: an input large enough to exhaust 16 GB, `--samples 2` seed derivation, the throughput
sweep, the cloud path, and run-to-run reproducibility.
