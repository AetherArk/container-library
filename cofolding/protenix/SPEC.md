# Protenix container: interface specification

Derived from the ESMFold2/Chai-1 contract in `../esmfold2/` and `../chai1/` (worker CLI, native
input directory, job/attempt layout, receipts, benchmark instrumentation, OOM isolation, provider
independence, in-container interface scores). Deviations are listed in §9 and each one is forced by
the Protenix engine, not by preference.

## 1. Purpose

One resident Protenix 2.0.0 process predicts protein–ligand complexes from a directory of native
input files and an explicit external seed list, keeps the released checkpoint loaded across every
input × seed, retains the native mmCIF, PAE, PDE and per-atom pLDDT, computes ipSAE-family
interface scores from each PAE before it can be dropped, and writes machine-verifiable receipts.

## 2. Image

- Base `nvidia/cuda:12.8.1-devel-ubuntu24.04` (pinned by digest), Python 3.12 venv at `/opt/venv`.
- `torch==2.7.1` / `torchvision==0.22.1` / `torchaudio==2.7.1` from the `cu128` wheel index, then
  `requirements.lock`, then `protenix==2.0.0` with `--no-deps`.
- `requirements.lock` is the `pip freeze` of the working `local/protenix:2.0.0` environment
  (cuequivariance(-ops(-torch))-cu12 0.8.0, biotite 1.4.0, gemmi 0.6.7, rdkit 2025.9.3,
  numpy 2.4.1, …), minus `torch`/`torchvision`/`torchaudio`, `triton` and the `nvidia-*-cu12`
  runtime wheels (exact dependencies of the torch wheel installed immediately before it), and
  minus two unused packages that no code path imports and that would not build on Python 3.12:
  `deepspeed` (only the never-selected `deepspeed` triangle-attention backend imports it) and
  `scikit-learn-extra` (no cp312 wheel; its sdist does not build against numpy>=2).
- Model weights and the CCD/cluster cache are external: the five artifacts in `assets.json`,
  fetched by `fetch_assets.py` with sizes and sha256. `PROTENIX_ROOT_DIR` points at the mounted
  asset directory before any `protenix` / `configs` import, so no download path is taken at run
  time. The mount is read-only.
- The bundled fast-layer-norm CUDA kernel (`protenix.model.layer_norm`) is compiled **at build
  time** into site-packages for every architecture `nvcc` supports (sm_70…sm_120), so a run never
  invokes a CUDA toolchain. `build_probe.py` imports every module the worker reaches and fails the
  build if the lock is incomplete.
- Installed entrypoints: `cofold-run` (worker), `cofold-throughput` (sweep/aggregate),
  `cofold-cloud-run` (cloud transport).
- Non-root user 1000, `ENTRYPOINT ["cofold-run"]`, labels mirroring the other images
  (`org.opencontainers.image.*`, `ai.aetherark.protenix.source|revision|model_revision`).
- Offline inference; no provider SDK, no credentials, no network at run time.

### Recorded stack divergence

The proven local image is Ubuntu 22.04 / Python **3.11.15** with `torch 2.7.1+cu128` and
`protenix 2.0.0` built from `/opt/Protenix`. This container keeps the torch/CUDA pins and every
Python package version from that freeze, but builds on the `nvidia/cuda:12.8.1-devel-ubuntu24.04`
base because that is the base the other containers in this repository use and it carries
Python 3.12 without a third-party PPA. `protenix` declares `python_requires>=3.11` and the
cuequivariance 0.8.0 ops ship cp312 wheels, so the interpreter differs while the numeric stack
does not. The image has **not been built**; the 3.12 path is the first thing to verify (see
PUBLISHING.md).

`--build-context common=cofolding/common` installs the shared numpy-only interface-scoring module
at `/opt/cofold/cofolding/common`; that module is reused unchanged.

## 3. Inputs

The shared native YAML format, one system per file, entities in `sequences`, no seeds inside the
YAML:

```yaml
version: 1
sequences:
- protein: {id: A, sequence: <canonical 20-letter>}   # one or more chains, ids unique
- protein: {id: B, sequence: <canonical 20-letter>}
- ligand:  {id: L, smiles: <SMILES>}                   # one or more SMILES ligands
```

Rules enforced before preparation (each rejection names the rule):

1. At least one `protein` entity; every sequence must be non-empty canonical
   `ACDEFGHIKLMNPQRSTVWY`; entity ids must be unique across proteins and ligands.
2. Every `ligand` needs a SMILES that parses with RDKit. Protenix has no fixed ligand-slot limit,
   so several ligands are accepted; each becomes its own chain.
3. `msa:` must be absent or `empty`. A file path is rejected: this container runs without MSA.
4. `templates`, `constraints`, `properties`/affinity, `modifications`, `cyclic`, RNA/DNA entities,
   `ion` entities and CCD codes are rejected with the engine capability. An ion must be passed as
   a one-atom SMILES ligand.
5. Protein-only (apo) systems are accepted; Protenix folds them.
6. Referenced files are hash-bound into the job identity exactly as in the other containers.

### Protenix's native input

Protenix reads an AF3-style JSON list, not YAML. The YAML is converted per job into

```json
[{"name": "<stem>", "sequences": [
   {"proteinChain": {"sequence": "...", "count": 1, "id": ["A"]}},
   {"ligand": {"ligand": "CCO", "count": 1, "id": ["B"]}}]}]
```

and both files are kept in the attempt (`source/` and `inputs/`). One YAML entity is one chain
(`count: 1`), so `entity_id` equals the entity position. The chain id is written explicitly — the
container's own `A`, `B`, … letters — instead of letting Protenix's free-letter allocator walk
input ids, which keeps the token map and the written mmCIF in agreement for any input ids.

`use_msa=false` selects Protenix's single-sequence dummy-MSA path
(`protenix/data/utils.make_dummy_feature`). `use_template=false`: the template branch needs the
`mmcif/` tree and is outside this container's qualified scope.

## 4. Command line

```
cofold-run --input /inputs --output /outputs --assets /models --checkpoint /models \
           --seeds 11 12 --samples 5 --steps 200 --recycles 10
```

| Flag | Meaning | Other containers |
|---|---|---|
| `--input` | directory of native YAML systems | same |
| `--output` | new attempt root (or `--resume` target) | same |
| `--assets` | populated `PROTENIX_ROOT_DIR` (`checkpoint/`, `common/`) | same position |
| `--checkpoint` | the same directory; hashed once per invocation | same position |
| `--seeds` | external integer seed list, one base seed per entry | same |
| `--samples` | independent samples per base seed | same |
| `--steps` | `sample_diffusion.N_step` (engine default 200) | `--steps` |
| `--recycles` | `model.N_cycle` (engine default **10** for protenix-v2) | Chai-1 default 3 |
| `--loops` | alias for `--recycles` | alias |
| `--max-parallel-samples` | must be `1`; other values are rejected | same |
| `--output-format` | `mmcif` only (`pdb` rejected) | same |
| `--derived` | `tensors` (default) writes `metrics/derived.json`; `none` records why | same |
| `--resume` | verify and reuse intact jobs | same |
| `--benchmark-metadata` | optional allowlisted caller metadata JSON | same |
| `--job-timeout` | per input × seed job seconds; recorded as `error_kind: 'timeout'` | same |

`--backend`, `--no-kernels`, `--capture`, `--use-potentials` and Chai-1's `--no-low-memory` are not
accepted: protenix 2.0.0 exposes no selectable kernel backend or potential/restraint feature at the
worker CLI, the released protenix-v2 configuration runs its cuequivariance triangle kernels
unconditionally, capture is always written as native arrays, and Protenix has no `low_memory`
switch (its memory knob is automatic chunking driven by token count).

Seed mapping (frozen): base seed `S` and zero-based sample index `i` give
`sample_seed = S*1000 + i`, passed to `protenix.utils.seed.seed_everything` before every forward.
`S*1000 + samples - 1` must stay within signed 32-bit range.

Reproducibility: not measured for Protenix in this change. Determinism is claimed only for the
recorded seed and settings, not for bitwise identity. Note that Protenix generates the ligand
reference conformer with RDKit `EmbedMolecule` during featurization; the base seed is applied
before featurization so that conformer is reproducible for a recorded seed, which Chai-1 does not
need.

## 5. Outputs

Identical layout to the other containers, with Protenix-specific native files named in §5.1.

```
OUTPUT/
  run.json                 inputs, seeds, settings, identity, job outcomes, load count, status
  progress.json            lightweight live status (+ schema, updated_at_unix, pid)
  termination.json         launcher-written, only for a killed or signalled run
  benchmark.json           stage timers, hardware, sampled GPU/memory
  predictions.jsonl        append-only per-sample rows
  events.jsonl             append-only structured events
  jobs/<identity-key>/<attempt-id>/
    receipt.json           {schema, status, identity, sha256: {path: hash}}
    failure.json           failed attempts only: schema, error_kind, phase, exception_type,
                           redacted message, traceback_tail, attempt_path
    chemistry.json         chains, input ids, written CIF chain ids, SMILES/hash records
    protocol.json          resolved configuration overrides, inherited defaults, seed mapping
    checkpoint-lock.json   asset directory path, revision, per-file hashes
    inputs/                the exact YAML plus the generated AF3-style .json
    source/                the exact consumed YAML
    native/                per sample N (one-based)
      <rid>_sample_N.cif                Protenix's written structure (chains A, B, ...)
      confidence_<rid>_sample_N.json    pLDDT mean, pTM/ipTM, tokens, atoms, chain map
      pae_<rid>_sample_N.npz            native per-token PAE matrix (N_token x N_token)
      pde_<rid>_sample_N.npz            native per-token PDE matrix
      plddt_<rid>_sample_N.npz          native per-atom pLDDT (0-100)
      tokens_<rid>_sample_N.json        explicit PAE token -> chain/residue/atom map
      protenix/<rid>.seed_<seed>/<rid>/seed_<seed>/predictions/
                                        unmodified DataDumper output (cif, _wounresol.cif,
                                        <rid>_summary_confidence_sample_0.json,
                                        <rid>_full_data_sample_0.json)
    metrics/derived.json   ipSAE-family interface scores (shared module)
    benchmark_metrics.json allocator peaks, stage timings for this job
    benchmark_failure.json allocator/load context written on failure
```

`predictions.jsonl` keeps the shared schema: `job_key`, `input`, `seed`, `sample_index`,
`seed_occurrence`, `status`, `verified`, `error_kind`, `output`, `metrics`, `metrics_scope`.
`metrics_scope` is `input_seed_batch_not_individual_sample`.

### 5.1 Where the PAE is

Protenix has no dedicated PAE writer. The confidence head produces
`pred_dict["full_data"][i]["token_pair_pae"]` (`[N_token, N_token]`) and keeps it only there:
`runner/dumper.py` writes it **only when `need_atom_confidence=True`**, as the
`token_pair_pae` key of `<rid>_full_data_sample_<rank>.json`, and `get_clean_full_confidence`
rounds that copy to two decimals in place. The per-sample `<rid>_summary_confidence_sample_<rank>.json`
never contains the matrix.

This container therefore

- sets `need_atom_confidence=True`, so the engine's own writer emits the PAE as native evidence
  under `native/protenix/…/predictions/` (CIF, `_wounresol.cif`, summary confidence, full-data
  confidence);
- copies the **unrounded** tensor out of `pred_dict["full_data"][i]["token_pair_pae"]` **before**
  the writer runs (the cleaner rounds it in place) and writes it as `pae_<rid>_sample_N.npz`;
- scores that same array with the shared ipSAE module;
- has the independent verifier compare the npz against the writer's rounded copy, so a
  transcription mistake cannot pass.

`pde_*` and `plddt_*` come from the same dict (`token_pair_pde`, `atom_plddt`). Per-sample scalars
(`plddt`, `ptm`, `iptm`, `ranking_score`, `has_clash`, `chain_ptm`, `chain_pair_iptm`) come from
`summary_confidence[i]` and are mirrored into `confidence_*.json` for the shared reader.

Unlike the other engines, Protenix reports **pLDDT per atom**, not per token, so
`plddt_*.npz` has `N_atom` entries and `confidence_*.json` records both `tokens` and `atoms`.
`plddt` is written on the native 0-100 scale used by the CIF `B_iso_or_equiv` column.

### 5.2 Chain naming

Chain ids in the written mmCIF are the container's `A`, `B`, `C`, … in entity order, because the
generated JSON passes them explicitly. The attempt's `chemistry.json` records both the input `id`
and the `cif_chain`, and the token map uses the written CIF chain ids so scores refer to the
structure as written. This matches Chai-1 and differs from ESMFold2, which preserves input ids.

### 5.3 Token layout

`protenix/data/tokenizer.py` gives one token per standard protein residue and one token per atom
for non-standard residues and ligands. The token map is derived from Protenix's own tokenized
feature columns (`token_index`, `residue_index`, `asym_id`, `atom_to_token_idx`, `is_ligand`,
`ref_element`, `ref_atom_name_chars`) and validated against the PAE size, the entity order, the
polymer residue counts and the ligand heavy-atom counts. Any disagreement raises
`derived.TokenMapError`, which is recorded verbatim in `metrics/derived.json` as
`derived_null_reason`; nothing is guessed.

## 6. Identity, resume and receipts

- Job identity = input bytes and referenced file hashes + settings + checkpoint + assets +
  implementation + worker version; `seed_occurrence` disambiguates repeated seeds.
- A completed job writes `receipt.json` with the sha256 inventory of the whole attempt;
  `--resume` reuses it only when the identity matches and every hash still matches.
- Interrupted attempts stay in place; a resumed job gets a new attempt directory.
- Exit codes: 0 all requested jobs completed or reused, 1 failure/fatal, 2 rejected before any job,
  124 whole-run timeout, 143 stopped at a job boundary, 137 killed. `run.json.status` is
  authoritative; `termination.json` records a launcher kill or signal.

## 7. Verification

Independent reader checks on every completed job, without trusting the engine's own report:

- CIF parses; all coordinates finite; chain set equals the chain letters recorded in
  `chemistry.json`.
- Protein residue count equals the input sequence length, per chain and in total.
- Ligand heavy-atom count and element counts equal the RDKit reference for the consumed SMILES;
  a missing ligand is a failed job, never an apo fallback.
- PAE and PDE are square over the token count; pLDDT has the atom count and is finite and within
  0-100; confidence JSON carries the same token count, atom count and chain mapping, and the
  written atom count.
- Raw `native/protenix/…/predictions/` must contain the dumper's CIF, summary confidence and
  full-data confidence when the directory is present, and the rounded `token_pair_pae` in the
  full-data JSON must agree with `pae_*.npz` within the rounding step (checked for files up to
  64 MiB).

## 8. Instrumentation carried over from the other containers

- `benchmark.py` stage timers: identity, preparation, featurization inside prediction, load,
  verification, derived metrics, receipt.
- `resources.py`: CPU/RAM/IO sampling, cgroup discovery, sampled GPU memory, explicit reasons.
- `errors.py`: CUDA OOM classification across the engine call scope, per-system skip after an
  OOM, continue with independent systems without reloading.
- `failure.json` per failed attempt and `--job-timeout`, with the same recorded-timeout semantics.
- `throughput_benchmark.py`: fixed-budget sweeps and five-host aggregation. Defaults follow the
  engine (`--steps 200`, `--recycles 10`). Only `--levels 1` is measurable; `--no-kernels` and
  `--use-potentials` are refused with explicit messages.
- Checkpoint-load accounting: `checkpoint_load_count` counts the worker's single model
  initialization. Protenix loads the checkpoint once per process; no per-fold reload exists, so
  there is no separate native component counter as in Chai-1.

## 9. Deviations from the other containers

| Area | ESMFold2 / Chai-1 | Protenix | Reason |
|---|---|---|---|
| Native input | YAML consumed directly (Chai-1: YAML→FASTA) | YAML→AF3-style JSON | protenix 2.0.0 reads JSON only |
| Protein chains | one or more | one or more | same shape |
| Ligands | one or more SMILES | one or more SMILES | protenix has no ligand-slot count |
| Ion entities | n/a | rejected by name; pass a one-atom SMILES | the AF3-style `ion` entity is outside the qualified scope |
| Apo systems | ESMFold2 rejects, Chai-1 accepts | accepted | Protenix folds protein-only systems |
| MSA | `empty` only | `empty` only (`use_msa=false`, dummy single-sequence MSA) | offline route has no MSA |
| Templates/constraints/affinity | rejected | rejected | not in the qualified scope |
| `--recycles` default | 3 (Chai-1 `num_trunk_recycles`) | 10 (`model.N_cycle`, the v2 default) | engine default is the container default |
| Memory switch | Chai-1 `--no-low-memory` | absent | Protenix chunks automatically; no such flag |
| Sample parallelism | always 1 | always 1 (`N_sample` pinned to 1) | one forward = one scored structure |
| Output format | mmCIF | mmCIF | native writer emits mmCIF |
| Chain ids in CIF | ESMFold2 keeps input ids; Chai-1 assigns `A`, `B`, … | `A`, `B`, … by entity order | the JSON id is written explicitly |
| `--backend` / `--no-kernels` | Boltz2/ESMFold2 switch | rejected | no worker-level kernel backend in protenix 2.0.0 |
| `--capture` | Boltz2 opt-out | rejected (native arrays always written) | the engine returns no activation packet |
| pLDDT granularity | per token (Chai-1/ESMFold2) | per atom | `atom_plddt` is what the confidence head scores |
| PAE file | separate native PAE file/npz | inside the confidence output, no file of its own | see §5.1 |
| torch / CUDA | 2.11.0+cu130 | 2.7.1+cu128 | 2.7.1+cu128 is the measured Protenix stack and supports sm_120 |
| Python | 3.12 | 3.12 | proven image is 3.11; see §2 |
| Size limit | none stated | `n_token > 2560` refused by upstream for protenix-v2 | upstream guard against OOM |
| Derived interface scores | shared module, token map from engine arrays | same module, token map from Protenix token features | token identity comes from the tokenizer |

## 10. Acceptance for this container

CPU today: YAML capability matrix, identity/receipt/resume lifecycle with a fake engine, token-map
refusal cases, derived scores on synthetic PAE, verifier regressions, CLI contract, cloud
preparation, sweep contract, and an import check that needs no GPU.

GPU acceptance on a 16 GB RTX 5070 Ti (sm_120), torch 2.7.1+cu128, protenix 2.0.0, ten recycles,
200 steps: the image builds with the bundled fast-layer-norm kernel carrying an sm_120 cubin, the
Trp-cage/ethanol and homodimer cells both complete (46.7 s and 49.5 s), the Trp-cage fold peaks at
3946 MiB, `--derived none` writes no metrics, and the derived interface scores match the reference
implementation exactly on the run's own CIF and PAE. Not verified: an input large enough to exhaust
16 GB, multi-sample seed derivation, the throughput sweep, the cloud path, and run-to-run
reproducibility (the earlier containers have shown tolerance-level variation rather than byte
identity, and Protenix has not been measured for it).

Two build-time defects found here and fixed: the bundled kernel compiled without an sm_120 cubin,
because Protenix's own architecture detection on a GPU-less build host stops at compute_100, so the
kernel is now compiled explicitly with an sm_120 gencode and the build fails if the cubin is absent;
and the engine refused the launcher's two mounts (`/models` and `/checkpoint`) even when the caller
passed one host directory, because it compared paths rather than content.
