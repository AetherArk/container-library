# Chai-1 container: interface specification

Derived from the ESMFold2 container contract in `../esmfold2/` (worker CLI, native YAML inputs,
job/attempt layout, receipts, benchmark instrumentation, OOM isolation, provider independence,
in-container interface scores). Deviations are listed in §9 and each one is forced by the
Chai-1 engine, not by preference.

## 1. Purpose

One resident Chai-1 process predicts protein–ligand complexes from a directory of native
input files and an explicit external seed list, keeps the pinned TorchScript modules available
across every input × seed, retains native outputs and arrays, computes ipSAE-family interface
scores from each PAE before it can be dropped, and writes machine-verifiable receipts.

## 2. Image

- Base `nvidia/cuda:13.0.2-devel-ubuntu24.04`, Python 3.12 venv at `/opt/venv`.
- `torch==2.11.0+cu130` installed explicitly, then `chai_lab==0.6.1` with `--no-deps`; remaining
  direct dependencies pinned in `requirements.lock`.
- Model weights external: the eight artifacts listed in `assets.json`, fetched by
  `fetch_assets.py` with sizes and sha256. `CHAI_DOWNLOADS_DIR` points at the mounted asset
  directory before `chai_lab` is imported, so no download path is taken at run time.
- Installed entrypoints: `cofold-run` (worker), `cofold-throughput` (sweep/aggregate),
  `cofold-cloud-run` (cloud transport).
- Non-root user 1000, `ENTRYPOINT ["cofold-run"]`, labels mirroring the other images
  (`org.opencontainers.image.*`, `ai.aetherark.chai1.source|revision|model_revision`).
- Offline inference; no provider SDK, no credentials, no network at run time.

### Recorded dependency divergence

`chai_lab 0.6.1` metadata declares `torch<2.7,>=2.3.1`. The target RTX 5070 Ti is `sm_120`, and
sm_120 kernels first ship in the Blackwell-capable torch 2.7+ wheels. The image therefore installs
`torch==2.11.0+cu130` and `chai_lab` with `--no-deps`. Consequences:

- `pip check` inside the image reports the chai_lab torch bound as unsatisfied. That is expected,
  and it is why the Dockerfile does not run `pip check`.
- Loading the released TorchScript modules under torch 2.11 is **not verified** in this change:
  no image was built and no GPU job was run. This is the first thing to test on GPU.

## 3. Inputs

Same native YAML format as the other containers: a directory of YAML files, one system per file,
entities in `sequences`, no seeds inside the YAML.

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
2. Every `ligand` needs a SMILES that parses with RDKit. Chai-1 has no fixed ligand-slot limit,
   so several ligands are accepted; each becomes its own FASTA entity and its own chain.
3. `msa:` must be absent or `empty`. A file path is rejected: this container runs without MSA.
4. `templates`, `constraints`, `properties`/affinity, `modifications`, `cyclic`, RNA/DNA entities,
   glycans, CCD codes and pocket/covalent conditioning are rejected with the engine capability.
5. Protein-only (apo) systems are accepted because Chai-1 supports them. ESMFold2 rejects them;
   this is a documented capability superset, not a silent degradation.
6. Referenced files are hash-bound into the job identity exactly as in the other containers.

The YAML is converted to the exact FASTA Chai-1 consumes (`>protein|<id>` / `>ligand|<id>`) and
both files are retained in the attempt (`source/` and `inputs/`). Entity order fixes Chai-1's
one-indexed asym ids.

## 4. Command line

```
cofold-run --input /inputs --output /outputs --assets /models --checkpoint /models \
           --seeds 11 12 --samples 5 --steps 200 --recycles 3
```

| Flag | Meaning | ESMFold2 equivalent |
|---|---|---|
| `--input` | directory of native YAML systems | same |
| `--output` | new attempt root (or `--resume` target) | same |
| `--assets` | populated `CHAI_DOWNLOADS_DIR` (models_v2/, conformers_v1.apkl, esm/) | same position |
| `--checkpoint` | the same directory; hashed once per invocation | same position |
| `--seeds` | external integer seed list, one base seed per entry | same |
| `--samples` | independent samples per base seed | same |
| `--steps` | `num_diffn_timesteps` | `--steps` |
| `--recycles` | `num_trunk_recycles` (default 3, Chai-1's own default) | `--recycles` (`num_loops`, default 20) |
| `--loops` | alias for `--recycles` | alias |
| `--no-low-memory` | disable Chai-1 `low_memory` (default on) | no equivalent |
| `--max-parallel-samples` | must be `1`; other values are rejected | same |
| `--output-format` | `mmcif` only (`pdb` rejected) | same |
| `--derived` | `tensors` (default) writes `metrics/derived.json`; `none` records why | same |
| `--resume` | verify and reuse intact jobs | same |
| `--benchmark-metadata` | optional allowlisted caller metadata JSON | same |
| `--job-timeout` | per input × seed job seconds; recorded as `error_kind: 'timeout'` | same |

`--backend`, `--no-kernels`, `--capture` and `--use-potentials` are not accepted: chai_lab 0.6.1
exposes no selectable kernel backend, capture is always written as native arrays, and there is no
potential/restraint feature in this qualified scope.

Seed mapping (frozen): base seed `S` and zero-based sample index `i` give
`sample_seed = S*1000 + i`. Chai-1's `set_seed` seeds numpy, torch and the stdlib from that value;
the worker also calls `torch.manual_seed` before each fold. `S*1000 + samples - 1` must stay within
signed 32-bit range.

Reproducibility: not measured for Chai-1 in this change. Determinism is claimed only for the
recorded seed and settings, not for bitwise identity; do not assume ESMFold2's tolerance.

## 5. Outputs

Identical layout to the other containers, with Chai-1-specific native files named in §5.1.

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
    protocol.json          resolved engine settings, seed mapping, fold-call record
    checkpoint-lock.json   asset directory path, revision, per-file hashes
    inputs/                the exact YAML plus the generated .fasta
    source/                the exact consumed YAML
    native/                per sample N (one-based)
      <rid>_sample_N.cif                Chai-1's written structure (chains A, B, ...)
      confidence_<rid>_sample_N.json    pLDDT mean, pTM/ipTM, tokens, chain map
      pae_<rid>_sample_N.npz            native PAE matrix (candidates x N x N collapsed to N x N)
      pde_<rid>_sample_N.npz            native PDE matrix
      plddt_<rid>_sample_N.npz          native per-token pLDDT
      tokens_<rid>_sample_N.json        explicit PAE token -> chain/residue/atom map
      chai_sample_N/                    unmodified chai_lab output (pred/scores files)
    metrics/derived.json   ipSAE-family interface scores (shared module)
    benchmark_metrics.json allocator peaks, stage timings for this job
    benchmark_failure.json allocator/load context written on failure
```

`predictions.jsonl` keeps the shared schema: `job_key`, `input`, `seed`, `sample_index`,
`seed_occurrence`, `status`, `verified`, `error_kind`, `output`, `metrics`, `metrics_scope`.
`metrics_scope` is `input_seed_batch_not_individual_sample`.

### 5.1 Chain naming

Chai-1 writes chains as `A`, `B`, `C`, … in asym/entity order, not the input `id`. The attempt's
`chemistry.json` records both (`id` and `cif_chain`), and the token map uses the written CIF chain
ids so scores refer to the structure as written. This differs from ESMFold2, which preserves input
ids in the CIF.

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
  `chemistry.json` (`A`, `B`, … by asym order).
- Protein residue count equals the input sequence length, per chain and in total.
- Ligand heavy-atom count and element counts equal the RDKit reference for the consumed SMILES;
  a missing ligand is a failed job, never an apo fallback.
- PAE and PDE are square over the token count; pLDDT has the token length; all values finite.
- Confidence JSON carries the same token count and chain mapping.
- Raw `chai_sample_N/` must contain both chai_lab files when the directory is present.

## 8. Instrumentation carried over from ESMFold2

- `benchmark.py` stage timers: identity, preparation, load, prediction, verification, receipt.
- `resources.py`: CPU/RAM/IO sampling, cgroup discovery, sampled GPU memory, explicit reasons.
- `errors.py`: CUDA OOM classification across the engine call scope, per-system skip after an
  OOM, continue with independent systems without reloading.
- `failure.json` per failed attempt and `--job-timeout`, with the same recorded-timeout semantics.
- `throughput_benchmark.py`: fixed-budget sweeps and five-host aggregation. Defaults follow the
  engine (`--steps 200`, `--recycles 3`). Only `--levels 1` is measurable; `--no-kernels` and
  `--use-potentials` are refused with explicit messages.
- Checkpoint-load accounting: `checkpoint_load_count` counts the worker's single engine
  initialization. Chai-1 loads six TorchScript components inside *every* fold call; that native
  reload count is recorded separately as `metrics.native_component_loads` (6 × samples) rather
  than hidden in the resident-load count.

## 9. Deviations from the ESMFold2 contract

| Area | ESMFold2 | Chai-1 | Reason |
|---|---|---|---|
| Protein chains | one or more | one or more | same shape |
| Ligands | one ligand + optional heme slot | one or more SMILES ligands | Chai-1 has no fixed ligand-slot count |
| Apo systems | rejected | accepted | Chai-1 supports protein-only folding |
| MSA | `empty` only | `empty` only | qualified offline route has no MSA |
| Templates/constraints/affinity | rejected | rejected | not in the qualified scope |
| Sample parallelism | always 1 | always 1 | one fold call per sample |
| Output format | mmCIF | mmCIF | native writer emits mmCIF |
| Chain ids in CIF | input ids preserved | `A`, `B`, … by asym order | Chai-1 assigns chain letters |
| `--backend` / `--no-kernels` | fused/reference switch | rejected | no kernel-backend switch in chai_lab 0.6.1 |
| `--capture` | rejected (always captures) | rejected (native arrays always written) | no TorchScript activation packet is exposed |
| `capture/` directory | written unconditionally | absent; `native/` holds CIF, PAE, PDE, pLDDT, token map and the raw chai_lab files | chai_lab does not return an activation/result object to capture |
| torch version | 2.11.0 satisfying `esm` | 2.11.0 overriding `chai_lab`'s `<2.7` bound | sm_120 needs a Blackwell-capable wheel |
| Native load accounting | one resident load | resident load + six per-fold component loads | chai_lab loads TorchScript per call |
| Derived interface scores | shared module, token map from PAE inputs | same module, token map from collated features | token identity comes from the tokenized context |

## 10. Acceptance for this container

CPU today: YAML capability matrix, identity/receipt/resume lifecycle with a fake engine, token-map
refusal cases, derived scores on synthetic PAE, verifier regressions, CLI contract, cloud
preparation, sweep contract, and an import check that needs no GPU.

GPU done on a 16 GB RTX 5070 Ti (sm_120) with torch 2.11.0+cu130, chai_lab 0.6.1 and
pre-populated assets: the image builds and every submodule imports; the 20-residue Trp-cage plus
ethanol cell completes in 44-46 s with a measured peak of 8278 MiB of 16303 MiB; the homodimer cell
completes in 80 s; `--derived none` writes no metrics; derived interface scores match the reference
implementation exactly on that run's own CIF and PAE. Still unverified: a cell large enough to
exhaust 16 GB, so the CUDA-out-of-memory classification has not fired in anger here; `--samples 2`
seed derivation; the throughput sweep; and the cloud path.
