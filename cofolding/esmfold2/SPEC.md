# ESMFold2 container: interface specification

Derived from the Boltz2 container contract in `../boltz2/` (worker CLI, native YAML inputs,
job/attempt layout, receipts, benchmark instrumentation, OOM isolation, provider independence).
Deviations are listed in §9 and each one is forced by the ESMFold2 engine, not by preference.

## 1. Purpose

One resident ESMFold2 process predicts protein–ligand complexes from a directory of native
input files and an explicit external seed list, keeps the model loaded across every input ×
seed × sample, retains full native outputs and tensors, and writes machine-verifiable receipts.

## 2. Image

- Base `nvidia/cuda:13.0.2-devel-ubuntu24.04`, Python 3.12 venv at `/opt/venv`.
- `torch==2.11.0+cu130`, `esm[fused]==3.4.1.post1`, `transformers>=4.57.6,<5`,
  `transformer-engine[pytorch]==2.19.0`, `gemmi`; pinned set in `requirements.lock`.
- Model weights external: `biohub/ESMFold2` revision
  `69869f737beffec5294845ede23db5fc0b4f509e`, fetched by `fetch_assets.py` with file hashes.
- Installed entrypoints: `cofold-run` (worker), `cofold-throughput` (sweep/aggregate),
  native `esmfold2-infer` retained for single-cell runs.
- Non-root user 1000, `ENTRYPOINT ["cofold-run"]`, labels mirroring the Boltz image
  (`org.opencontainers.image.*`, `ai.aetherark.esmfold2.source|revision`).
- Offline inference; no provider SDK, no credentials, no network at run time.

## 3. Inputs

Same format as Boltz: a directory of native YAML files, one system per file, entities in
`sequences`, no seeds inside the YAML.

```yaml
version: 1
sequences:
- protein: {id: A, sequence: <canonical 20-letter>}   # one or more chains, ids unique
- protein: {id: B, sequence: <canonical 20-letter>}   # optional additional chain
- ligand:  {id: L, smiles: <SMILES>}                   # exactly one cofold ligand
- ligand:  {id: B, smiles: <heme SMILES>}              # optional, must canonicalize to heme
```

Rules enforced before preparation (each rejection names the rule):

1. At least one `protein` entity; every sequence must be non-empty canonical
   `ACDEFGHIKLMNPQRSTVWY`, and protein chain ids must be unique and distinct from ligand ids.
2. At most one non-heme `ligand`; its SMILES must parse with RDKit.
3. At most one heme ligand; it is recognised by canonical SMILES equality with the reference
   heme in `resident/heme.py`. A second ligand that is not that heme is rejected, because the
   engine has one ligand channel plus one heme channel.
4. `msa:` must be absent or `empty`. A file path is rejected: this engine runs without MSA.
   The heme slot is the ligand whose parsed molecule contains iron; the other ligand is the
   cofold ligand.
5. `templates`, `constraints`, `properties`/affinity, `modifications`, `cyclic`, RNA/DNA entities
   and pocket or covalent-bond conditioning are rejected with the engine's capability name.
6. Referenced files are hash-bound into the job identity exactly as in Boltz.

`cofolding/test-cases/` is shared with the Boltz container, so the same YAML files exercise both
images. `trpcage-ethanol`, `cyp` and the two-chain example (`examples/two-chain.yaml`) are
supported; chains keep the input ids in the written CIF. The upstream input builder
(`StructurePredictionInput`) also accepts RNA/DNA chains, per-chain MSA, pocket conditioning and
covalent bonds; none of those are qualified here.

## 4. Command line

```
cofold-run --input /inputs --output /outputs --assets /models --checkpoint /models/esmfold2 \
           --seeds 11 12 --samples 5 --steps 200 --recycles 20 --backend fused
```

| Flag | Meaning | Boltz equivalent |
|---|---|---|
| `--input` | directory of native YAML systems | same |
| `--output` | new attempt root (or `--resume` target) | same |
| `--assets` | directory holding the pinned checkpoint files and `ccd.pkl` | same |
| `--checkpoint` | checkpoint directory (or Hugging Face id when online) | same position |
| `--seeds` | external integer seed list, one base seed per list entry | same |
| `--samples` | independent samples per base seed | same |
| `--steps` | `num_sampling_steps` | `--steps` |
| `--recycles` | `num_loops` | `--recycles` |
| `--backend` | `fused` (qualified) or `reference` | `--no-kernels` selects reference |
| `--max-parallel-samples` | must be `1`; other values are rejected | same flag, batching supported there |
| `--output-format` | `mmcif`/`cif` only | same flag, `pdb` also allowed there |
| `--resume` | verify and reuse intact jobs | same |
| `--benchmark-metadata` | optional allowlisted caller metadata JSON (`/inputs/<name>`, forwarded by the launcher) | same |
| `--job-timeout` | per input × seed job seconds; overrun is recorded as `error_kind: 'timeout'` | same |

Defaults: `--samples 1 --steps 200 --recycles 20 --backend fused --max-parallel-samples 1`, which are
the engine's own `fold()` defaults. `protocol.json` records both the keyword arguments the worker
sets (`num_loops`, `num_sampling_steps`, `num_diffusion_samples`, `seed`, `include_embeddings`,
`complex_id`) and the native defaults it inherits (`noise_scale`, `lm_dropout=0.3`, and the rest).
The process RNG is seeded with the sample seed before every native call, because `fold()` applies
LM dropout at inference when `lm_dropout > 0`. Measured on GPU: with a fixed seed, PAE is bitwise
identical across runs; coordinates agree to within 0.02 Angstrom but are not bitwise identical,
with fused kernels, the reference backend, or `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Seed mapping (frozen): base seed `S` and one-based sample index `i` give
`sample_seed = S*1000 + i - 1`, the native mapping. `S*1000 + samples - 1` must stay within
signed 32-bit range.

## 5. Outputs

Identical layout to Boltz, so one launcher and one reader serve both images.

```
OUTPUT/
  run.json                 inputs, seeds, settings, identity, job outcomes, load count, status
  progress.json            lightweight live status (+ schema, updated_at_unix, pid)
  termination.json         launcher-written, only for a killed or signalled run
  benchmark.json           stage timers, hardware, sampled GPU/memory
  predictions.jsonl        append-only per-sample rows
  events.jsonl             append-only structured events
  jobs/<identity-key>/<attempt-id>/
    receipt.json           {schema, status, identity, sha256: {path: hash}} for every file in the attempt
    failure.json           failed attempts only: schema, error_kind, phase, exception_type,
                           redacted message, traceback_tail, attempt_path (receipt-referenced)
    chemistry.json         consumed SMILES, canonical SMILES, heavy atoms, charge, formula
    protocol.json          resolved engine settings, seed mapping, precision statement
    checkpoint-lock.json   model id, revision, per-file hashes
    capture/               per sample N (one-based)
      prepared_features_sample_N.pt     observed forward kwargs
      raw_model_output_sample_N.pt      observed forward result
      full_result_sample_N.pt           complete native result object
      full_result_sample_N.npz/.json    portable arrays plus JSON structure
      *_inventory.json                  path, shape, dtype, finiteness per tensor
    native/                per sample N (one-based)
      <rid>_sample_N.cif                native writer output
      confidence_<rid>_sample_N.json    pLDDT/PAE summary scalars, per-residue pLDDT, chain mapping
      pae_<rid>_sample_N.npz            native PAE matrix
      plddt_<rid>_sample_N.npz          native per-token pLDDT
      embeddings_<rid>_sample_N.npz     sequence/pair pooled embeddings
    benchmark_metrics.json allocator peaks, stage timings for this job
    benchmark_failure.json allocator/load context written by the native boundary on failure
```

`predictions.jsonl` keeps the Boltz schema: `job_key`, `input`, `seed`, `sample_index`,
`seed_occurrence`, `status`, `verified`, `error_kind`, `output`, `metrics`, `metrics_scope`.
`metrics_scope` is `input_seed_batch_not_individual_sample`, the single value both containers emit.

## 6. Identity, resume and receipts

- Job identity = input bytes and referenced file hashes + settings + checkpoint + assets +
  implementation + worker version, exactly as in Boltz; `seed_occurrence` disambiguates
  repeated seeds.
- A completed job writes `receipt.json` with the sha256 inventory of the whole attempt;
  `--resume` reuses it only when the identity matches and every hash still matches.
- Interrupted attempts stay in place; a resumed job gets a new attempt directory.
- Exit codes: 0 all requested jobs completed or reused, 1 failure/fatal, 2 rejected before any job,
  124 whole-run timeout, 143 stopped at a job boundary, 137 killed. `run.json.status` is
  authoritative; `termination.json` records a launcher kill or signal.

## 7. Verification

Independent reader checks on every completed job, without trusting the engine's own report:

- CIF parses; all coordinates finite; chain set equals the requested entities (`A`, `L`, heme
  `H` when present).
- Ligand heavy-atom count and element counts equal the RDKit reference for the consumed SMILES;
  a missing ligand is a failed job, never an apo fallback.
- Protein residue count equals the input sequence length.
- `confidence.json` per-residue pLDDT length equals the residue count; PAE matrix is square with
  that shape; all values finite.
- Captured `full_result` coordinates equal the CIF coordinates; the tensor inventory matches the
  saved arrays.

## 8. Instrumentation carried over from Boltz

- `benchmark.py` stage timers: identity, preparation, load, prediction, verification, receipt.
- `resources.py`: CPU/RAM/IO sampling, cgroup discovery, sampled GPU memory and utilization,
  explicit unavailable reasons.
- `errors.py`: CUDA OOM classification across the whole engine call scope, per-system skip after
  an OOM, continue with independent systems without reloading the model.
- `failure.json` per failed attempt (redacted, receipt-referenced diagnosis) and `--job-timeout`,
  which records `error_kind: 'timeout'` at a sample/job boundary and continues with later jobs.
- `throughput_benchmark.py`: fixed-budget sweeps and five-host aggregation. Defaults follow the
  engine: `--steps 200`, `--recycles 20`, `--levels 1`, `--samples 1`. Because this engine cannot
  batch samples, only `--levels 1` is meaningful; both the sweep and the worker reject any other
  level with an explicit message, so a level ladder fails loudly instead of measuring the same
  work twice. `--use-potentials` has no equivalent and is rejected rather than forwarded.
- `--checkpoint` may be a file or a model directory. A directory is hashed once with the worker's
  own sorted per-file scheme, so `sweep.json` and the worker receipts carry the same digest.

## 9. Deviations from the Boltz contract

| Area | Boltz | ESMFold2 | Reason |
|---|---|---|---|
| Protein chains | any number | one or more, ids unique | same shape; per-chain MSA and RNA/DNA are not qualified |
| Ligands | any number | one ligand + optional heme | engine has two ligand slots |
| MSA | local A3M or `empty` | `empty` only | no MSA path in the qualified build |
| Templates/constraints/affinity | rejected | rejected | not supported either |
| Sample parallelism | `max_parallel_samples` 1..N | always 1 | engine samples one trajectory per call |
| Output format | mmCIF/PDB | mmCIF | native writer emits mmCIF |
| Timing scope | per input-seed batch | per sample | one native call per sample |

## 10. Acceptance for this container

CPU tonight: YAML capability matrix, identity/receipt/resume unit tests, fake-engine lifecycle,
resource and OOM regressions, independent verifier against synthetic artifacts, image build and
import smoke test.
GPU tomorrow: one real CYP cell and one Trp-cage cell through `cofold-run`, numeric check of
ligand retention and finite coordinates, full tensor inventory, one resume, then the throughput
command with `--levels 1`.
