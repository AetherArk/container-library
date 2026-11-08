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
- protein: {id: A, sequence: <canonical 20-letter>}   # exactly one protein
- ligand:  {id: L, smiles: <SMILES>}                   # exactly one cofold ligand
- ligand:  {id: B, smiles: <heme SMILES>}              # optional, must canonicalize to heme
```

Rules enforced before preparation (each rejection names the rule):

1. Exactly one `protein` entity; its sequence must be non-empty canonical `ACDEFGHIKLMNPQRSTVWY`.
2. At most one non-heme `ligand`; its SMILES must parse with RDKit.
3. At most one heme ligand; it is recognised by canonical SMILES equality with the reference
   heme in `resident/heme.py`. A second ligand that is not that heme is rejected, because the
   engine has one ligand channel plus one heme channel.
4. `msa:` must be absent or `empty`. A file path is rejected: this engine runs without MSA.
   The heme slot is the ligand whose parsed molecule contains iron; the other ligand is the
   cofold ligand.
5. `templates`, `constraints`, `properties`/affinity, `modifications`, `cyclic`, and multi-chain
   proteins are rejected with the engine's capability name.
6. Referenced files are hash-bound into the job identity exactly as in Boltz.

`cofolding/test-cases/` is shared with the Boltz container, so the same YAML files exercise both
images. `trpcage-ethanol` and `cyp` are supported; `zika` is rejected (two protein chains) and
the rejection is part of the test suite.

## 4. Command line

```
cofold-run --input /inputs --output /outputs --assets /models --checkpoint /models/esmfold2 \
           --seeds 11 12 --samples 5 --steps 100 --recycles 20 --backend fused
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
| `--metadata` | optional allowlisted host metadata for the benchmark | same |

Defaults: `--samples 1 --steps 100 --recycles 20 --backend fused --max-parallel-samples 1`.

Seed mapping (frozen): base seed `S` and one-based sample index `i` give
`sample_seed = S*1000 + i - 1`, the native mapping. `S*1000 + samples - 1` must stay within
signed 32-bit range.

## 5. Outputs

Identical layout to Boltz, so one launcher and one reader serve both images.

```
OUTPUT/
  run.json                 inputs, seeds, settings, identity, job outcomes, load count, status
  progress.json            lightweight live status
  benchmark.json           stage timers, hardware, sampled GPU/memory
  predictions.jsonl        append-only per-sample rows
  events.jsonl             append-only structured events
  jobs/<identity-key>/<attempt-id>/
    receipt.json           {status, identity, sha256: {path: hash}} for every file in the attempt
    chemistry.json         consumed SMILES, canonical SMILES, heavy atoms, charge, formula
    protocol.json          resolved engine settings, seed mapping, precision statement
    checkpoint-lock.json   model id, revision, per-file hashes
    capture/
      prepared_features.pt observed forward kwargs
      raw_model_output.pt  observed forward result
      full_result.pt       complete native result object
      full_result.npz/.json portable arrays plus JSON structure
      full_result_inventory.json  path, shape, dtype, finiteness per tensor
    native/
      prediction.cif       native writer output
      confidence.json      pLDDT/PAE summary scalars, per-residue pLDDT, chain mapping
      confidence_pae.npz   native PAE matrix
    benchmark_metrics.json allocator peaks, stage timings for this job
    model_parameters_before.json / model_parameters_after.json
```

`predictions.jsonl` keeps the Boltz schema: `job_key`, `input`, `seed`, `sample_index`,
`seed_occurrence`, `status`, `verified`, `error_kind`, `output`, `metrics`, `metrics_scope`.
`metrics_scope` is `input_seed_sample_not_batch`, because each sample is its own native call.

## 6. Identity, resume and receipts

- Job identity = input bytes and referenced file hashes + settings + checkpoint + assets +
  implementation + worker version, exactly as in Boltz; `seed_occurrence` disambiguates
  repeated seeds.
- A completed job writes `receipt.json` with the sha256 inventory of the whole attempt;
  `--resume` reuses it only when the identity matches and every hash still matches.
- Interrupted attempts stay in place; a resumed job gets a new attempt directory.
- Exit codes: 0 all requested jobs completed or reused, 1 failure/fatal, 143 stopped at a job
  boundary.

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
- `throughput_benchmark.py`: fixed-budget sweeps and five-host aggregation. Because this engine
  cannot batch samples, only `--levels 1` is meaningful. The worker itself rejects any other
  level with an explicit message, so a level ladder fails loudly instead of measuring the same
  work twice.

## 9. Deviations from the Boltz contract

| Area | Boltz | ESMFold2 | Reason |
|---|---|---|---|
| Protein chains | any number | exactly one | engine takes one protein sequence per call |
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
