# Resident ESMFold2 worker

- One model/checkpoint load per invocation; unchanged native YAML × external `--seeds` list.
- Native `builder.fold` per sample (one diffusion trajectory per call), native mmCIF writer; model
  remains on CUDA.
- Defaults: one sample, 20 loops, 100 sampling steps, fused kernels, mmCIF.
- `msa: empty` only; relative references are rejected because this engine has no MSA or template path.
- Structure prediction only; the native `esmfold2-infer` CLI remains available in the image.

## Run

- `cofold-run --input /inputs --output /outputs --assets /models --checkpoint /models --seeds 11 12 --samples 3 --steps 100 --recycles 20`.
- `--checkpoint` and `--assets` are normally the same model directory; it is hashed once per
  invocation and each job records the per-file lock.

## Outputs

- Per sample: native mmCIF, `confidence_<id>_sample_<n>.json`, `pae_*.npz`, `plddt_*.npz`,
  `embeddings_*.npz`.
- `capture/`: observed forward kwargs, observed raw output, the complete native result, the
  portable NPZ bundle and per-tensor inventory.
- `run.json`, `progress.json`, `benchmark.json`, `predictions.jsonl`, `events.jsonl`, and
  `jobs/<identity-key>/<attempt-id>/receipt.json`.

## Recovery

- `--resume`: reuse intact outputs matching input, checkpoint directory, implementation and
  settings; preserve earlier attempts.
- No checkpoint load when all requested jobs are reused.
- Preparation failures stay local; unusable CUDA or a changed model ends the invocation without
  reload. SIGTERM finishes the current job and stops at the boundary.

## Verification

- Independent reader checks: chain set equals the requested entities, protein residue count,
  ligand and heme heavy-atom/element counts from the consumed SMILES, finite coordinates, PAE and
  pLDDT shapes, and coordinate agreement between the captured tensors and the written CIF.
- CPU lifecycle tests use an explicit test double; GPU acceptance is separate.
