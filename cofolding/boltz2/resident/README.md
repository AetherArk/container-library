# Resident Boltz2 worker

- One model/checkpoint load per invocation; unchanged native YAML × external `--seeds` list.
- Native `predict_step`, BF16 inference, native writer; model remains on CUDA.
- Defaults: one sample, three recycles, 200 steps, one parallel sample, mmCIF.
- Local MSAs or `msa: empty`; relative references resolve from the YAML directory.
- Structure prediction only; native `boltz` remains available via `--entrypoint boltz`.

## Run

- `cofold-run --input /inputs --output /outputs --assets /models --checkpoint /checkpoint --seeds 11 12 --no-kernels`.
- Shared host launcher and local tests: [LOCAL_TESTING.md](../../../LOCAL_TESTING.md).

## Outputs

- Raw structures, confidence/embedding arrays, consumed features, full forward/prediction tensors.
- `run.json`: inputs, seeds, settings, job outcomes, checkpoint/model identity; provisional metadata format.
- `progress.json`: lightweight live status; full `run.json` written at start and termination.
- Job attempts: `jobs/<identity-key>/<attempt-id>/`; completion receipt and parameter-device/pointer snapshots.

## Recovery

- `--resume`: reuse intact outputs matching input/dependencies, checkpoint, implementation and settings; preserve earlier attempts.
- No checkpoint load when all requested jobs are reused.
- Preparation failures remain local; unusable CUDA or changed resident model ends the invocation without reload.
- SIGTERM: finish current job, stop at boundary; exits 0 completed, 1 failed/fatal, 143 stopped.
- Trusted local checkpoint/output files; assets remain immutable during execution.

## Verification

- Native tensor, atom, confidence, sample mapping and writer checks; additional source chemistry/MSA checks for the established single-protein/SMILES route.
- CPU lifecycle tests use an explicit test double; packaged GPU testing is separate.
