# Container development requirements

Applies to every method container added under this repository.

## Interface

- Native input format of the underlying method, unchanged; no repository-specific schema.
- Same job/receipt/output layout as the existing containers, so launchers and readers work across
  methods. Publish a `SPEC.md` for anything that differs.
- One checkpoint load per invocation, across every input × seed in that invocation.

## Heavy outputs

- Anything that writes full model tensors next to the deliverable must have an opt-out flag:
  `--capture {full,none}` in the worker CLI, forwarded unchanged by the launcher
  (`run_cofold.py`) and by any batch driver.
- The capture-producing mode stays the default.
- With capture off, write the structures, confidence JSON, per-model arrays, receipts, metrics and a
  small summary packet the verifier can read. The verifier must accept that mode and reject a stray
  capture directory instead of skipping verification.
- Prove the opt-out is output-neutral rather than assuming it: one input and seed in each mode, native
  files byte-identical, only timing fields differing. Then run once with the flag absent to show the
  default still captures. A default value in code is not a run.
- Record both modes and the measured size difference in `TESTING.md`.

## Images

- Weights and asset caches stay outside the image; `fetch_assets.py --download` populates a mount.
- Record the tested image tag and digest in the root `README.md` table and in the method README.
- A lesson that applies beyond one method goes in a repository-level document under `docs/`, linked
  from that method's README; the method's own operations document keeps only its prepare/run/results
  contract.
