# OpenFold3 on Blackwell sm_120 — throughput headroom review (code inspection only)

## 1. Inspected

- Repo: https://github.com/aqlaboratory/openfold-3 (local clone `/home/dan/of3`)
- Commit: `d97abeffc603f43b8dd68b3bb03ab15c7fb42787`, 2026-11-06, "Merge pull request #501 … cif-gz-support"
- Files read: `projects/of3_all_atom/{model.py,runner.py,config/model_config.py,config/model_setting_presets.yml,project_entry.py}`; `core/model/structure/diffusion_module.py`; `core/model/layers/{diffusion_transformer.py,attention_pair_bias.py,triangular_multiplicative_update.py}`; `core/model/primitives/attention.py`; `core/kernels/{cueq_utils.py,triton/evoformer.py,triton/dot_precision.py}`; `entry_points/{experiment_runner.py,validator.py,import_utils.py}`; `core/data/framework/data_module.py`; `pixi.toml`; `docker/Dockerfile.pixi`; `docs/source/kernels.md`; `examples/example_runner_yamls/*.yml`.
- Weight sets: `docs/source/parameters_reference.md:20-35` — `of3-ob-2025-06-30-174k` needs `>=0.5.0`; p2 (`of3-p2-155k.pt`) needs an older release. Tag `0.4.5` already ships the same Triton kernel set with `use_triton_triangle_kernels: True` under `eval` (`git show 0.4.5:…/model_config.py`, line 119), so this analysis transfers; p2-only deltas are not throughput-relevant.

## 2. Where the runtime goes

Default inference is 1 trunk (4 cycles, `num_recycles: 3`, model_config.py:178) then **200 diffusion steps × 5 samples** (model_config.py:184-186), each a full 24-block transformer plus atom encoder/decoder (diffusion_module.py:309-337; model_config.py:298,346,362). My static MAC count (estimate, ±2×, 384 tokens / 3 k atoms / MSA 256) puts ~90 % of arithmetic in the sampling loop.

Stage classification on a 16 GB sm_120 part:

- **Trunk pairformer/MSA (48 + 4 blocks):** compute-bound above ~256 tokens; chunked at `chunk_size: 1024` (`model_setting_presets.yml`) and unchunked below `per_sample_token_cutoff: 750` (model_config.py:57).
- **Diffusion loop:** latency/dispatch-bound at small–medium N — 200 sequential steps at ~10² launches each, roughly 2·10⁴ launches/query — becoming compute-bound only at large atom counts. This is where wall-clock lives.
- **Elementwise/modulation/masking:** bandwidth-bound, all fp32 (validator.py:158) on a 448–896 GB/s card.

## 3. Candidate optimizations

### 3.1 bf16 (or fp16) autocast for inference
`validator.py:158` defaults to `32-true`; `import_utils.py:36` only turns on TF32. Everything runs fp32, including the Triton dots (`current_dot_precision`, dot_precision.py:46-60). Mixed precision shrinks both the compute-bound trunk and the bandwidth-bound elementwise work.
**Estimate:** 25–45 % e2e; confidence high on speed, medium on quality; cost ~1 h + validation.

### 3.2 CUDA graphs / `torch.compile` around the denoising step
No `torch.compile` or graph capture anywhere (only `@torch.compiler.disable` guards, attention.py:547,616,742). The step (diffusion_module.py:309-337) is shape-static per query, with the five samples folded into the batch dim (model.py:670-676).
**Estimate:** 10–25 % small inputs, 4–10 % large; confidence medium; cost 2–4 days (re-capture per shape, allocator pool).

### 3.3 Per-step host sync in the sampler
`diffusion_module.py:314`: `gamma = self.gamma_0 if c_tau > self.gamma_min else 0`, where `c_tau` is a 0-dim **device** tensor from `noise_schedule` (line 52-90) → 200 device→host syncs/query, plus `t.to(device)` (line 315). Hoisting the schedule to floats removes them.
**Estimate:** 1–3 %; confidence high; cost <1 h.

### 3.4 Keep Triton on *and* add cuEquivariance
`examples/example_runner_yamls/cuequivariance.yml` sets `use_triton_triangle_kernels: false`. cueq falls back for fp32 head dim >32 (attention.py:71-89), and the diffusion transformer's head dim is 48 (model_config.py:322-323), so that yaml silently pushes the 90 %-of-FLOPs attention onto the plain `einsum`+`softmax` path (attention.py:160-170). Both flags can coexist — cueq is disabled *before* the mutual-exclusion check (attention.py:392-418) — so the trunk gets cueq and everything else keeps Triton.
**Estimate:** 2–6 % clean gain, but worth 10-30 % versus the shipped yaml; confidence high for the trap, medium for the gain; cost ~1 h.

### 3.5 sm_120 tiling for the in-tree kernels
Hard-coded, autotune-free launch parameters tuned elsewhere: forward attention `BLOCK_SIZE_Q=64, BLOCK_SIZE_KV=16, num_warps=4, num_stages=1` (evoformer.py:1181-1182,1257-1258); Triton linear/TMM tiles `128×64×64` (triangular_multiplicative_update.py:470-472,541-543). No `get_device_capability()` branch exists; Blackwell is only handled via `TRITON_PTXAS_BLACKWELL_PATH` (pixi.toml:331-332).
**Estimate:** 1–4 % e2e (trunk-only effect); confidence medium; cost 2–3 days.

### 3.6 Throughput batching (multi-query sets only)
`batch_size: int = 1`, `num_workers: int = 0` (data_module.py:161-162); `predict_step` assumes bs=1 (runner.py:946) and featurization is serialized behind the GPU.
**Estimate:** 10–30 % on queries/hour, ~0 on single-query latency; confidence medium; cost 3–5 days.

### 3.7 Not a gain, a hazard
`docker/Dockerfile.pixi:93` pins `TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"`. Enabling `use_deepspeed_evo_attention` on sm_120 has no kernel image; the default eval path is unaffected (False, model_config.py:118).

## 4. Experiments

Per weight set: 3 fixed queries from `examples/example_inference_inputs` (small/medium/large), 5 diffusion samples, seeds 42/43/44, `pixi run -e openfold3-cuda13-pypi`.

1. Baseline: `run_openfold predict --query-json=q.json --runner-yaml=triton.yml` with `profiling.yml` merged → `logs/profiler_traces/*.json`; also `nsys profile --trace=cuda,nvtx` for kernel-level evidence.
2. bf16: add `trainer: {precision: bf16-mixed}`; record wall clock, `torch.cuda.max_memory_allocated`, LDDT/PAE versus fp32.
3. cueq: `use_cueq_triangle_kernels: true` **and** `use_triton_triangle_kernels: true`; confirm no `cueq_would_fall_back` on the trunk and no Triton disable in stderr.
4. sm_120 tiling: `OF3_TRITON_EXP2=1`, then a patched `num_warps/num_stages` sweep; check Triton cache hits and kernel names.
5. Graphs: wrap `_sample_rollout` in `torch.cuda.graphs` with a static buffer; assert bitwise-equal `xl` for one step versus eager.

Acceptance: real gain only if median wall clock over 3 seeds improves ≥10 %, median LDDT stays within 0.005 of the fp32 baseline, and peak memory does not rise. Smaller deltas inside run-to-run variance are noise.

## 5. Verdict

**Medium-high (≈65–70 %) that >10 % e2e is reachable; high if "throughput" means queries/hour over a set, medium (≈45 %) for single-query latency.** ~90 % of arithmetic sits in a 200-step Python loop run in fp32 eager mode with no graph capture and a per-step sync; bf16 alone plausibly clears 10 %, and CUDA graphs add more. Trunk-side kernel work (cueq, tiling) is real but small in e2e terms.

Cheapest settling experiment: the one-line `precision: bf16-mixed` A/B (§4.2), a few hours, bounds the whole estimate.

## 6. What would falsify this

- A profiler trace showing the diffusion loop already back-to-back on the GPU (no gaps >10 %), i.e. launch overhead is not the bottleneck.
- bf16 showing no wall-clock change (kernels already bandwidth-saturated at fp32).
- LDDT dropping >0.01 under bf16, which would make the main lever unusable without re-validation.
- Peak memory already near 16 GB, making batching on the 5070 Ti impossible and the throughput claim 5090-only.
