# IntelliFold v2.0.4 on Blackwell sm_120 (RTX 5070 Ti, 16 GB): can engineering buy >10% e2e?

Inspection only; nothing was run. Gain figures are **estimates** from code and defaults.
The repo publishes no throughput numbers (README links only FoldBench accuracy plots).

## 1. Inspected

- https://github.com/IntelliGen-AI/IntelliFold, local clone `/home/dan/research/intellifold`,
  commit `bc84487aec48e0d33bff9f8ee6cff9013ea3ab00` (2026-10-10), `version = "2.0.4"`
  (`pyproject.toml:7`), branch `main` (no tag).
- Read: `README.md`, `docs/{usage,kernels}.md`, `intellifold/{cli,run_jax_inference,patches,parallel}.py`,
  `openfold/{model/*,utils/layer_norm/*,v2_flash_inference_config.py}`, `run_intellifold.py`,
  `third_party/alphafold3/{pyproject.toml,src/alphafold3/model/{model_config.py,network/*},docs/*}`.

Confirmed: two stacks — JAX AF3 engine vendored at `third_party/alphafold3` (`--flash cudnn|triton|xla`,
default `cudnn`: `intellifold/cli.py:95`, `run_jax_inference.py:364,448`) and a legacy PyTorch/OpenFold
path (`LAYERNORM_TYPE=fast_layernorm`, `USE_DEEPSPEED_EVO_ATTENTION`, `CUTLASS_PATH`: `docs/kernels.md:5-16`;
`openfold/model/primitives.py:37`). Pin `jax[cuda12]==0.9.1` (`third_party/alphafold3/pyproject.toml:21`).
No `Blackwell|sm_1xx|5090|5070|CUDA 13` string anywhere in `README.md`, `docs/`, or the vendored docs.

Verified outside the repo: `jax_cuda12_pjrt-0.9.1`'s `xla_cuda_plugin.so` contains 103
`sm_120` strings incl. `.target sm_120`, `-arch sm_120 -m 64`, `sm_120a`/`sm_120f`, and
`jax-cuda12-plugin==0.9.1` requires `nvidia-cudnn-cu12>=9.8` — so the cuda12 pin does **not** block or
cripple sm_120; `jax[cuda13]` also exists at 0.9.1 (cuDNN `>=9.12`), and JAX docs now recommend it
("We recommend migrating to the CUDA 13 wheels", docs.jax.dev/en/latest/installation.html).

## 2. Where the runtime goes

| Stage | Evidence | Bound on 16 GB sm_120 (estimate) |
|---|---|---|
| MSA/template search | `run_jax_inference.py:1000+`; docs/performance.md "Data Pipeline" | CPU/disk-bound; dominates wall clock unless MSA precomputed |
| Trunk: 12 pairformer blocks × 11 recycles (`--num_recycles 10`, 11 passes) at `c_z=512`, `c_m=256` | `patches.py:105-109`; `openfold/v2_flash_inference_config.py:40` | Compute-bound dense N² GEMMs; becomes **memory**-bound above ~1.2k tokens (pair tensor `N²×512×2 B` = 2.4 GB at N=1536) |
| Diffusion: 200 steps × 5 samples × 6 blocks, fused via `hk.scan(..., unroll=4)` | `diffusion_head.py:131,407`; `v2_flash_inference_config.py:49` | Compute/latency-bound, already single-dispatch batched |
| Confidence head / mmCIF | `heads.confidence` | negligible |

The only device gate is `compute_capability < 6.0` (`run_jax_inference.py:1016`).

## 3. Candidate optimizations

| # | Location | Change | Why sm_120 specifically | Gain e2e (est.) | Conf. | Cost |
|---|---|---|---|---|---|---|
| 1 | `run_jax_inference.py:352-357` | Tighten `--buckets` (128-step spacing below 1536) | Padding waste is squared in pair ops: 800 tokens → 1024 bucket = +64% pair work | 3–8% (up to 15% on dimer-ish 700–900-token inputs) | med | 0 (flag exists) |
| 2 | `third_party/alphafold3/pyproject.toml:21` | `jax[cuda13]==0.9.1` | Newer cuBLAS/cuDNN for sm_120; but 0.9.1 cuda12 already ships sm_120 code | 0–3% | med | 1 line + rebuild AF3 |
| 3 | `cli.py:95` / `run_jax_inference.py:364` | A/B `--flash triton` vs `cudnn` | cuDNN's sm_120 FMHA engines are new; tokamax just forwards to `jax.nn.dot_product_attention` (`tokamax/_src/ops/attention/jax_nn.py:137`) and tokamax skips cuDNN fwd+vjp on B200 (`benchmarks/attention.py:76-81`) — Blackwell cuDNN bugs are known upstream | −5…+8% | low | 0 (flag) |
| 4 | global | bf16 → fp8 | `bfloat16='all'` already (`model_config.py:34`) | ~0 viable: no fp8 attention path in jax 0.9.1 for biased DPA; accuracy risk | low | weeks |
| 5 | `openfold/utils/layer_norm/torch_ext_compile.py:22-57` | Fix `TORCH_CUDA_ARCH_LIST="7.0;8.0"` + hardcoded `-gencode` up to `sm_90` (SASS only, no PTX) | PyTorch path's fast layernorm has **no sm_120 image** → `no kernel image` at first call | PyTorch path only; 0 for JAX | high | 5 lines |
| 6 | `openfold/model/model.py:92-96` | `torch.compile`/CUDA-graph the 200-step eager sampler | Consumer Blackwell is fast enough that Python launch overhead dominates the PyTorch path | 10–30% **PyTorch path** | med | 1–2 days |
| 7 | `docs/kernels.md:15` | DS4Sci via CUTLASS `v3.5.1` | CUTLASS 3.5.1 predates Blackwell; sm_120 needs 3.8+ → path is a compile trap, not a win | 0 | high | — |

## 4. Experiments

Inputs: `examples/5S8I_A_wtemplate.yaml` (precomputed MSA, ~single chain) and one ~800-token dimer;
`--norun_data_pipeline`, `--seed 42`, `--num_diffusion_samples 5`, `--num_recycles 10`; the
compilation cache is on by default (`cli.py:166-172`).

1. **Stage split (decisive, 30 min).** `--num_diffusion_samples=1` vs `5` (flag exists,
   `run_jax_inference.py:382`; sample count scales diffusion linearly, so
   diffusion ≈ 1.25·(t₅−t₁)). For a true step sweep, edit `steps=200` at
   `diffusion_head.py:131` to 100. Metric: median of 3 wall clocks.
2. **Buckets.** Add `--buckets 256,384,512,640,768,896,1024,1152,1280,1408,1536,...`; same input.
3. **Attention kernel.** `--flash cudnn|triton|xla`; capture selection with
   `XLA_FLAGS=--xla_dump_to=/tmp/xla --xla_dump_hlo_as_text` and grep `__cudnn` / `triton`.
4. **CUDA 13.** Rebuild vendored AF3 with `jax[cuda13]`; rerun 1.
5. Peak memory: `nvidia-smi --query-gpu=memory.used --format=csv -lms 200`.

Acceptance: ≥10% lower median wall clock with bit-comparable structure (CA-RMSD < 0.1 Å vs baseline,
same seed, same sample count); otherwise reject.

## 5. Verdict

**Low** (est. 15–20%) that code/engineering changes alone deliver >10% end-to-end on the JAX path.
Reasons: the fast path is already taken (`cudnn` flash, `bfloat16='all'`, `hk.scan` diffusion,
batched samples, persistent compile cache); the dominant fixed cost is the 200-step × 5-sample
diffusion loop, which only accuracy knobs reduce; and on 16 GB the card's limit is *capacity*
(OOM above ~1.5–2k tokens), not kernel efficiency. The PyTorch front door can likely gain 10–30%, but that is
repairing a broken build, not tuning Blackwell. Cheapest decisive
experiment: the `--num_diffusion_samples 1` vs `5` delta in §4.1 — if diffusion is >70% of GPU time,
no trunk/kernel work can clear the bar.

## 6. What would falsify this

- The step-count sweep showing diffusion <40% of GPU time (trunk/bucketing then matters more).
- A profiler trace showing `cudnn` fused attention is *not* selected on sm_120 (silent XLA fallback),
  which would make candidate 3 a 1.5–3× attention win, not ±5%.
- `--flash xla` running far slower than `cudnn` while `--flash triton` beats `cudnn` by >15% on attention.
- MSA-dominated inputs where a tighter bucket moves >10% of wall clock.
- Any repo/issue statement of sm_120 timings — none exists.
