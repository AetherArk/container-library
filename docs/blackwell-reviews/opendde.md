# OpenDDE v1.2.0 on Blackwell sm_120 (RTX 5070 Ti, 16 GB): can engineering buy >10% e2e?

Inspection only; no run was performed. Every gain figure is an **estimate** from
code and defaults; the repo publishes no timings.

## 1. Inspected

- https://github.com/aurekaresearch/OpenDDE, clone `/home/dan/research/opendde`
- Commit `9d0d9c975178074c1ce722d2d6eefac8b7057062`, 2026-10-09, tag `v1.2.0`
- Read: `pyproject.toml`, `Dockerfile`, `README.md`, `CHANGELOG.md`,
  `docs/{kernels,inference_instructions,foldcp_e2e_baseline,tfg_constraint_guidance}.md`,
  `utils/environment.py`, `config/*.py`, `model/{opendde,generator}.py`,
  `model/modules/{primitives,transformer,diffusion,pairformer}.py`,
  `model/triangular/{triangular,layers}.py`, `model/layer_norm/*.py`,
  `distributed/foldcp/*.py`, `tfg/{potentials,rigid_contact}.py`, `runner/*.py`

Corrections to the brief: `@triton.jit` count is **8**, not 7
(`tfg/{group_contact:38,173,257, vina_steric_kernel:52,88, clash_field:39,
pose_search_kernel:19}`, `foldcp/triton_attention_bias.py:12`). No
`B300`/`roadmap`/`Blackwell`/`sm_100`/`CUDA 13` string exists in this checkout,
so the B300-roadmap claim is unsupported by v1.2.0 source.

## 2. Where the runtime goes

Defaults (`config/model_base.py:64,65,134`): `N_step=200`, `N_sample=5`,
`N_cycle=10`; the recycle loop runs 10× over 48 PairFormer blocks
(`opendde.py:1032`), then 1000 denoiser calls (`generator.py`). Two facts
dominate:

1. **Diffusion is FP32 by default.** `runner/inference.py:1500-1510` sets
   `skip_amp.sample_diffusion = True` for every input ≤3840 tokens;
   `opendde.py:1356` wraps the sampler in
   `autocasting_disable_decorator(...)`, which also casts all tensor args to
   `torch.float32` (`utils/torch_utils.py:172-179`). Only the trunk is BF16
   under `torch.autocast` (`runner/inference.py:1389-1396`).
2. **Torch-path attention is upcast to FP32 before SDPA.**
   `modules/primitives.py:291-296`: `q = q.to(dtype=torch.float32)` … then
   `F.scaled_dot_product_attention` at `:302`, with an FP32 bias mask. No
   fp16/bf16 flash path; no `torch.compile` anywhere in `opendde/` or `runner/`.

At N≈700 a bf16 pair tensor is ~0.4 GB, so `chunk_size=256` plus dynamic
thresholds (`opendde.py:1820-1840`) makes the trunk many small GEMMs →
launch/memory-bound. The diffusion sampler (1000 FP32 passes, 24 blocks,
`c_token 768`) is the compute-bound block where Blackwell tensor cores sit idle;
with TF32 off, nothing FP32 is compute-bound on sm_120.

## 3. Candidate optimizations

| # | Location | Change | Why sm_120 | Est. e2e gain | Conf. | Cost |
|---|---|---|---|---|---|---|
| A | `runner/inference.py:1500-1510` | `skip_amp.sample_diffusion=False` (reachable now: `OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP=1`, `:1514`) | 1000 BF16 tensor-core passes instead of FP32 CUDA-core work; halves activation traffic | **10–25%** | med-high | 1 env var; numerics change |
| B | `primitives.py:291-296` | Drop the FP32 upcast; bf16 SDPA, fp32 bias add | Restores tensor-core attention instead of FP32 SDPA math | 3–10% | med | ~30 lines, real numerics risk |
| C | `pyproject.toml:39,60-64`; `Dockerfile:1,48` | torch 2.11 + cu13 (or cu128), Triton ≥3.4 | cu126 wheels cannot contain sm_120 SASS; kernels JIT from compute_90 PTX at best. Native Blackwell codegen, compilable `fast_layernorm` | 5–20% | low-med | new image + cueq cu13 pin |
| D | `batch_inference.py:773` | `--enable_tf32 true` on the FP32 diffusion | TF32 tensor cores for those GEMMs, no code change | 5–15% | med | free, but silently disables TFG Triton kernels (`potentials.py:1424-1440`) and breaks guidance parity |
| E | `environment.py:284` (`if major < 8`) | Arch allow-list, not a lower bound | `auto` cannot tell sm_90 from sm_120; a cu12 cueq kernel without sm_120 SASS raises or silently degrades | guard | high | 1 branch, or pin `--trimul/--triatt torch` |
| F | `opendde.py:1867-1873,1887-1908` | Trunk cache | Key hashes **every** non-guidance feature, so any ligand change misses; Fold-CP/multi-seed bypass | ~0% default | high | none |
| G | `tfg/*` | Guidance kernels | Off unless `--use_tfg_guidance`; `triton_available()` (`rigid_contact.py:34-40`) only checks `find_spec`, so sm_120 attempts Triton, then falls back with one warning (`potentials.py:1429-1444`) | 0% default | high | none |
| H | `layer_norm/layer_norm.py:278`, `torch_ext_compile.py:31` | CUDA LayerNorm | Default is `LAYERNORM_TYPE=torch` (`layers.py:25`); it also compiles with `-maxrregcount=32`, which throttles it | 1–2% | low | nvcc 12.6 can't target sm_120 |

`auto` on sm_120 is risky (E), not merely suboptimal: the gate is `major < 8`
with a CC-7 fallback, no upper bound or known-arch list, and
`auto_triangle_kernel` returns `"cuequivariance"` whenever imports succeed
(`environment.py:97-99`). cueq is then really taken, because `hidden_scale_up`
forces `c_hidden_mul = c_z` (`pairformer.py:213-215`).

## 4. Experiments

```bash
export OPENDDE_ROOT_DIR=/data/opendde_data
opendde pred -i examples/cofold/input.json -o out_base -n opendde_v1 -s 42 \
  --use_msa false --use_template false --use_rna_msa false \
  --sample 5 --step 200 --cycle 10 --dtype bf16 \
  --triatt_kernel auto --trimul_kernel auto --enable_tf32 false \
  --foldcp_metrics_jsonl out_base/metrics.jsonl
# variant: OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP=1 ... -o out_amp
```

Same line with `-e 1` isolates the sampler; repeat on
`examples/cofold/protein_200.json` and one ~700-token antigen case. Three runs
per arm, median. Seeds fixed at 42.

Metrics: wall clock (`/usr/bin/time -v`); per-stage `elapsed_ms` and
`total_peak_mib` from `--foldcp_metrics_jsonl`
(`distributed/foldcp/metrics.py:140-200`); dispatch evidence from
`torch.profiler` as `tests/smoke/test_cueq_runtime.py:480-500` does. Acceptance:
median wall clock ≥10% better in 2 of 3 input sizes, identical schema, finite
coordinates, sample RMSD vs baseline ≤0.5 Å across all 5 samples, mean pLDDT
within ±1.0. Same protocol for D; for C run `opendde doctor` first.

## 5. Verdict

**High** likelihood of >10% e2e. The defaults leave the most expensive stage
(1000 FP32 denoiser passes) and all attention (FP32 upcast) off the tensor
cores, and both switches already exist in-tree — A and D are single-flag tests.
Confidence that A–D *compose* to 30%+ is much lower: chunking, launch overhead
and the 16 GB ceiling eat part of it.

Cheapest decisive experiment: `OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP=1` with
`-e 5 -p 200 -c 10`, seed 42, 3 runs, median wall clock + `total_peak_mib`.
One env var, no code change; it either clears 10% or proves the FP32 diffusion
default is not the story.

## 6. What would falsify this

- `test_cueq_runtime.py --preset opendde_v1 --tokens 256` showing the trunk is
  already ~90% of wall clock — then A cannot reach 10%.
- Profiler showing FP32 diffusion already on TF32/bf16 tensor ops, collapsing
  A and D to ~0.
- cu126 refusing to launch on sm_120: C is then a prerequisite, not a gain.
- Sample RMSD >0.5 Å or pLDDT drop >1.0 → verdict becomes **low**.
- `elapsed_ms` showing >40% of the run outside trunk/diffusion/confidence
  (MSA, featurisation, dumping) caps model-side gains below 10% e2e.
