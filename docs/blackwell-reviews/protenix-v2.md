# Protenix v2 on Blackwell sm_120 (RTX 5070 Ti, 16 GB, CUDA 13) — code-inspection review

Question: likelihood of >10% end-to-end gain from engineering changes, judged from code alone.

## 1. Inspected

- `https://github.com/bytedance/Protenix`, branch `main`, commit `bd54a05d047b8925a241056f36d619700604068a`, 2026-04-13 (+0800), `v2.0.0-3-gbd54a05`; working tree **dirty** (5 modified files), so upstream files are cited from `HEAD` and local edits flagged.
- Read: `requirements.txt`, `configs/*.py`, `protenix/model/{protenix,triangular/layers,triangular/triangular}.py`, `model/tri_attention/*`, `model/layer_norm/{layer_norm,torch_ext_compile}.py` + CUDA kernel, `model/modules/{pairformer,diffusion}.py`, `runner/inference.py`, `docs/training_inference_instructions.md`, `CHANGELOG.md`, `Dockerfile`.
- This host: `RTX 5070 Ti, 16303 MiB, 12.0`; `nvcc` is **12.0** and `--list-gpu-arch` stops at `compute_90`.

## 2. Where the runtime goes

`protenix-v2` is `c_z: 256`, `diffusion_batch_size: 64`, `N_cycle: 10` (`configs/configs_model_type.py:52-56`), ~464 M params. It is hard-capped at 2560 tokens: `if n_token > 2560 and configs.model_name in ["protenix-v2"]: raise AssertionError("...does not support n_token > 2560...")` (`runner/inference.py:397-400`). Published peaks of 18.2 GB @1000 tokens and 66.6 GB @2000 (`docs/training_inference_instructions.md:263-267`) put a 16 GB card at roughly ≤800 tokens (estimate).

Stage timers (`pairformer`, `diffusion`, `confidence`) are already logged at `protenix.py:526,578,607`. Structurally, 10 trunk passes × 48 pairformer blocks dominate; diffusion runs `N_step=200 × N_sample=5` denoiser calls in **fp32** (`configs_base.py:181-182`; autocast disabled at `protenix.py:341`, inputs upcast at `modules/diffusion.py:471-472`); the v2 confidence head also runs outside autocast (`runner/inference.py:409-410`).

Bound classification (estimate, not measured): at N≈800–1000 with c_z=256, triangle ops rewrite `[N,N,256]` bf16 intermediates (~0.4–0.5 GB each), so trimul/triatt are **bandwidth-bound**; diffusion is **launch/bandwidth-bound**; only trunk `Linear` projections are compute-bound.

## 3. Candidate optimizations

Inference defaults already take the cheap wins: `enable_tf32: True`, `enable_efficient_fusion: True`, `enable_diffusion_shared_vars_cache: True` (`configs_inference.py:32-34`), `allow_tf32` applied at `protenix.py:99`, both triangle ops `cuequivariance` (`configs_base.py:129-130`). Flipping TF32/fusion/caching is a **no-op** at HEAD.

| # | Location | Change | Why sm_120 | Est. e2e gain | Conf. | Cost |
|---|---|---|---|---|---|---|
| 1 | `requirements.txt:4-5` cu12 pin | use cu13 wheels | fast path is documented only for compute caps **10.0 and 10.3** (`triangular/layers.py:458-461`), i.e. B200/GB300, not 12.0 | 0–4% | med | low |
| 2 | `tri_attention/autotune_helpers.py:64-114` | add sm_120 configs | config lists are arch-agnostic, "~ok for things <= 512"; cache file is per-capability (`autotune.py:94`) so sm_120 retunes over a poor space | 0–3% | low | low |
| 3 | `layer_norm/torch_ext_compile.py:51-58` | add `("120","120")` | upstream `_wanted` stops at `("100","100")`; `_arch_list` (line 67) filters on `nvcc --list-gpu-arch` | 1–3%, also fixes a load failure | high mechanism / low size | low |
| 4 | `configs_inference.py:32-34` | already on | — | ~0% | high | none |
| 5 | `protenix.py:341`, `diffusion.py:454-472` | bf16 diffusion | 200×5 fp32/TF32 steps | 8–15%, but an **accuracy** change | med | med |
| 6 | `triangular/layers.py:438-439` | `if q.shape[-2] <= 16: triangle_attention = "torch"` | tiny crops only | ~0% | high | none |

Two traps. `torch_ext_compile.py:87` uses `-maxrregcount=32` (spills in a bandwidth-bound kernel), and `layer_norm.py:33` sets `fastln_is_installed` from the env var alone, so a build without an sm_120 image fails at **import** rather than falling back — workaround `LAYERNORM_TYPE=torch`. Trimul only takes cuequivariance when `c_z == c_hidden` (`triangular.py:494`), which v2 satisfies.

## 4. Experiments

Baseline, 3 repeats, one seed:

```
cd /home/dan/containers/protein_structure/protenix/Protenix; export PYTHONPATH=$PWD
/usr/bin/time -v python runner/inference.py --model_name protenix-v2 \
  --input_json_path examples/example.json --dump_dir out/base --seeds 101 \
  --model.N_cycle 10 --sample_diffusion.N_step 200 --sample_diffusion.N_sample 5 \
  --triangle_attention cuequivariance --triangle_multiplicative cuequivariance \
  2>&1 | tee out/base.log
```

Metrics: per-stage `time_tracker` lines; wall clock; `nvidia-smi --query-gpu=memory.used --format=csv -lms 200` peak; dispatch evidence via `TRITON_PRINT_AUTOTUNING=1` plus a `torch.profiler` trace filtered to `triangle_attention` / `triangle_multiplicative_update` (a 10.x kernel name proves path 1 fired; a generic name falsifies it).

Ablations, one variable each: `LAYERNORM_TYPE=torch`; `--triangle_attention triattention`; `-c 4` to size the trunk share. Microbench (30 min) `triangle_attention` at N∈{512,1024}, bf16, `torch.cuda.Event` timing, before/after the cu13 pin.

Acceptance: median-of-3 wall clock ≥10% better, RMSD <0.01 Å vs baseline, no stage regressing >5%. Kernel work is falsified if the profiler shows the same kernel name on both sides.

## 5. Verdict

**Low** for the kernel/config list alone; **medium (~40%)** for a bundle that adds `torch.compile`/inductor around the diffusion block loop, which is not compiled today (`torch.compile` appears only in docstrings and the LayerNorm JIT).

Reasoning: defaults already select cuequivariance + TF32 + fusion + diffusion cache, so remaining headroom is "good" vs "optimal", not "slow" vs "fast"; the Blackwell cuequivariance kernels are documented for 10.0/10.3, not 12.0, so a cu13 pin probably does not specialise sm_120; and at ≤800 tokens work is bandwidth/launch-bound, where 3–5% is the realistic kernel ceiling. Cheapest decisive test: the §4 microbench + profiler pair — identical kernel names before/after the cu13 pin kills the kernel thesis.

Weights caveat: `protenix-v2.pt` returns `HTTP/1.1 403 Forbidden` while `protenix_base_default_v1.0.0.pt` under the same bucket prefix returns `200 OK`, so v2 parameters are object-level withheld. Reviewable: every architectural/dispatch/config path above plus v1.0.0/v0.5.0 checkpoints. Not reviewable: any v2 timing, memory or dispatch measurement.

The 676 s → 226 s (2000 tokens, v0.6.3→v0.7.0) claim is **not reproducible here**: `git grep 676` on `v1.0.0`, `v1.0.5`, `v2.0.0` returns nothing; only 226 s is in-tree (line 267). Treat 3× as an external release note.

## 6. What would falsify the estimate

- A trace on any supported checkpoint with triangle kernels >50% of wall clock.
- cu13 dispatch naming an sm_120/12.0-specialised kernel (promotes item 1 from 0–4% to real).
- Diffusion ≥45% of wall clock at 16 GB-feasible sizes (makes item 5 decisive).
- A split with `pairformer` ≥60% and triangle ops ≤20% at ≤800 tokens — confirming the low verdict.
