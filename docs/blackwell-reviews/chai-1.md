# Chai-1 (chaidiscovery/chai-lab 0.6.1) — Blackwell sm_120 throughput review

## 1. Inspected

- Repo: https://github.com/chaidiscovery/chai-lab, commit `c691da74d6f348b19bc9f931abe95d66d0b88e78`, 2026-10-26, `__version__ = "0.6.1"` (`chai_lab/__init__.py:5`). Local clone `/home/dan/cofold_research/chai-lab`; no `downloads/`, weights absent.
- Read: `chai_lab/chai1.py` (1059 L), `data/dataset/embeddings/esm.py`, `model/utils.py`, `data/collate/{collate,utils}.py`, `data/dataset/all_atom_feature_context.py`, `main.py`, `README.md`, `Dockerfile.chailab`, `requirements.in`.

Confirmed: weights are TorchScript artifacts pulled from chaiassets.com (`utils/paths.py:63-80`), loaded via `torch.jit.load` (`chai1.py:145,148`); the only `model/` files are `diffusion_schedules.py` and `utils.py` — no attention, trunk, or diffusion `nn.Module` in the repo. No `triton`, `flash-attn`, `cuequivariance`, `transformer_engine`, `fp8`, `torch.compile`, `xformers`, or `torch.backends` call anywhere (grep returns nothing). README:21 asks for "A100 80GB or H100 80GB or L40S 48GB"; no Blackwell statement, no published throughput.

## 2. Where the runtime goes

| Stage | Evidence | Cost model (estimate) |
|---|---|---|
| ESM-2 3B, fp16 | `esm.py:21`; serial per-sequence loop `esm.py:117-131`; moved GPU→CPU each call `esm.py:48,51` | ~6 GB weights uploaded/downloaded per run; forwards unbatched |
| Feature + token embedder | `chai1.py:679-738` | one pass, moderate |
| Trunk ×3 recycles | `chai1.py:748-777`; MSA depth cap 16384 (`all_atom_feature_context.py:20`) | MSA-attention heavy, compute-bound |
| Diffusion | `chai1.py:845-885`; `second_order=True` (`chai1.py:248`) ⇒ **two** denoiser calls per step, 200 default steps (`chai1.py:514`) ≈ 399 forwards at `ds=5` | 60–80% of GPU time (estimate, unmeasured) |
| Confidence head | `chai1.py:894-910`, loop over 5 samples | pairwise over tokens, 5 launches |
| CIF writing/ranking | `chai1.py:975-1050` | CPU-side, negligible |

On 16 GB sm_120: `low_memory=True` is default (`chai1.py:519`), and `_component_moved_to` unconditionally ends with `component.jit_module.to("cpu")` (`chai1.py:164-166`) even under `--no-low-memory`, so the trunk is re-uploaded on each of 3 recycles. The 16 GB regime is therefore partly **PCIe/allocation-bound**, not FLOP-bound; the diffusion loop is compute-bound over masked padded atoms (`n_atoms = 23 * n_tokens`, `collate/utils.py:36`).

## 3. Candidate optimizations

| # | Location | Change | Why sm_120 16 GB | Gain (est.) | Confidence | Cost |
|---|---|---|---|---|---|---|
| 1 | `chai1.py:164-166` | Keep the trunk resident across recycles; make offload conditional on `low_memory` | removes 2 redundant multi-GB H2D copies per fold | 2–8% (larger on small crops) | medium | 20 LOC |
| 2 | `chai1.py:607,780,888` | Drop `torch.cuda.empty_cache()` between stages; set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | empty_cache frees the pool, forcing fresh `cudaMalloc` right before the 399-step loop | 1–5% | medium | 3 lines / env |
| 3 | `chai1.py:140` | `set_fusion_strategy([("STATIC",3),("DYNAMIC",2)])` (NVFuser on) | sm_120 pointwise chains fuse well; currently zero fusion | 0–15% | low (disabled deliberately; risk of wrong results) | 1 line |
| 4 | process entry | Run N FASTAs in one process; `_component_cache` (`chai1.py:151`) amortizes `torch.jit.load` + CUDA module init | avoids full cold start per job | 5–20% *pipeline* throughput, ~0 single-fold | high | wrapper script |
| 5 | `esm.py:117-131` | Cache ESM embeddings by sequence across jobs; batch sequences | 3B load is the largest fixed cost on 16 GB | 3–15% for shared-chain campaigns | high | 30 LOC |
| 6 | `chai1.py:894-910` | One confidence-head call, batch dim 5 | one launch sequence instead of five | 1–4%; may OOM at crop 1536 | low | 5 LOC + probe |
| 7 | global | `torch.backends.cuda.matmul.allow_tf32 = True` | interface tensors are cast `.float()` (`chai1.py:789-816`); helps only if fp32 GEMMs survive inside the artifact | 0–6% | low | 1 line |

**Cannot be attempted without model source:** FlashAttention-3/cuequivariance swap, FP8/int8 weight quantisation, tensor parallelism, a 640/1280 crop bucket, trunk/diffusion depth changes, noise-schedule changes, CUDA-graph capture of artifact internals, fixing the 23× atom padding, or any kernel-level work.

## 4. Experiments

Baseline, seed fixed, profiler for the per-stage split:

```bash
cd /home/dan/cofold_research/chai-lab && python -c "from chai_lab.chai1 import run_inference as r; \
r(fasta_file='examples/example.fasta', output_dir='/tmp/b0', seed=42, num_diffn_timesteps=200, \
num_diffn_samples=5, num_trunk_recycles=3, use_esm_embeddings=True)"
```
Repeat under `torch.profiler` (`schedule(wait=1,warmup=1,active=3)`). Variants: (a) lever 2 env+patch, (b) lever 1, (c) lever 3, (d) 8-job loop, cold vs warm process. Metric: median wall-clock per fold over 3 reps, diffusion seconds separate. Acceptance: only ≥5% median gain with CIF coordinates identical at fixed seed (≤1e-3 Å); coordinate drift is a quality change and is excluded.

## 5. Verdict

**Low-to-medium (~30%) for >10% on single-fold latency; medium (~55%) for >10% on campaign throughput** (levers 4+5). The hot loop is 399 calls into an opaque, unfused TorchScript graph whose shapes and kernels we cannot touch; reachable levers are host-device traffic and process reuse — fixed costs, not per-step costs. Cheapest decisive test: lever 4's 8-job loop, cold vs warm process, plus lever 2's allocator env. One hour, no model edits.

## 6. What would falsify this

A profiler trace with diffusion <50% of wall clock (fixed costs dominant ⇒ larger upside), or fusion-on giving identical output >10% faster (lever 3 is the only per-step lever). If ESM exceeds 25% of a small-complex run, levers 4/5 already clear 10% and the verdict moves to high.
