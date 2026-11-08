# RoseTTAFold3 (models/rf3) — sm_120 throughput review

**Question:** how likely is >10% end-to-end throughput on RTX 5070 Ti 16 GB / CUDA 13 (sm_120) from engineering changes, judged from code alone?
**Answer:** medium (~40–50%). Nothing was measured; flagged figures are estimates.

## 1. Inspected

- Repo: https://github.com/RosettaCommons/foundry, commit `0d0b0ba44fab7ca76aa4b625c014ce5964c7d284`, 2026-11-04.
- Read: `pyproject.toml`, `_core/__init__.py`, `_core/model/layers/attention.py`, `model/layers/{attention,af3_diffusion_transformer,pairformer_layers,inference_kernels}.py`, `model/kernels/{exact_layer_norm.py,exactln_fwd.cu}`, `model/RF3{,_structure}.py`, `diffusion_samplers/inference_sampler.py`, `inference_engines/rf3.py`, configs `inference_engine/rf3.yaml`/`model/components/rf3_net.yaml`/`trainer/rf3.yaml`, tests `test_{attention,dense_sdpa,triangle_sdpa,exact_kernels}.py`.
- Grep-confirmed absent: `triton`, `flash_attn`, `tensorrt`, `fp8`, `torch.compile`, `sm_100`, `sm_120`, `B200`, `Blackwell`.

## 2. Where the runtime goes

Defaults: `n_recycles: 10`, `num_steps: 50`, `diffusion_batch_size: 5` (`configs/inference_engine/rf3.yaml:12,16,13`); 48 pairformer blocks (`rf3_net.yaml:48`); diffusion `n_block: 24` (`:146`); atom encoder/decoder 3 blocks each, 32-query/128-key windows (`:34,:156`). One prediction = ~480 pairformer blocks (`c_z=128`) + 50 × 24 token blocks (`c_token=768`) + 6 sparse atom blocks, under `precision: bf16-mixed` (`configs/trainer/rf3.yaml:20`).

- cuEquivariance gate: `SHOULD_USE_CUEQUIVARIANCE` set only if CUDA is available and `DISABLE_CUEQUIVARIANCE` unset, guarded by `except ImportError` alone (`_core/__init__.py:31-50`). Pins `cuequivariance_ops_cu12>=0.6.1` etc. (`pyproject.toml:68-70`). bf16-only: `assert pair.dtype == torch.bfloat16` (`model/layers/attention.py:154-156`, `:340-342`).
- Triangle ops route to `cuet.triangle_attention` / `triangle_multiplicative_update` when `attention_backend=="auto"` (`attention.py:97-102, 170; 284`), else to einsum (`:193-197`).
- **Dense attention defaults to the *unfused* path on CUDA**: `use_dense_sdpa` is true only for `backend=="sdpa"` or `auto`+MPS (`_core/model/layers/attention.py:13-17`), so `auto` on CUDA runs `einsum → +bias → softmax → einsum` (`af3_diffusion_transformer.py:469-478`, `pairformer_layers.py:287`), materialising `[B,I,I,H]`.
- Sparse atom attention is hand-written blockwise gather + softmax with `qbatch=32, kbatch=128` hardcoded (`af3_diffusion_transformer.py:510-566`), gathering `Z_local` then `ln_0`+`to_b` (`:547-548`).
- Exact LayerNorm is gated on `torch 2.11.0`, `torch.version.cuda == "12.8"`, `"H200" in device name`, plus ≥131072 rows (`inference_kernels.py:17-25, 51-56`).

**Compute vs memory bound (estimates).** At L≈200–256 a pairformer block is ~25–30 GFLOP of *small* matmuls (K=N=128); ~0.9 TB/s against ~150–200 TFLOP bf16 puts break-even near 200 FLOP/byte, and these tiles sit at ~150–250 — mixed, maybe 25–40% of peak. The diffusion atom path is the reverse: its `Z_local` gather is ≈67 MB per layer per step at 2000 atoms, ≈20 GB per design. Expect trunk ≈50–65%, diffusion ≈35–50%; measure first. IPD/NVIDIA applied TensorRT **only to the Pairformer trunk**, "work on the diffusion module is ongoing"; their published 5.8 s → 2.5 s at 256 residues used cuEquivariance plus TensorRT, neither in this repo.

## 3. Candidate optimizations

| # | Location | Change | Why sm_120 | Gain (est.) | Conf. | Cost |
|---|---|---|---|---|---|---|
| A | `pyproject.toml:68-70` | pin `cuequivariance*>=0.11` | v0.11.0 notes: "Native sm120f Triangle Attention kernel for RTX 6000 Pro Blackwell, replacing the generic kernel used on that SKU in 0.10"; 0.11 needs PyTorch 2.11.x, already pinned here | 10–30% | med | med (`attention_pair_bias` API break; revalidate numerics) |
| B | `_core/model/layers/attention.py:13-17` | SDPA on CUDA via `dense_attention_backend=sdpa` | drops materialised `[B,I,I,16]` + separate softmax; fused kernels exist for sm_120 in torch ≥2.9 | 2–6% | med | ~zero (flag) |
| C | `inference_kernels.py:17-25` | widen H200 gate to sm_120 | kernel already JITs per arch: `--gpu-architecture=sm_{cc}` (`exact_layer_norm.py:81`) | 1–3% | low | high: gate guards *bitwise* ATen replication (`exactln_fwd.cu:240`), so widening spends exactness on noise |
| D | `inference_sampler.py:148-186` | CUDA-graph / `torch.compile` the 50 identical-shape steps | ≈14k launches, no host syncs in loop, Python-heavy per block | 5–12% | med-high | med (shape bucketing) |
| E | `af3_diffusion_transformer.py:510` | tune `qbatch/kbatch` (64/128, 32/256) | gather volume vs tile efficiency trades against 0.9 TB/s | 2–5% of diffusion | low | low |
| F | `_core/__init__.py:41-46` | `CUEQ_DISABLE_AOT_TUNING=0` | tuning off by default; no AOT cache exists for sm_120 | 2–8% of cueq time | low-med | low |

Not engineering but dominant: `n_recycles 10→6` cuts ~40% of trunk blocks, `num_steps 50→30` ~40% of diffusion — the only route to 2×, and it changes outputs.

## 4. Experiments

Baseline (shipped `5vht` example, L≈200; repeat at 256 residues). Log the dispatch that actually happens, then time three runs:

```bash
python -c "import torch,rf3._core as c;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(),c.SHOULD_USE_CUEQUIVARIANCE)"
for i in 1 2 3; do /usr/bin/time -f "%e s" rf3 fold inputs=models/rf3/tests/data/5vht_from_json.json \
  out_dir=/tmp/rf3/base$i seed=0 ckpt_path=$CKPT 2>&1 | tail -1; done
```
Variants, same seed/input: (B) `triangle_attention_backend=sdpa dense_attention_backend=sdpa`; (A/C/E/F) patches; (D) `torch.compile(diffusion_module)` (`RF3_structure.py:106`, `no_grad`). Split evidence: `nsys profile -t cuda,nvtx rf3 fold ...`, plus `torch.profiler` around `engine.run` → `key_averages().table(sort_by="cuda_time_total", row_limit=15)`.
Metrics: median of 3 runs after 2 warmups (CV <5%), `max_memory_allocated`, top-15 kernel table (cuEquivariance vs `flash` vs `einsum`), NVTX trunk/diffusion ms. Accept at ≥7% median gain, peak <16 GB, RMSD <0.05 Å, pLDDT Δ<0.5.

## 5. Verdict

**Medium, ~40–50%** for >10% from public-repo engineering; ~70% if flags, a 0.11 pin and a graph-captured diffusion loop all land; low (<10%) for 2× without TensorRT-class trunk compilation. The trunk's geometry ops already have a fused kernel, so its ceiling is that kernel's *version*, not its existence — while the diffusion module (35–50% of time) is eager, unfused, launch-heavy and unoptimised by IPD. Cheapest decisive experiment: 3-run baseline vs `DISABLE_CUEQUIVARIANCE=1` vs `sdpa` flags plus a profiler table — ~30 minutes, and it shows whether cuEquivariance is engaged on sm_120 at all and how much GPU time it holds.

## 6. What would falsify this

- cuEquivariance kernels already dominate GPU time at high occupancy → trunk headroom collapses; verdict → low.
- Diffusion share <25% → D/E shrink to noise; pipeline/MSA/IO >30% → GPU work cannot reach 10%.
- `sdpa` on sm_120 dispatches to the non-fused math backend → B is worthless.
- Cutting ~20% of launches moves wall clock <2% → the launch-overhead thesis behind D is false.
