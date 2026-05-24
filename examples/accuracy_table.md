# 8-bit b-posit accuracy — measured, across real layers

W8A8 matmul `X @ Wᵀ` in 8-bit b-posit vs full precision, on real
**Qwen2.5-Coder-0.5B** weight matrices (a [128, 512] slice of each), `X` a fixed
Gaussian stand-in. Relative error = ‖Y_bp − Y_fp‖ / ‖Y_fp‖.

Two columns: **naive** (direct bp8 quantize, no scaling) and **scaled** (the
reproducibility-safe per-channel power-of-two recipe in
`reference/bposit_quantize.py`). Every result accumulates in the exact 256-bit
quire and is bit-reproducible across hardware.

| layer | naive bp8 | scaled bp8 | improvement |
|---|---|---|---|
| L0 attn q_proj  | 0.0953 | 0.0899 |  5.6% |
| L0 attn v_proj  | 0.1335 | 0.0911 | 31.8% |
| L0 mlp down_proj| 0.1305 | 0.0884 | 32.2% |
| L0 mlp gate_proj| 0.1255 | 0.0882 | 29.7% |
| L0 mlp up_proj  | 0.1333 | 0.0903 | 32.2% |
| L12 attn o_proj | 0.1333 | 0.0910 | 31.7% |
| L12 mlp gate_proj| 0.1259 | 0.0873 | 30.7% |
| **mean** | **0.1253** | **0.0895** | **28.6%** |

## How to read this honestly

- **~9% is the 8-bit floor.** The scaling recipe was found by an OpenEvolve
  search and the search could not push below ~9% with any reproducibility-safe
  (power-of-two) scaling. So 8-bit b-posit is **not** best-in-class low-bit
  accuracy — INT8/AWQ wins there. b-posit's niche is **bit-reproducibility**
  across GPU/CPU/RISC-V, which INT8/AWQ cannot give. For accuracy-grade work,
  the **16-bit rung is bf16-class and reproducible**.
- **Why power-of-two, not max-scaling.** Per-channel *max* scaling (the INT8
  trick) barely helps posits — posits are tapered, not uniform, so bounding the
  max doesn't place mass in the high-precision band. The win comes from
  **RMS-centering** each row near magnitude ~1.0 (where b-posit is densest), via
  an exact power-of-two shift that preserves reproducibility.
- **Verification.** Each row above passed three forensic gates (valid codes /
  exact-quire re-derivation / cross-run reproducibility) against the reference
  oracle, on layers the recipe did not train on. The recipe trained only on L0
  gate_proj; the other six are held-out.

## Lower rungs (bp4, aip5) — measured, and the error is high

Same 7 layers, same recipe, same forensic gates, but quantizing **both** operands
to the 4-bit (`bp4`) and 5-bit (`aip5`) rungs. Mean relerr of the scaled result:

| rung | bits | vs fp16 | mean relerr (scaled) |
|---|---|---|---|
| bp8  | 8 | 50% smaller | **0.090** |
| aip5 | 5 | 69% smaller | **0.325** |
| bp4  | 4 | 75% smaller | **0.512** |

Read this plainly: **bp4/aip5 are not drop-in low-bit weight formats.** At 32–51%
relative error, uniformly quantizing a layer to 4 or 5 bits is far too lossy for
general inference — the per-channel power-of-two scale helps a lot (bp4 naive runs
0.7–3.0 before scaling) but can't overcome having only 16/32 representable values.
Their purpose is the **dynamic/mixed-precision** scheme: spend 4/5 bits only on
error-tolerant tensors (or let a scheduler pick precision per task), keep bp8/bp16
where it matters. All rungs share the same exact quire, so a mixed-precision sum is
still bit-reproducible. The "75/69% smaller" memory wins are real; the accuracy
cost shown here is the honest other half.

Reproduce: `reference/bposit_quantize.py` (`quantize_w8a8`), or
`examples/hf_bposit_demo.py` for the single-model walkthrough. Rung sweep:
the `characterize_rungs` harness in the OpenEvolve workspace.
