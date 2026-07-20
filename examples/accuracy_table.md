# 8-bit b-posit accuracy — measured, across real layers

W8A8 matmul `X @ Wᵀ` in 8-bit b-posit vs full precision, on real
**Qwen2.5-Coder-0.5B** weight matrices (a [128, 512] slice of each), `X` a fixed
Gaussian stand-in. Relative error = ‖Y_bp − Y_fp‖ / ‖Y_fp‖.

Two scaling recipes — **naive** (direct quantize) and **scaled** (the
reproducibility-safe per-channel power-of-two recipe in
`reference/bposit_quantize.py`) — and, the decisive axis, two **rounding modes**:
**truncate-toward-zero** (b-posit's `encode` default) vs **round-to-nearest** (the
posit standard, `bposit_fast.quantize_bp8(x, nearest=True)`). Every result
accumulates in the exact 256-bit quire and is bit-reproducible across hardware in
**both** rounding modes.

| layer | naive bp8 (trunc) | scaled bp8 **truncate** | scaled bp8 **round-to-nearest** |
|---|---|---|---|
| L0 attn q_proj  | 0.0953 | 0.0917 | **0.0354** |
| L0 attn v_proj  | 0.1335 | 0.0911 | **0.0382** |
| L0 mlp down_proj| 0.1305 | 0.0898 | **0.0384** |
| L0 mlp gate_proj| 0.1255 | 0.0900 | **0.0372** |
| L0 mlp up_proj  | 0.1333 | 0.0894 | **0.0384** |
| L12 attn o_proj | 0.1333 | 0.0892 | **0.0377** |
| L12 mlp gate_proj| 0.1259 | 0.0894 | **0.0360** |
| **mean** | **0.1253** | **0.0901** | **0.0373** |

## How to read this honestly

- **The "~9% floor" was a truncation artifact, not a real floor.** b-posit's naive
  `encode` rounds *toward zero* — a biased rounding that shrinks every value.
  Rounding to the **nearest** representable posit (the standard, unbiased choice)
  **more than halves** the per-layer error (9.0% → 3.7% mean), and the gap is far
  larger end-to-end: full-model WikiText-2 perplexity is **near-lossless** with
  round-to-nearest (SmolLM2-135M **+3.1%**, Qwen2.5-Coder-0.5B **+0.68%**) vs
  +13–20% under truncation, because the truncation bias compounds through depth
  (see the repo README + `examples/w8a8_rounding_demo.py`). So 8-bit b-posit is
  **reproducible AND competitively accurate** — the combination INT8/AWQ can't
  give (they get accuracy but not bit-identical-across-hardware). Round-to-nearest
  stays fully deterministic, so this costs nothing in reproducibility. For
  accuracy-grade work the **16-bit rung is bf16-class and reproducible**.
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

Per-layer detail (same 7 layers, **naive** = direct quantize, **scaled** = the
power-of-two recipe). This is the breakdown behind the means above:

| layer | aip5 naive | aip5 scaled | bp4 naive | bp4 scaled |
|---|---|---|---|---|
| L0 attn q_proj   | 0.4536 | 0.3604 | 0.7195 | 0.5323 |
| L0 attn v_proj   | 0.7188 | 0.3241 | 3.0210 | 0.5168 |
| L0 mlp down_proj | 0.6037 | 0.3210 | 1.7900 | 0.5068 |
| L0 mlp gate_proj | 0.5900 | 0.3166 | 1.2886 | 0.5101 |
| L0 mlp up_proj   | 0.6093 | 0.3169 | 1.6484 | 0.5017 |
| L12 attn o_proj  | 0.6547 | 0.3205 | 1.8830 | 0.5108 |
| L12 mlp gate_proj| 0.5948 | 0.3120 | 1.5353 | 0.5049 |
| **mean** | **0.6036** | **0.3245** | **1.6979** | **0.5119** |

Two things jump out. **(1)** Scaling matters far more at the low rungs than at bp8:
bp4 naive ranges 0.72–3.02 (one outlier channel-set blows up to 3×) but the
power-of-two RMS-centering pulls *every* layer into a tight 0.50–0.53 band — it's
salvaging codes that naive quantization throws into saturation. **(2)** Even fully
scaled, the floor is set by representable-value count, not by tuning: ~0.51 (bp4,
16 values) and ~0.32 (aip5, 32 values) are as good as a uniform per-layer cast
gets. That gap between the rungs is exactly why the mixed/dynamic-precision scheme
exists — you don't pay the bp4 floor on a tensor that can't tolerate it.

Read this plainly: **bp4/aip5 are not drop-in low-bit weight formats.** At 32–51%
relative error, uniformly quantizing a layer to 4 or 5 bits is far too lossy for
general inference — the per-channel power-of-two scale helps a lot (bp4 naive runs
0.7–3.0 before scaling) but can't overcome having only 16/32 representable values.
Their purpose is the **dynamic/mixed-precision** scheme: spend 4/5 bits only on
error-tolerant tensors (or let a scheduler pick precision per task), keep bp8/bp16
where it matters. All rungs share the same exact quire, so a mixed-precision sum is
still bit-reproducible. The "75/69% smaller" memory wins are real; the accuracy
cost shown here is the honest other half.

## Mixed precision in one exact quire (still reproducible)

You can quantize different weight channels to different rungs and accumulate the
mixed-rung dot product in the **same** 256-bit quire — the result stays
bit-identical across hardware, which uniform float/INT8 cannot do. Policy: bp8 for
the highest-output-energy channels, aip5 for the rest; activations kept at bp8.
On L0 mlp gate_proj (`examples/mixed_precision_demo.py`):

| policy | avg bits/weight | relerr | reproducible |
|---|---|---|---|
| uniform aip5 (W) | 5.00 | 0.220 | ✓ |
| **mixed (25% bp8 / 75% aip5)** | **5.75** | **0.172** | ✓ |
| uniform bp8 (W) | 8.00 | 0.088 | ✓ |

Honest read: mixed precision **interpolates** between the rungs — at 5.75 bits
(28% smaller weights than bp8) it beats uniform aip5 for +0.75 bits, but it's no
free lunch; bp8 is still much more accurate. The point of the demo is the
**property**: mixing rungs in one exact quire does not break reproducibility — the
basis for a dynamic-precision scheme that picks bit-width per channel/task and
still produces auditable, hardware-independent bits.

Reproduce: `python examples/rung_sweep.py` (prints this bp4/aip5/bp8 table from
scratch — real HF layer if available, else synthetic), `reference/bposit_quantize.py`
(`quantize_w8a8`) for the recipe, or `examples/hf_bposit_demo.py` for the
single-model W8A8 walkthrough.
