<!-- Copyright (c) 2026 Anomly, Inc. SPDX-License-Identifier: Apache-2.0 -->

# §6 Evaluation — reproduction transcripts

## §6.3 Accuracy vs k — k=64 1.0×, k=256 1.5×, k=1024 3.5×

Measured on: physical Blackhole p150b (on-device b-posit16 engine vs exact quire; quire→b-posit32 readout, seq=8 out=8, 4 seeds/k). Producer: iter4b on-device engine-vs-quire rig (kernels public in open-bposit; the accuracy-sweep rig not published).

```
| k (dot len) | bposit16+engine | bposit16+EXACT quire | quire advantage |
| 64          | 3.372e-04       | 3.328e-04            | 1.0x lower      |
| 256         | 5.073e-04       | 3.386e-04            | 1.5x lower      |
| 1024        | 1.126e-03       | 3.233e-04            | 3.5x lower      |

12 runs in 11.8s
thermal: max_temp=67.2C pauses=0; temp end 67.1C
```

## §6.3 / Table 1 — k=5120 (Devstral-24B q_proj): engine 1.1296e-3, quire 7.8714e-5, 14.4×

Measured on: physical Blackhole p150b, real Devstral-Small-2507 layer-0 q_proj weights (model-00001-of-00010.safetensors). Producer: internal engine-vs-quire linear-layer rig on the real q_proj layer (not published; weights not redistributed).

```
## (2) Real-scale accumulation: q_proj matmul, k=5120 (Devstral's real depth)
  b-posit16 + Tensix engine : 1.1296e-03
  b-posit16 + EXACT quire   : 7.8714e-05
  quire is 14.4x more accurate at real k=5120
  (iter4b trend: 1.0x@k64 → 1.5x@k256 → 3.5x@k1024 → 14.4x@k5120); 2.9s
```

CPU re-verification of the quire number (seed 0, X~0.3, K=5120, OUT=8, SEQ=2):

```
- Quire RMS 7.8714e-05 @ k=5120 ... got 7.8714e-05 — bit-identical to the paper.
- Table internal consistency: 1.1296e-3 / 7.8714e-5 = 14.35 → 14.4× ✓ (arithmetic is fine).
```

CAVEAT (from the internal number-audit note): the 14.4× characterizes the deployed low-precision b-posit16 engine path (a Blackhole Tensix compute-path artifact of bf16-level compute and/or sub-fp32 accumulation), not a pure fp32 accumulator. A faithful "b-posit16 + fp32 accumulate" on well-conditioned data at k=5120 is comparable to or better than the quire (~1×). The paper states this scope explicitly; the durable win is the cancellation/outlier regime, not this row.

## §6.4 Attention Outliers — fp32 picks wrong arg-max, quire correct; KL 0.403, max softmax-weight err 0.409

Measured on: physical Blackhole silicon (D=32, 3 keys, ±2^33 massive-activation outliers; fp32 add via bit-exact softfp32 on the same baby core). Producer: `targets/blackhole/programming_examples/bposit_quire_attention` (published).

```
Softmax → quire/exact argmax = key0 (the true answer); fp32 argmax = key2 (WRONG). An argmax
flip. KL(exact‖fp32)=0.403, max softmax-weight error 0.409. PASS on real silicon
```

```
bposit_quire_attention on silicon: D=32, 3 keys, ±2^33 massive-activation outliers. exact-quire
argmax=key0 (TRUE); fp32 argmax=key2 (WRONG). KL(exact||fp32)=0.403, max weight err 0.409. quire bit-exact.
```

## §6.5 Gradient/Loss — quire recovers 33/50/60/67% of dropped signal at T=64/128/192/256

Measured on: physical Blackhole silicon. Producer: `targets/blackhole/programming_examples/bposit_quire_gradient` (published).

```
bposit_quire_gradient on silicon: gradient accumulation g=Σδ_t, T={64,128,192,256}. fp32 drops 33/50/60/67%
of the gradient (tiny per-step grads < running-sum ULP, dropped; worse with more steps — the large-model-training
regime); exact-quire bit-exact to oracle 4/4. Training-case complement to the inference demos. Real silicon.
```

## §6.5 Gradient/Loss — on-device cross-entropy = 2.625 bits, 9/9 tests, tails retained

Measured on: physical Blackhole baby core. Producer: `targets/blackhole/programming_examples/bposit_quire_cross_entropy` (published).

```
- device H(p,q) bp32 = 0x45400000 = 2.625 bits; golden = 0x45400000 → 9/9 match, bit-exact.
- device negated quire == golden quire (full 256-bit limb match). PASS on real silicon.
```

```
bposit_quire_cross_entropy on silicon (after the LUT-overflow fix): H(p,q)=−Σ p_i·log2(q_i)=2.625 bits,
9/9 bit-exact vs golden/cross_entropy.json. Exact-quire SUM 100% on-device (tail-lossless, rounding-stable);
log2 host-precomputed (honest split, 64K LUT doesn't fit baby core).
```

## §6.6 Biased Rounding — per-row bias z-score +154 (sequential BF16); random-sign control −0.8; quire 0.0

Measured on: host (flash-attention bf16-accumulation bias, ICLR'26 failure mode arXiv:2510.04212). Producer: internal flash-attention bias demo (not published).

```
bf16 per-row error z=+154 / mean +26.3% of |O| / 100% one-direction under paper conditions;
random-sign CONTROL z=-0.8 (mechanism, not artifact); exact quire z=0.0 bitwise-exact
```

```
directional bias z = +154, +26% of |Ō|, 356× the random-walk envelope; quire bitwise exact, gate 10/10
```

## §6.7 W8A8 Inference — Llama-3.2-1B, all 88 linears b-posit W8A8; next-token = stock Bfp8; logit cosine 0.9924

Measured on: physical Blackhole p150b (device retained) with the full-model matmul run on the host via NumPy (silicon-validated numerics). Metric is single-prompt next-token cosine, not perplexity. Producer: internal full-model W8A8 rig (not published); the codec and the INT8→exact-int32 numerics are the published `targets/blackhole/kernel` headers.

```
- b-posit W8A8 in the model: full Llama, all 88 linears routed through b-posit W8A8 (per-row int8 +
  exact int32) → next-token matches stock Bfp8 ('Paris'), logit cosine 0.9924. Matmul exact-int32 on
  host (= silicon-validated numerics); rest on card.
```

## §6.7 W8A8 Inference — W8A8 matmul on Tensix matrix engine: 24/24 int32 accumulators match golden

Measured on: physical p150b, thermal-guarded. Producer: internal Tensix INT8 matmul example (not published — not part of the exact-quire lineage, see `targets/blackhole/README.md`).

```
| 1 | bposit_matmul_int8 | ✅ PASS | W8A8 int8 matmul on the Tensix matrix engine, 24/24 int32 acc match golden |
```

```
| 1 | b-posit W8A8 int8 matmul on Tensix matrix engine | 24/24 golden, bit-exact |
```

## §6.7 W8A8 Inference — b-posit16 vs BF16 GEMM RMS lower by 43/30/11% at N=64/128/256

Measured on: physical Blackhole silicon (on-device GEMM accuracy sweep). Producer: on-device GEMM accuracy sweep (kernels public, rig not published).

```
 size   bf16_rms     bposit16_rms   fp32_rms     b-posit16 better
  64    1.6732e-03   9.4889e-04     9.4888e-04   43.3%
 128    2.5077e-03   1.7426e-03     1.7421e-03   30.5%
 256    4.0365e-03   3.5964e-03     3.5959e-03   10.9%
```

```
| 5 | b-posit16 vs bf16 GEMM accuracy | b-posit16 43%/30%/11% lower error, ≈ fp32 |
```

## §6.8 Rounding Mode — SmolLM2-135M WikiText-2 PPL 11.81 → 13.35 (+13.0%) truncate / 12.17 (+3.0%) RTNE

Measured on: host, WikiText-2 perplexity, all linears in b-posit8 W8A8 (power-of-two per-group scales, G=64, LM head fp). Producer: internal perplexity harness (not published).

Note: the source draft records +13.1% (truncate) / +3.1% (RTNE); the paper displays +13.0% / +3.0% — display-precision rounding of the same measured PPLs (11.81 → 13.35 → 12.17, which match exactly).

```
| model (family) | fp32 PPL | W8A8 truncate-toward-zero | W8A8 round-to-nearest |
| SmolLM2-135M (Llama) | 11.81 | 13.35 (+13.1%) | 12.17 (+3.1%) |
```

## §6.8 Rounding Mode — Qwen2.5-Coder-0.5B PPL 19.10 → 22.84 (+19.6%) truncate / 19.24 (+0.7%) RTNE

Measured on: host, WikiText-2 perplexity, same W8A8 config. Producer: internal perplexity harness (not published).

Note: the source draft records +19.5% (truncate) / +0.68% (RTNE); the paper displays +19.6% / +0.7% — display-precision rounding of the same measured PPLs (19.10 → 22.84 → 19.24).

```
| model (family) | fp32 PPL | W8A8 truncate-toward-zero | W8A8 round-to-nearest |
| Qwen2.5-Coder-0.5B (Qwen) | 19.10 | 22.84 (+19.5%) | 19.24 (+0.68%) |
```

## §6.8 Rounding Mode — float64 accum over b-posit8 operands bit-identical to 256-bit quire, max diff 0

Measured on: host, real SmolLM2-135M layer (down_proj, K=512, bp8 round-to-nearest). Producer: internal accumulation-vs-quantization check (not published).

```
Accumulation vs quantization — VERIFIED (0.0 difference). ... We verified on a real SmolLM2-135M
layer (down_proj, K=512, bp8 round-to-nearest) that float64 accumulation over those operands is
bit-identical to the exact 256-bit quire — max difference 0.0.
```

## §6.9 Takum — near unity: b-posit16 RMS 9.81e-5 vs takum16 1.00e-4

Measured on: host evaluation (eS=2 harness profile). Producer: a clean-room Python port of libtakum (ISC), validated bit-exact against the compiled C library (not published).

```
all three b-posit16 rows match the paper to displayed precision (9.805e-05 / 7.613e-02 / 2.866e-03)
```

## §6.9 Takum — wide range: takum16 2.66e-3 vs b-posit16 7.61e-2

Measured on: host evaluation (eS=2 harness profile). Producer: the same takum port (not published).

```
we do NOT claim b-posit wins raw format accuracy — takum16 is decisively better over wide dynamic
range (2.66e-03 vs 7.61e-02); b-posit16 wins only the near-unit golden zone.
```

## §6.9 Takum — cancellation: takum16 & BF16 1.02e-3 vs b-posit16 2.87e-3 (2.866e-03)

Measured on: host evaluation (eS=2 harness profile). Producer: the same takum port (not published).

Note: the b-posit16 numbers in the takum comparison use the harness eS=2 profile, not the primary eS=3 hardware profile — a representational comparison, as the paper states.

```
all three b-posit16 rows match the paper to displayed precision (9.805e-05 / 7.613e-02 / 2.866e-03)
```
