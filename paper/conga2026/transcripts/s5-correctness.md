<!-- Copyright (c) 2026 Anomly, Inc. SPDX-License-Identifier: Apache-2.0 -->

# §5 Correctness & Reproducibility — reproduction transcripts

## §5.1 Bit-Exact Validation — 9/9 stages pass on Blackhole p150b

Measured on: physical Blackhole p150b. Producer: an internal one-command chain driver (not published); every stage it runs is a published example under `targets/blackhole/programming_examples/` and prints its own gate line.

The one-command reproduction gates each stage of the chain on its bit-exact PASS line:

```
Builds and runs the whole chain (carry → decode → product → placement → dot → matmul → K-loop →
cancellation-WIN → multi-core), gating each on its bit-exact PASS line → 9/9 ALL GREEN on the card.
```

Per-stage results from the proven chain (each a standalone tt-metal programming example, gated bit-for-bit against the scalar oracle `bp16_madd_q256`):

```
| step   | what                                            | result on Blackhole      |
| carry  | lane-parallel exact 256-bit q256_add            | 32/32 lanes bit-exact    |
| M1     | bp16_decode → per-lane (sign, M, E2)            | 32/32                    |
| M2     | product (Ma·Mb, Ea+Eb, sign)                    | 32/32                    |
| M3     | signed windowed placement = bp16_prod_to_q256   | 32/32 × 8 limbs          |
| M3.5   | exact dot = bp16_madd_q256 chain over K         | 32/32 (K=4, K=32)        |
| M4     | exact-quire MATMUL Y=ΣX·W                        | 128/128 elements         |
| K-loop | whole K-dot in ONE dispatch (quire L1-resident) | 32/32                    |
| value  | cancellation WIN                                | exact vs fp32/bf16       |
| scale  | multi-core across the full grid                 | 4160/4160                |
```

## §5.5 Order Independence — host bf16 GEMM: 30,077 of 32,768 logits disagree, KL up to 0.13

Measured on: host CPU (real GPT-2 weights). Producer: internal reproducibility suite (not published).

GEMM shape is 2048×768 @ 768×16 = 32,768 outputs:

```
# GPT-2-ish logit-GEMM proportions: the shape the claim is quoted at (2048x768 @ 768x16
# = 32,768 outputs).
ROWS, K, COLS = 2048, 768, 16
```

Measured result (bf16 accumulated in bf16, rounding after every add, two faithful kernel orders):

```
| 1 | TIM — logit reproducibility across two faithful kernel orders | 30,077 / 32,768 logits differ (bf16), KL up to 0.13, log-prob off by 1.44 | 0 differ — bit-identical, KL exactly 0 |
```

## §5.5 Order Independence — U200 silicon: 131,072/131,072 output codes bit-identical across two K orders

Measured on: physical U200 GEMM engine (400 PEs, one private 256-bit quire each), 2026-07-09. Producer: internal on-card driver (not published; FPGA bitstream not public).

GEMM shape: 8192-row wte slice × 768 @ 768 × 16 real layer-0 hidden states; natural contraction order vs a fixed disclosed K-axis permutation of both operands:

```
| — | on real silicon — same logit GEMM run twice on the U200 engine | (float would differ) | 131,072 / 131,072 output codes bit-identical |
```

```
131,072/131,072 output codes bit-identical, KL exactly 0, and 32/32 random outputs bit-exact vs the
CPU exact-Fraction quire
```

Cross-referenced in the internal measurement notes: "RL mismatch eliminated on the physical U200 (131,072/131,072 bit-identical)."

## §5.5 Order Independence — reward verification: 4,096 answers, 8 BF16 shapes flip 52 verdicts; exact quire flips 0

Measured on: host CPU (4,096 real GPT-2 answer embeddings). Producer: internal verifier-determinism demo, 4,096 answers × 8 reduction orders, 2 %-of-range boundary band (not published).

```
| 3 | Verifier determinism — 4,096 real answers re-scored under 8 real reduction shapes | 52 answers (12% of the boundary band) flip pass/fail | 0 flips — verdict cannot change |
```

```
52/4,096 answers — all in the 2%-of-range boundary band, i.e. 12% of that band (52/433) — flip
pass/fail depending only on kernel shape. ... The exact-quire verifier: score codes bit-identical
under all 8 permuted-K orders, 0/4,096 flips, by construction.
```

## §5.6 Rendering — float32 differs for 100% of pixels (12285/12288), up to 14 ULP; quire bit-identical

Measured on: real GPU radiance capture (64×64 px × 128 samples/pixel = 12288 pixel-channels), same samples reduced across 8 accumulation orders. Producer: internal radiance-reproducibility script (not published).

```
| reduction of the same samples, 8 accumulation orders | pixels differing | spread |
| float32 sequential (naive / GPU-schedule path) | 12285 / 12288 (100%) | ≤ 14 ULP (max 5.3e-5) |
| shipped exact path | 0 | 0.0 — bit-identical |
| true 256-bit quire | 0 | 0.0 — bit-identical |
```
