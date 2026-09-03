<!-- Copyright (c) 2026 Anomly, Inc. SPDX-License-Identifier: Apache-2.0 -->
# CoNGA'26 — result → evidence index

*A Vectorized Exact Quire for b-posit Arithmetic on AI Accelerators* (CoNGA'26).
This directory backs the paper's Artifact Availability statement. Every quantitative
result in the paper is listed below with one of three provenance classes:

- **public** — the producing kernel/script is in this repository (or in
  [mosyne-bposit](https://github.com/anomly-labs/mosyne-bposit) for the RTX 5090
  results) and the number can be regenerated on the named hardware;
- **transcript** — the producing rig is not published (proprietary FPGA flow, model
  weights we cannot redistribute, or internal evaluation harness) but the measured
  output is reproduced verbatim in `transcripts/`;
- **derived** — computed from other listed numbers; re-derived by
  `verify_paper_math.py`.

Nothing here is simulated. Every number was measured on the hardware named.

## Re-run what is re-runnable

```bash
# arithmetic the paper states in prose (18.3 years, 47,041 tiny×tiny products,
# eq. (4)–(8), Table 2 ratios, perplexity deltas, FPGA/SFPU ratios, ...)
git clone https://github.com/anomly-labs/mosyne-bposit ../mosyne-bposit   # for bposit16_reference.py
python3 paper/conga2026/verify_paper_math.py         # -> verify_paper_math.transcript.txt

# the kernel truncates sub-2^-96 products toward zero exactly like the oracle
python3 paper/conga2026/check_quire_patch.py         # -> check_quire_patch.transcript.txt
python3 paper/conga2026/tiny_product_vectors.py --out /tmp/tiny.json

# the three SAT proofs (needs yosys)
./formal/run_formal.sh

# codec vs oracle over all 65,534 codes; ET-SoC1 / RV64 cross-host reproduction
make -C targets/coreet verify-full

# Blackhole SFPU kernels (needs a Blackhole card + tt-metal, see targets/blackhole/README.md)
./build/programming_examples/bposit_cancel_sfpu      # etc.
```

The `*.transcript.txt` files beside the scripts are the outputs of these commands
from the layout of this repository on 2026-09-03.

## Index

Paper section numbers follow the submitted `main.tex`. "Producer" is the kernel or
script that measured the number; the path says where it lives when public.

### §4 Exact quire implementation, §5 Correctness

| result | producer | class |
|---|---|---|
| headroom ≥ 2^63 same-sign MACs ≈ 18.3 years at 16 lanes / 1 GHz | `verify_paper_math.py` | derived |
| 47,041 of 1.04·10^9 positive operand pairs (0.0045 %) have product LSB below 2^-96; truncated toward zero identically in kernel and oracle | `tiny_product_vectors.py`, `check_quire_patch.py` (37,456 unordered pairs, 0 mismatches on all three kernel entry points) | public |
| eq. (4) compare-free carry, (5) carry-save value preservation, (6) `tc = oc + sign`, (7) popcount sign fold, (8) cancellation dot = 62.0 | `verify_paper_math.py` (host re-execution); `targets/blackhole/programming_examples/bposit_quire_sfpu_add`, `bposit_cancel_sfpu` (silicon) | public |
| 9/9 validation stages on Blackhole p150; single tile 128/128; 130×32 grid 4160/4160; arbitrary (M,N,K) 4480/4480; ttnn op 256/256 | `targets/blackhole/programming_examples/bposit_matmul_sfpu`, `_mc_sfpu`, `_full_sfpu`; `targets/blackhole/ttnn_op` (the 9-stage driver script is internal; stage list in `transcripts/s5-correctness.md`) | public |
| 3 Yosys SAT proofs (CSA compressor, signed reduction, quire addend) | `formal/` | public |
| oracle agreement over all 65,534 non-special posit16 codes; b-posit8 exhaustive | `reference/bposit_ref.py`, `reference/gen_bp8_exhaustive.py`, `targets/coreet` | public |
| ET-SoC1 independent host: encoder 132,811/132,811, b-posit8 65,536/65,536, RTL exponent-clamp defect caught; RV64 W8A8 byte-identical to x86 | `targets/coreet` (`make verify-full`) | public |
| order independence: host bf16 30,077 of 32,768 logits differ, KL ≤ 0.13; U200 quire engine 131,072/131,072 bit-identical across two K orders | host demo and U200 engine are internal | transcript (`s5-correctness.md`) |
| reward verification: 4,096 answers, 8 BF16 reduction shapes flip 52 verdicts, exact quire flips 0 | internal | transcript (`s5-correctness.md`) |
| rendering: float32 differs on 12,285/12,288 pixels up to 14 ULP at 128 spp; quire bit-identical | internal | transcript (`s5-correctness.md`) |

### §6 Evaluation

| result | producer | class |
|---|---|---|
| cancellation dot `[+2^40, 1×62, −2^40]`: quire 62.0, fp32/bf16 0.0 | `bposit_cancel_sfpu`; `verify_paper_math.py` | public |
| accuracy vs k (Table 2): 1.0× / 1.5× / 3.5× at k = 64 / 256 / 1024; 14.4× at k = 5120 (Devstral-24B q_proj) | on-device b-posit16 engine vs quire; the k = 5120 row needs the Devstral weights (not redistributed) | transcript (`s6-evaluation.md`, with the engine-vs-fp32 caveat); ratios derived |
| attention outliers q·k = ±2^33: fp32 wrong arg-max, quire correct; KL 0.403, max softmax-weight error 0.409 | `targets/blackhole/programming_examples/bposit_quire_attention` (`gen_attention_golden.py` re-derives the KL on the host) | public |
| gradient/loss: 33/50/60/67 % of dropped signal recovered at T = 64/128/192/256; cross-entropy 2.625 bits, 9/9 | `bposit_quire_gradient`, `bposit_quire_cross_entropy` (host generators re-derive the percentages; silicon transcripts in each `BUILD_NOTES.md`) | public |
| biased rounding: per-row z-score +154 (sequential BF16), −0.8 random-sign control, 0.0 quire | internal | transcript (`s6-evaluation.md`) |
| W8A8 Llama-3.2-1B, 88 linears: next token = stock Bfp8, logit cosine 0.9924; Tensix INT8 24/24 accumulators; GEMM RMS −43/−30/−11 % at N = 64/128/256 | codec + kernels public (`targets/blackhole/kernel`, `bposit_bf16_codec`); full-model rig internal | transcript (`s6-evaluation.md`) |
| rounding mode: SmolLM2-135M PPL 11.81 → 13.35 (+13.0 %) trunc / 12.17 (+3.0 %) RTNE; Qwen2.5-Coder-0.5B 19.10 → 22.84 (+19.6 %) / 19.24 (+0.7 %); float64-vs-quire max diff 0 | internal perplexity harness; percentages re-derived by `verify_paper_math.py` | transcript (`s6-evaluation.md`) + derived |
| takum comparison: near-unity 9.81e-5 vs 1.00e-4; wide-range 2.66e-3 vs 7.61e-2; cancellation 1.02e-3 vs 2.87e-3 (eS = 2 profile) | internal clean-room takum port | transcript (`s6-evaluation.md`) |

### §7 Performance and hardware cost

| result | producer | class |
|---|---|---|
| 128×64×128 exact-quire matmul on 130 cores: 2.653 ms, 395 M products/s, 64/64 sampled bit-exact; 8 seeds × 16³ 2048/2048 bit-exact; 11 / 20 MAC/µs streaming K-loop on 64 / 130 cores | `bposit_quire_matmul_multicore_perf`, `bposit_quire_matmul_validate`, `bposit_dotk_stream_sfpu` (timings are silicon measurements; see each `BUILD_NOTES.md`) | public |
| ≈10^5 below the native 91 TFLOP/s engine | derived | derived |
| U200 625-PE engine @90 MHz: 25.95 G (b-posit16) / 47.80 G (5-bit) exact products/s, bit-exact; 400-PE @140 MHz variant | proprietary FPGA flow (only the CSA cell is published, `formal/bposit_csa_cs42.v`) | transcript (`s7-performance.md`) |
| 65–120× the SFPU throughput | `verify_paper_math.py` | derived |
| exact vs rounded MAC: 4,179 LUTs @341.6 MHz vs 9,673 LUTs @32.6 MHz (2.3× smaller, 10.5× faster); Yosys 0.84× cells; 867 differential dot tests | Vivado / Yosys on the single-lane exact and rounded MAC RTL (the chip RTL is not published; its exactness-critical cells are the proven `formal/` blocks); the RTNE encoder model is `mosyne-bposit/kernels/bposit16_reference.py` | transcript (`s7-performance.md`) + derived |
| RTX 5090 INT8 IMMA: 167.8 TPOPS @2048³ (1.00× BF16), 197.4 @4096³ (1.14×), up to 1.33×; Qwen2.5-Coder-1.5B 133.2 vs 132.8 tok/s; 5 runs → 1 bit-identical result vs 5 distinct with fp32 atomicAdd; 3B FFN 3.56 % L2 | `mosyne-bposit/kernels/` (`bench_imma_vs_qmma.cu`, `bench_robust.cu`, `test_determinism_headline.cu`, `test_ffn_layer_*.cu`) | public |

## Known scope notes (also stated in the paper)

- The two order-independence counts are two shapes of the same 8192×768 · 768×16
  GEMM: 30,077/32,768 is a host 32,768-logit instance, 131,072/131,072 the full
  8192×16 output on the U200 engine.
- Table 2's k = 5120 row (14.4×) characterizes the deployed low-precision engine path
  against the quire; an fp32 accumulator on the same well-conditioned data is ≈1×.
  The paper's fundamental claim is the cancellation / outlier regime.
- The Llama-3.2-1B W8A8 result is a host-executed matmul with the device's numerical
  semantics and a single-prompt next-token cosine, not a perplexity suite.
- Products with LSB below 2^-96 are truncated toward zero on placement (see the
  first table); the exactness claim is scoped to every other product.

## Files

```
README.md                          this index
verify_paper_math.py               re-executes the paper's stated arithmetic     (+ .transcript.txt)
check_quire_patch.py               kernel == oracle on every sub-2^-96 product    (+ .transcript.txt)
tiny_product_vectors.py            golden vectors for those products
transcripts/s5-correctness.md      measured logs for §5 results whose rig is not published
transcripts/s6-evaluation.md       measured logs for §6 results whose rig is not published
transcripts/s7-performance.md      measured logs for §7 results whose rig is not published
```
