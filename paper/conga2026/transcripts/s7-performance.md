<!-- Copyright (c) 2026 Anomly, Inc. SPDX-License-Identifier: Apache-2.0 -->

# §7 Performance & Hardware Cost — reproduction transcripts

## §7.1 Vector-Unit Cost — 128×64×128 exact-quire matmul across 130-core grid: 2.653 ms, 395M products/s, 64/64 bit-exact

Measured on: physical Blackhole silicon (130 cores, full grid), thermally guarded (max ASIC ~66°C). Producer: `targets/blackhole/programming_examples/bposit_quire_matmul_multicore_perf` (published).

```
- bposit_quire_matmul_multicore_perf: 128×64×128 exact-quire matmul, 2.653 ms,
  395M exact-products/s (6.18M exact-quire-dots/s); sample 64/64 bit-exact. PASS.
```

## §7.1 Vector-Unit Cost — 8-seed 16³ validation: 2048/2048 outputs bit-exact

Measured on: physical Blackhole silicon. Producer: `targets/blackhole/programming_examples/bposit_quire_matmul_validate` (published).

```
- bposit_quire_matmul_validate: 8 seeds × 16³, 2048/2048 outputs bit-exact incl. 8 full
  256-bit quire byte-gates. PASS.
```

## §7.1 Vector-Unit Cost — native Tensix engine ~91 TFLOP/s vs SFPU quire ~0.79 GFLOP/s (~10^5 slower)

Measured on: physical Blackhole silicon. Producer: the perf example above for the quire rate; the native-engine figure is tt-metal's bf16 matmul on the same card.

```
| path                           | throughput                    | accuracy                                    |
| bf16 / b-posit16 Tensix engine | ~91 TFLOP/s                   | engine error (masks storage, fp32 cancellation) |
| exact 256-bit quire (this kernel) | ~0.79 GFLOP/s (395M MAC/s) | bit-exact, recovers what fp32 loses         |

Exact-quire is ~5 orders of magnitude slower than the hardware MAC array here — it computes
the 256-bit accumulation as multi-word integer arithmetic in the kernel running on the Tensix cores.
```

Cross-referenced native-engine throughput (internal measurement notes):

```
| 6 | bf16 matmul throughput | ~91 TFLOP/s burst, ~63.5 sustained (self-throttled) |
```

## §7.2 FPGA U200 — 625-PE @90 MHz: 25.95 G (b-posit16) / 47.80 G (5-bit) exact products/s, bit-exact

Measured on: physical AMD Alveo U200 card. Producer: 625-PE dual-mode GEMM engine RTL and its on-card batch testbench (not published; proprietary FPGA flow — the exactness-critical cells are the proven blocks in `formal/`).

```
- 625-PE @90 MHz → 25.95 GMAC/s (16b) / 47.80 (5b) — overall record (+74%/+87% vs 625@60).
```

```
- Scaled bit-exact arrays to 625 PEs on one FPGA; HW-verified throughput 25.95 GMAC/s (16b) /
  47.80 GMAC/s (5b) @90 MHz; dual-mode @125 MHz.
```

## §7.2 FPGA U200 — ~65–120× the SFPU-emulated throughput

Measured on: derived ratio (U200 GMAC/s ÷ SFPU emulated rate). Producer: derived (no separate measurement).

```
- FPGA ratio recomputes as 65.7–121.0× ("65–120×" is fine, "65–121×" is exact).
```

## §7.2 FPGA U200 — 400-PE variant + mid-multiplier pipeline stage: 25.27 / 45.47 G products/s @140 MHz, WNS −0.103 ns

Measured on: physical AMD Alveo U200, 2026-07-08 (400-PE = 20×20 array, "stage-3" decode pipe = mid-piped multiplier; requested 140 MHz). Producer: proprietary FPGA flow (not published); build-config record note reproduced verbatim:

```
# 140MHz RESULT (2026-07-08): ROUTED clean (Router Completed, NO conflicted nets), final
# signoff WNS -0.103 (~138.0MHz netlist Fmax — STAGE-3 raised the 400-PE ceiling from ~121-125,
# +10%+ as designed).
# HW-VERIFIED on real U200 (2026-07-08 ~18:05): known-answer PASS BOTH modes (fails=0).
# BENCH NTILES=512 P=256 (record conditions): 16-bit 25.27 GMAC/s, 5-bit 45.47 GMAC/s
# = NEW 400-PE-class RECORDS (+14.6% / +10.5% over the 125MHz db+diet 22.06/41.16), pure
# stage-3 clock-lever win at identical array size. All-time records (625-PE @90:
# 25.95/47.80) still stand
# Honest note: kernel programmed at the 140 ask with signoff WNS -0.103; device operates
# bit-exact at ambient margin (both modes verified on silicon).
```

The 625-PE @90 MHz run above remains the design of record; this build is the same
exactness-critical datapath at a smaller array and a higher clock.

## §7.2 — exact MAC place-and-routes at 341.6 MHz in isolation; tapeout die did not power on (power-rail)

Measured on: Vivado post-route (xcu200, OOC, 200 MHz constraint) for the P&R number; custom-silicon tapeout for the power-on status. Producer: Vivado P&R of the single-lane exact MAC RTL (not published).

```
| Fmax (post-route) | 341.6 MHz | 32.6 MHz | exact ~10.5× faster |
```

Tapeout status (as stated in the paper, §7.2): "we did tape out a real chip ... the full datapath design is on custom silicon. That die didn't power on (a power-rail issue), so we don't yet have on-silicon measurements from it; bring-up is waiting on a respin."

## §7.3 Exact vs Rounded — exact 4,179 LUTs @341.6 MHz vs rounded 9,673 LUTs @32.6 MHz (2.3× smaller, 10.5× faster); 867 differential dot tests

Measured on: Vivado full place-and-route (xcu200, OOC, 200 MHz constraint). Producer: Vivado P&R of the single-lane exact MAC RTL vs the RTNE-rounded MAC RTL (neither published); the rounded baseline is verified against the published RTNE encoder model `mosyne-bposit/kernels/bposit16_reference.py`.

```
| | exact bposit16_mac | rounded bposit16_mac_rounded | ratio |
| CLB LUTs (post-route) | 4,179 | 9,673 | exact 2.31× smaller |
| CLB Registers | 294 | 48 | (exact holds the 256-bit quire) |
| CARRY8 chains | 216 | 782 | 3.6× more in the rounded path |
| DSP48 | 1 | 1 | identical product front-end |
| Fmax (post-route) | 341.6 MHz | 32.6 MHz | exact ~10.5× faster |
```

Rounded baseline is not a strawman:

```
The rounded MAC is a correct baseline — acc_{i+1} = RNE(acc_i + a_i·b_i), verified bit-exact vs
bposit16_reference.encode_bposit16(…, mode="rtne") across 867 differential dot-product tests
```

## §7.3 Exact vs Rounded — Yosys generic synth: exact ≈ 0.84× cells of rounded

Measured on: Yosys generic synthesis of both MACs. Producer: Yosys generic synthesis of the same two MACs (RTL not published).

```
Headline (the clean exact/rounded ratio, now measured): the exact MAC is 67,076 cells vs the correct
RNE-rounded MAC's 79,490 — exactness is ~16% smaller, i.e. 0.844×.
```
