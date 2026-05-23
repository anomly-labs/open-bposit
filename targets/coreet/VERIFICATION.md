# Phase-0 b-posit RTL — forensic verification report

**2026-05-22.** Verification of the Phase-0 b-posit RTL (`bposit_qmac`,
`bposit_encode`, `bposit_dot`) applying **forensic-grade discipline**:
differential verification against a *trusted oracle*, **exhaustive where feasible**
(not sampled), with an **explicit coverage matrix so gaps are known and stated**
— the safeguard against an incomplete block reaching silicon.

## Oracle & method

- **Oracle:** `bposit16_reference.py` (`quire256_to_bposit32`, `code_value`,
  `decode_aiposit5`), itself proven bit-exact against the CUDA/GPU path and the
  x86/RISC-V port (a separate cross-arch port). Same bits on
  GPU ≡ x86 ≡ RISC-V; this report adds RTL to that chain.
- **Method:** RTL simulated under `iverilog -g2012`; outputs diffed bit-for-bit
  against oracle-generated vectors. Reproducible: `make verify`.
- **Determinism:** all vectors regenerate deterministically from the reference
  (`gen_rtl_vectors.py`, `gen_encoder_vectors.py`); no hidden state.

## Coverage matrix — VERIFIED

| Block (file) | Coverage | Result |
|---|---|---|
| `bposit_encode` quire256→bposit32 | **EXHAUSTIVE behavioural**: 132,811 vectors — every scale −52..52 incl. ±48/±49 **MAXPOS/MINPOS/MAXNEG/MINNEG** saturation; single-bit walks + all-ones + alternating + random fractions across **every** regime→exp(3/2/1/0)→frac field-width transition; both signs; the `lo<0` fraction branch; zero; **+120k uniform-random 256-bit** | **132811/132811 PASS** |
| `bposit_qmac` 256-bit quire datapath | aip5 decode ROM (all 32 codes = reference) → int product → 256-bit exact accumulate; 25 dot products incl. edge cases | **25/25 PASS** |
| `bposit8_dot` W8A8 GEMM cell | bp8 (mant,exp)→variable-shift quire→encode; 22 dots incl. NaR/zero/max·max/min·min/mixed-special | **quire 22/22 + bposit32 22/22 PASS** |
| `bposit_dot` aip5 GEMM cell (MAC→encode) | **edge cases** NaR→0, zero→0, mixed-special, max·max, min·min, max·(−max), sign **cancellation→0**, wide-dynamic-range, NaR-sprinkled; + 16 random dots | **25/25 bposit32 PASS** |
| decode ROM (`aip5_dec.hex`) | all 32 aip5 codes, generated from the reference (cannot drift) | exact by construction |
| `bposit_lowp_alu` bp4/aip5 elementwise mul+add | **EXHAUSTIVE compute** (decode→fixed-point→total_e-sat+threshold encode): all 256 (bp4) + 1024 (aip5) pairs × {mul,add} = 2,560, vs independent reference | **2560/2560 PASS** |
| RTL lint | `iverilog -Wall` on all RTL + TB | **clean** |

## Coverage matrix — KNOWN GAPS (honestly stated, not silent)

These are **not yet in RTL** and must NOT be assumed working:

3. **LUT nonlinearities** (silu/exp/recip/sqrt) — not in RTL (map to `trans_*_rom`).
4. **No gate-level / synthesis / timing** — this is RTL behavioural sim only. No
   STA, no gate sim, no power. (The "half-working tapeout" failure mode lives
   here — flagged explicitly as the next gate.)
5. **Not integrated into CORE-ET** — standalone modules; not yet wired into
   `vpu_tensorbposit` / the minion pipeline (Phase 1).
6. **Quire-width bound:** 256-bit Q96 holds aip5 products (≤2^108 each) for any
   realistic K (overflow only beyond ~2^147 accumulations). Stated, not assumed —
   re-check if widening formats or for adversarial inputs.

## Bottom line

The Phase-0 **GEMM-cell datapath (aip5 decode → exact quire MAC → bposit32
encode) is verified RE-grade**: the encoder exhaustively, the datapath with full
edge-case + random coverage, lint clean — bit-identical to the reference and
therefore to the GPU/x86/RISC-V chain. The gaps above are the explicit Phase-0/1
worklist; nothing is assumed correct that hasn't been diffed against the oracle.

Reproduce: `make verify`.
