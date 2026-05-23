#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate iverilog $readmemh test vectors for the b-posit RTL, from the SAME
Python reference the GPU/RISC-V port is proven against. Emits, for M dot
products of length K (aiposit5 operands):

  a.hex / b.hex          M*K aip5 codes (one hex/line)
  eq.hex                 M expected quire values (256-bit, one 64-hex/line)
  ebp32.hex              M expected bposit32 results (8-hex/line)
  dims.svh               localparam M, K

A match in the RTL sim == the hardware datapath is bit-identical to the
reference (== the GPU/x86/RISC-V port). aip5 (range 2^-6..2^6) → products are
exact in the 256-bit quire.
"""
from __future__ import annotations
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bposit_ref as r  # noqa: E402

K = 32
SEED = 20260522
HERE = Path.cwd()
NAR, ZERO, MAXP, MINP, MAXN, MINN = 0x10, 0x00, 0x0f, 0x01, 0x11, 0x1f  # aip5 specials/extremes


def _edge_dots():
    """Deterministic edge-case dot products (codes) — the cases sampling misses:
    specials, sign cancellation, extreme magnitudes, mixed special+real."""
    half = K // 2
    dots = [
        ([NAR]*K,            [3]*K),                       # NaR -> 0 contribution
        ([ZERO]*K,           [MAXP]*K),                    # zero -> 0
        ([NAR, ZERO]*half,   [MAXP, MINP]*half),           # mixed special
        ([MAXP]*K,           [MAXP]*K),                    # max*max (high magnitude)
        ([MINP]*K,           [MINP]*K),                    # min*min (low magnitude)
        ([MAXP]*K,           [MAXN]*K),                    # max*-max (negative)
        ([MAXP]*half + [MAXN]*half, [MAXP]*K),             # cancellation to ~0
        ([MAXP, MINP]*half,  [MINP, MAXP]*half),           # wide dynamic range in one dot
        ([8]*K,              [NAR if i%4==0 else 7 for i in range(K)]),  # NaR sprinkled
    ]
    return dots


def main():
    rng = random.Random(SEED)
    A, B, EQ, EBP = [], [], [], []
    n_rand = 16
    dots = _edge_dots() + [
        ([(lambda c: 0 if c == 0x10 else c)(rng.randrange(32)) for _ in range(K)],
         [(lambda c: 0 if c == 0x10 else c)(rng.randrange(32)) for _ in range(K)])
        for _ in range(n_rand)
    ]
    M = len(dots)
    for da, db in dots:
        q = 0
        for a, b in zip(da, db):
            A.append(a); B.append(b)
            q += int(r.code_value(a, "aip5") * r.code_value(b, "aip5") * (1 << r.QUIRE_FRAC_BITS))
        # 256-bit two's-complement representation of the accumulated quire
        EQ.append(q & ((1 << 256) - 1))
        EBP.append(r.quire256_to_bposit32(q) & 0xFFFFFFFF)

    # aip5 decode ROM: code -> signed Q16 value (0 for NaR/zero), 32-bit two's-comp hex
    dec = []
    for c in range(32):
        d = r.decode_aiposit5(c)
        v = 0 if d.is_special else int(r.decoded_to_fraction_5(d) * (1 << 16))
        dec.append(v & 0xFFFFFFFF)
    (HERE / "aip5_dec.hex").write_text("\n".join(f"{v:08x}" for v in dec) + "\n")

    (HERE / "a.hex").write_text("\n".join(f"{c:02x}" for c in A) + "\n")
    (HERE / "b.hex").write_text("\n".join(f"{c:02x}" for c in B) + "\n")
    (HERE / "eq.hex").write_text("\n".join(f"{v:064x}" for v in EQ) + "\n")
    (HERE / "ebp32.hex").write_text("\n".join(f"{v:08x}" for v in EBP) + "\n")
    (HERE / "dims.svh").write_text(f"localparam int M = {M};\nlocalparam int K = {K};\n")
    print(f"wrote vectors: M={M} dots of K={K} aip5 MACs (a/b/eq/ebp32.hex, dims.svh)")

    # ---- bposit8 / W8A8 path: (mantissa, exp) decode ROMs + dot vectors ----
    bp8_mant, bp8_exp = [], []
    for c in range(256):
        d = r.decode_bposit8(c)
        if d.is_special:
            bp8_mant.append(0); bp8_exp.append(0)
        else:
            m = (1 << d.f_width) + d.f_bits
            e = 4 * d.k + d.e - d.f_width
            bp8_mant.append((-m if d.sign else m) & 0xFFFF)   # signed 16-bit two's-comp
            bp8_exp.append(e & 0xFF)                          # signed 8-bit (range ~-28..24)
    (HERE / "bp8_mant.hex").write_text("\n".join(f"{v:04x}" for v in bp8_mant) + "\n")
    (HERE / "bp8_exp.hex").write_text("\n".join(f"{v:02x}" for v in bp8_exp) + "\n")

    BNAR, BZERO, BMAXP, BMINP = 0x80, 0x00, 0x7f, 0x01
    bp8_dots = [
        ([BNAR]*K, [3]*K), ([BZERO]*K, [BMAXP]*K),
        ([BMAXP]*K, [BMAXP]*K), ([BMINP]*K, [BMINP]*K),
        ([BMAXP]*K, [0xFF]*K),                              # max * (max negative-ish)
        ([BNAR, BZERO]*(K//2), [BMAXP, BMINP]*(K//2)),
    ] + [
        ([(lambda c: 0 if c == 0x80 else c)(rng.randrange(256)) for _ in range(K)],
         [(lambda c: 0 if c == 0x80 else c)(rng.randrange(256)) for _ in range(K)])
        for _ in range(16)
    ]
    A8, B8, EQ8, EBP8 = [], [], [], []
    for da, db in bp8_dots:
        q = 0
        for a, b in zip(da, db):
            A8.append(a); B8.append(b)
            q += int(r.code_value(a, "bp8") * r.code_value(b, "bp8") * (1 << r.QUIRE_FRAC_BITS))
        EQ8.append(q & ((1 << 256) - 1)); EBP8.append(r.quire256_to_bposit32(q) & 0xFFFFFFFF)
    (HERE / "bp8_a.hex").write_text("\n".join(f"{c:02x}" for c in A8) + "\n")
    (HERE / "bp8_b.hex").write_text("\n".join(f"{c:02x}" for c in B8) + "\n")
    (HERE / "bp8_eq.hex").write_text("\n".join(f"{v:064x}" for v in EQ8) + "\n")
    (HERE / "bp8_ebp.hex").write_text("\n".join(f"{v:08x}" for v in EBP8) + "\n")
    (HERE / "bp8_dims.svh").write_text(f"localparam int M8 = {len(bp8_dots)};\nlocalparam int K8 = {K};\n")
    print(f"wrote bp8 vectors: {len(bp8_dots)} dots, mant/exp ROMs (bp8_*.hex)")


if __name__ == "__main__":
    main()
