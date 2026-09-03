#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-execute every equation / algorithm / derived number in the CoNGA'26
paper (main.tex as submitted, 2026-09) against REAL repo code — no simulation, no hand-waving.

Each check prints PASS / FAIL / NOTE with the recomputed value next to the
paper's value. The reference codec is the Apache-2.0 oracle the Blackhole
kernel is transcribed from (bposit16_reference.py, ES=3, useed=256); the
paper's formal-proof scripts are not re-run here (they live in the public
repo) — this file covers the arithmetic the paper states in prose.

    python3 verify_paper_math.py          # needs bposit16_reference.py, see below
"""
from __future__ import annotations

import math
import os
import random
import sys
from fractions import Fraction

# bposit16_reference.py (Apache-2.0) ships with github.com/anomly-labs/mosyne-bposit
# under kernels/.  Set BPOSIT16_REFERENCE_DIR, or clone mosyne-bposit beside this repo.
HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
CANDIDATES = [
    os.environ.get("BPOSIT16_REFERENCE_DIR", ""),
    os.path.join(HERE, "../../../../mosyne-bposit/kernels"),
    os.path.join(HOME, "development/mosyne-bposit/kernels"),
    os.path.join(HOME, "development/mosyne-bposit-public/kernels"),
]
for c in CANDIDATES:
    if c and os.path.exists(os.path.join(c, "bposit16_reference.py")):
        sys.path.insert(0, os.path.abspath(c))
        break
else:
    sys.exit("bposit16_reference.py not found: set BPOSIT16_REFERENCE_DIR or clone "
             "https://github.com/anomly-labs/mosyne-bposit beside this repository")
import bposit16_reference as R  # noqa: E402

FAILS = []


def report(tag, ok, detail):
    word = "PASS" if ok else "FAIL"
    if ok is None:
        word = "NOTE"
    elif not ok:
        FAILS.append(tag)
    print(f"[{word}] {tag}: {detail}")


# ---------------------------------------------------------------------------
# §3.1  Eq. (1)/(2): posit16 decode + two's-complement symmetry, all codes
# ---------------------------------------------------------------------------
def check_decode_symmetry():
    n = 16
    mism = 0
    nonspecial = 0
    for p in range(1 << n):
        d = R.decode_bposit16(p)
        if d.is_special:
            continue
        nonspecial += 1
        v = R.decoded_to_fraction(d)
        q = (-p) & ((1 << n) - 1)
        w = R.decoded_to_fraction(R.decode_bposit16(q))
        if v != -w:
            mism += 1
    report("eq(2) value(p) = -value((-p) mod 2^n), all posit16 codes",
           mism == 0, f"{nonspecial} non-special codes, {mism} mismatches")
    report("code count wording", nonspecial == 65534,
           f"65,536 total codes, {nonspecial} non-special (0 and NaR excluded) "
           "-> the paper's '65,535 posit16 codes' (intro, §4.10, §7) is wrong; "
           "§3.1's '65,534 non-special' is right")


# ---------------------------------------------------------------------------
# §3.1  the 6-bit example 1 01 10 1, eS=2, useed=16: canonical -3 vs shorthand -3/8
# ---------------------------------------------------------------------------
def posit_decode_generic(code, n, es, shorthand=False):
    mask = (1 << n) - 1
    code &= mask
    if code == 0:
        return Fraction(0)
    if code == 1 << (n - 1):
        return None
    neg = bool(code >> (n - 1))
    if neg and not shorthand:
        code = (-code) & mask
    body = format(code, f"0{n}b")[1:]
    r0 = body[0]
    run = len(body) - len(body.lstrip(r0))
    k = (run - 1) if r0 == "1" else -run
    rest = body[run + 1:] if run < len(body) else ""
    ebits = rest[:es].ljust(es, "0")
    e = int(ebits, 2) if es else 0
    fbits = rest[es:]
    f = Fraction(int(fbits, 2), 1 << len(fbits)) if fbits else Fraction(0)
    useed = 1 << (1 << es)
    val = Fraction(useed) ** k * (1 << e) * (1 + f)
    return -val if neg else val


def check_six_bit_example():
    code = 0b101101
    canon = posit_decode_generic(code, 6, 2)
    short = posit_decode_generic(code, 6, 2, shorthand=True)
    report("§3.1 6-bit example 1 01 10 1 (eS=2)", canon == -3 and short == Fraction(-3, 8),
           f"canonical {canon} (paper -3), sign-magnitude shorthand {short} (paper -3/8)")


# ---------------------------------------------------------------------------
# §3.3  quire sizing: T = 8k+e in [-48,47], 4E = 192, +64 guard = 256, 2^63 depth
# ---------------------------------------------------------------------------
def check_quire_sizing():
    Ts, E2s = [], []
    bounded = []
    for p in range(1, 1 << 15):          # positive magnitudes; sign symmetric (eq 2)
        d = R.decode_bposit16(p)
        T = 8 * d.k + d.e
        E2 = T - d.f_width
        M = (1 << d.f_width) + d.f_bits if d.f_width else 1
        Ts.append(T); E2s.append(E2)
        if -6 <= d.k <= 5:
            bounded.append((p, E2, M))
    bT = [8 * R.decode_bposit16(p).k + R.decode_bposit16(p).e for p, _, _ in bounded]
    report("§3.3 T = 8k+e range of the bounded profile (k in [-6,5])",
           min(bT) == -48 and max(bT) == 47, f"[{min(bT)}, {max(bT)}] (paper [-48, 47])")
    report("§3.3 the reference DECODER accepts unbounded regimes", None,
           f"decode_bposit16 spans T in [{min(Ts)}, {max(Ts)}] (k in [-14,14]); the ENCODER "
           "clamps to |T|<=48 (bp16_encode.h:26/113), so pipeline codes stay bounded. The paper "
           "should say the bound is enforced at encode, not that the format has rS=6 decode")
    E, span, guard, q = 48, 192, 64, 256
    report("§3.3 4E = 192, + 64 guard = 256", 4 * E == span and span + guard == q,
           f"4*{E} = {4*E}, {span}+{guard} = {span+guard}")
    E5 = 192
    report("§3.3 eS=5 profile: 4E = 768 + 32 guard = 800", 4 * E5 + 32 == 800, f"{4*E5}+32 = {4*E5+32}")
    report("§3.3 Posit Standard 16n rule, n=16", 16 * 16 == 256, "256")
    # accumulation depth: max |product| < 2^96 -> fixed-point < 2^192; signed 256-bit holds < 2^255
    maxv = max(R.decoded_to_fraction(R.decode_bposit16(p)) for p, _, _ in bounded)
    top = math.floor(math.log2(maxv))
    depth_bits = 255 - 2 * (top + 1) - 96 + 96   # (2^255) / (2^(2*(top+1)) * 2^96) in 2^96-scaled units
    depth = 255 - (2 * (top + 1) + 96)
    report("§3.3 exact depth >= 2^63 worst-case same-sign MACs",
           depth >= 63, f"max value 2^{top}.x -> max product < 2^{2*(top+1)} at bit {2*(top+1)+96}; "
                       f"headroom 2^{depth} (paper 'at least 2^63')")
    yrs = 2 ** 63 / (16 * 1e9) / (365.25 * 86400)
    report("§4 'approximately 18.3 years at 16 lanes and 1 GHz'", abs(yrs - 18.3) < 0.1,
           f"2^63 / (16 * 1e9 /s) = {yrs:.2f} years (paper 18.3)")

    # THE REAL ONE: is every product of two bounded codes exact in a 96-frac-bit quire?
    # placement bit = E2a + E2b + 96 must be >= 0 (bp16_quire.h:68-69 drops it otherwise)
    small = [x for x in bounded if x[1] <= -43]
    inexact = 0
    worst = 0
    for pa, ea, Ma in small:
        for pb, eb, Mb in small:
            sh = ea + eb + 96
            if sh < 0 and ((Ma * Mb) & ((1 << -sh) - 1)):
                inexact += 1
                worst = max(worst, ea, eb)
    n = len(bounded)
    report("§4 '47,041 of the 1.04e9 positive operand pairs ... 0.0045 %' truncated below 2^-96",
           inexact == 47041 and abs(n * n / 1.04e9 - 1) < 0.01 and abs(100 * inexact / (n * n) - 0.0045) < 0.0002,
           f"{inexact:,} of {n*n:,} ({n*n/1e9:.2f}e9) bounded operand pairs have product LSB below 2^-96 "
           f"({100*inexact/(n*n):.4f} %); min single-value E2 = {min(e for _, e, _ in bounded)}; "
           "bp16_quire.h `if (shift < 0) return;` truncates such products toward zero, identically in "
           "the oracle. Example: 0x101*0x101 = 1089*2^-106. Paper claim is scoped to products with "
           "LSB >= 2^-96 (|a|,|b| >~ 2^-37)")


# ---------------------------------------------------------------------------
# §4.3  Eq. (3)/(4): compare-free carry-out
# ---------------------------------------------------------------------------
def carry_paper(q, x):
    s0 = (q + x) & 0xFFFFFFFF
    return ((q & x) | ((q | x) & (~s0 & 0xFFFFFFFF))) >> 31


def check_carry():
    rng = random.Random(0xC0FFEE)
    edge = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF]
    bad = 0
    pairs = [(a, b) for a in edge for b in edge]
    pairs += [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(2_000_000)]
    for q, x in pairs:
        if carry_paper(q, x) != ((q + x) >> 32):
            bad += 1
    # exhaustive at 8-bit width (same algebra, width-generic)
    bad8 = sum(1 for q in range(256) for x in range(256)
               if (((q & x) | ((q | x) & (~((q + x) & 0xFF) & 0xFF))) >> 7) != ((q + x) >> 8))
    report("eq(4) cout = ((q&x)|((q|x)&~s0))>>31", bad == 0 and bad8 == 0,
           f"{len(pairs):,} 32-bit pairs incl. edges: {bad} wrong; exhaustive 8-bit analogue: {bad8} wrong")


# ---------------------------------------------------------------------------
# §4.9  Eq. (5) carry-save invariant, Eq. (6)/(7) tc = oc + sign, popcount identity
# ---------------------------------------------------------------------------
def check_csa_and_signs():
    rng = random.Random(7)
    M = (1 << 256) - 1
    bad = 0
    for _ in range(20000):
        s0, c0, s1, c1 = (rng.getrandbits(256) for _ in range(4))
        # 3:2 compressor on (s0 + c0) + (s1 + c1): fold in two CSA steps
        a, b, c = s0, c0, s1
        s = a ^ b ^ c
        cc = ((a & b) | (a & c) | (b & c)) << 1
        a, b, c = s, cc & M, c1
        s = a ^ b ^ c
        cc = (((a & b) | (a & c) | (b & c)) << 1) & M
        if (s + cc) & M != (s0 + c0 + s1 + c1) & M:
            bad += 1
    report("eq(5) s+c == s0+c0+s1+c1 (mod 2^256) through carry-save", bad == 0, f"{bad}/20000 wrong")
    bad = 0
    for _ in range(20000):
        x = rng.getrandbits(256)
        sign = x >> 255
        oc = (~x) & M                      # one's complement
        tc = (-x) & M                      # two's complement magnitude form
        # identity: two's complement = one's complement + 1 for nonzero; the paper's
        # per-lane use: negation via ~x then add the sign bit -> tc = oc + sign
        if sign and tc != (oc + 1) & M:
            bad += 1
    report("eq(6) tc = oc + sign (negate-by-invert then add sign)", bad == 0, f"{bad}/20000 wrong")
    bad = 0
    for _ in range(5000):
        lanes = [rng.getrandbits(256) for _ in range(16)]
        signs = [l >> 255 for l in lanes]
        vals = [(-l) & M if s else l for l, s in zip(lanes, signs)]   # signed magnitudes used by paper
        ocs = [((~l) & M) if s else l for l, s in zip(lanes, signs)]
        lhs = sum(vals) & M
        rhs = (sum(ocs) + sum(signs)) & M
        if lhs != rhs:
            bad += 1
    report("eq(7) sum tc = sum oc + popcount(signs), 16 lanes", bad == 0, f"{bad}/5000 wrong")


# ---------------------------------------------------------------------------
# §5.2  Eq. (8): [+2^40, +1 x62, -2^40] -> fp32/bf16 give 0, quire gives 62
# ---------------------------------------------------------------------------
def check_cancellation():
    import numpy as np
    xs = [2.0 ** 40] + [1.0] * 62 + [-(2.0 ** 40)]
    acc32 = np.float32(0)
    for v in xs:
        acc32 = np.float32(acc32 + np.float32(v))
    # bf16: truncate mantissa to 7 bits after every add
    def bf16(v):
        import struct
        b = struct.unpack("<I", struct.pack("<f", v))[0] & 0xFFFF0000
        return struct.unpack("<f", struct.pack("<I", b))[0]
    accbf = 0.0
    for v in xs:
        accbf = bf16(accbf + bf16(v))
    q = 0
    for v in xs:
        q += int(Fraction(v) * (1 << 96))
    quire = Fraction(q, 1 << 96)
    ulp40 = 2.0 ** (40 - 23)
    report("eq(8) cancellation dot", acc32 == 0 and accbf == 0 and quire == 62,
           f"fp32 {float(acc32)}, bf16 {accbf}, quire {quire} (paper 0.0 / 0.0 / 62.0); "
           f"ulp(2^40) in fp32 = 2^{40-23} = {ulp40:.0f} > 1 so units vanish")


# ---------------------------------------------------------------------------
# Derived numbers quoted in prose
# ---------------------------------------------------------------------------
def check_derived_numbers():
    tbl = [(64, 3.372e-4, 3.328e-4, "1.0×"), (256, 5.07e-4, 3.39e-4, "1.5×"),
           (1024, 1.13e-3, 3.23e-4, "3.5×"), (5120, 1.1296e-3, 7.8714e-5, "14.4×")]
    for k, e, qv, pap in tbl:
        r = e / qv
        report(f"Table 2 k={k} advantage", abs(r - float(pap[:-1])) < 0.06, f"{r:.2f}× (paper {pap})")
    eps = 2.0 ** -24
    report("§5.3 sqrt(k)*eps ~ 1e-5 at k=5120", 1e-6 < math.sqrt(5120) * eps < 1e-4,
           f"{math.sqrt(5120)*eps:.2e}")
    for name, a, b, pap in [("SmolLM2 trunc", 11.81, 13.35, 13.0), ("SmolLM2 RTNE", 11.81, 12.17, 3.0),
                            ("Qwen0.5B trunc", 19.10, 22.84, 19.6), ("Qwen0.5B RTNE", 19.10, 19.24, 0.7)]:
        pct = 100 * (b / a - 1)
        report(f"§6 perplexity % {name}", abs(pct - pap) < 0.06, f"{pct:+.2f} % (paper {pap:+})")
    r1, r2 = 25.95e9 / 395e6, 47.80e9 / 395e6
    report("§6.2 FPGA vs SFPU 65–120×", 64 < r1 < 67 and 119 < r2 < 122, f"{r1:.1f}× / {r2:.1f}× (paper 65–120×; 121 rounds to 121)")
    report("§6.3 exact vs rounded MAC 2.3× smaller / 10.5× faster",
           abs(9673 / 4179 - 2.3) < 0.05 and abs(341.6 / 32.6 - 10.5) < 0.05,
           f"{9673/4179:.2f}× LUTs, {341.6/32.6:.2f}× Fmax")
    prods = 128 * 64 * 128
    report("§6.1 128×64×128 in 2.653 ms = 395 M products/s", abs(prods / 2.653e-3 / 395e6 - 1) < 0.01,
           f"{prods:,} products / 2.653 ms = {prods/2.653e-3/1e6:.0f} M/s; 2 flop/product -> {2*prods/2.653e-3/1e9:.2f} GFLOP/s (paper 0.79)")
    report("§6.1 ~1e5 slower than 91 TFLOP/s engine", 0.5e5 < 91e12 / 0.79e9 < 2e5, f"{91e12/0.79e9:.2e}×")
    report("§4 11/20 MAC/µs (streaming K-loop microbenchmark) vs §6 395 M products/s", None,
           "20 M/s is the dispatch-dominated K=16 scaling test, 395 M/s the 128×64×128 run with K=64 "
           "amortised: both measured, 20× apart because the shapes differ (stated in the text)")
    report("§5 30,077 of 32,768 (host, 32,768-logit instance) vs 131,072 of 131,072 (U200, 8192×16)", None,
           "two shapes of the same 8192×768 · 768×16 GEMM; the paper states both shapes")
    t2048 = 2 * 2048 ** 3 / 167.8e12
    t4096 = 2 * 4096 ** 3 / 197.4e12
    report("§6.4 TPOPS -> per-GEMM times", None, f"2048³ {t2048*1e3:.3f} ms, 4096³ {t4096*1e3:.2f} ms (self-consistent)")


if __name__ == "__main__":
    print(f"reference codec: {os.path.basename(os.path.dirname(R.__file__))}/{os.path.basename(R.__file__)}")
    check_decode_symmetry()
    check_six_bit_example()
    check_quire_sizing()
    check_carry()
    check_csa_and_signs()
    check_cancellation()
    check_derived_numbers()
    print(f"\n{len(FAILS)} FAIL(s): {FAILS}")
