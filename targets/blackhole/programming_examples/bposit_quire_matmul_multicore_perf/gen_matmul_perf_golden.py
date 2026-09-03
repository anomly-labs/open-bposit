#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate the bp16 exact-quire MULTI-CORE-PERF MATMUL golden (SAMPLED).

This is the THROUGHPUT variant of bposit_quire_matmul_multicore. It scales the
SAME proven scalar exact-quire dot to a LARGE matmul that saturates the full
Blackhole baby-core grid:

    C[N,N] = A[N,K] . B[K,N]    with  N = 128,  K = 64

so there are N*N = 16384 output elements, each an EXACT-quire dot of length
K = 64:  C[i,j] = Sum_k A[i,k] * B[k,j].  That is N*N*K = 1,048,576 EXACT
bp16xbp16 products fed into 256-bit Kulisch quires (QUIRE_FRAC_BITS = 96) with NO
per-product rounding (mirrors kernel/bp16_quire.h bp16_prod_to_q256); the K-length
accumulation is exact integer add (associative); the single rounding is in the
per-output quire->bp16 readout (quire256_to_bposit16 == bp16_encode_quire256).

CORRECTNESS WITHOUT A HUGE-GOLDEN BOTTLENECK: the pure-Python Fraction oracle is
too slow to gate all N*N*K ~ 1.05M products. Instead we gate a SAMPLE of S = 64
outputs spread across the whole output grid (a coarse stride over (i,j)) plus one
fully-baked gate quire. The DEVICE still computes the FULL N*N matmul and the host
TIMES it; only these S sampled outputs are checked bit-exact. The sample spans the
4 grid corners, center, and a deterministic diagonal/stride spread so every region
of the core grid is represented.

Same numeric cross-checks as the small example, but applied only to the SAMPLED
outputs and their K products (keeps gen fast):
  (1) every per-element product (of a sampled output) vs the oracle bposit16_mul;
  (2) any (a,b) pair present in golden/bp16_mul.json must match its prod_code;
  (3) each sampled output quire is order-independent (forward == reverse).

Emits `quire_matmul_perf_golden_cases.h`:
  - full operand code matrices A (N*K) and B (K*N) row-major (these ARE needed on
    device for the full matmul — they are the real inputs, not just sample data);
  - the SAMPLE list of S flat output indices;
  - the expected bp16 readout for each sampled output;
  - the full expected quire of one gated sampled output (byte-level gate).

Run:
    python3 gen_matmul_perf_golden.py > quire_matmul_perf_golden_cases.h
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get("BPOSIT16_REFERENCE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../mosyne-bposit/kernels")))  # bposit16_reference.py from github.com/anomly-labs/mosyne-bposit
from fractions import Fraction
from bposit16_reference import (  # noqa: E402
    encode_bposit16,
    decode_bposit16,
    decoded_to_fraction,
    bposit16_mul,
    quire256_to_bposit16,
    QUIRE_FRAC_BITS,
)

MASK256 = (1 << 256) - 1

# ---- perf dims (large enough to saturate ~140 baby cores with many outs/core) -
N = 128  # square C[N,N]; A is N x K, B is K x N
K = 64   # contraction length (4x the small example's K=16)
# 16384 outputs across ~140 cores => ~117 outputs/core. 16384*64 = 1,048,576
# EXACT bp16 products. Per-core L1 for a FULL private copy: A=N*K and B=K*N codes
# = 8192 + 8192 words = 64 KB, well within the Blackhole 1536 KB L1.

# ---- sampling: S outputs spread across the whole output grid ------------------
# A deterministic spread so every region of the 14x10 core grid is represented:
# the 4 corners + center, then a strided diagonal/lattice fill up to S total.
S = 64


def C_(x):
    return encode_bposit16(Fraction(x)) & 0xFFFF


def code_fraction(code):
    d = decode_bposit16(code)
    if d.is_special:  # zero or NaR -> contributes 0 (same as bp16_prod_to_q256)
        return Fraction(0)
    return decoded_to_fraction(d)


def exact_prod_to_quire_int(acode, bcode):
    """Exact a*b as a signed quire integer (val * 2^96, truncated toward zero).
    Exact for dyadic bp16 products as long as E2_a+E2_b >= -96 (holds here)."""
    prod = code_fraction(acode) * code_fraction(bcode)
    return int(prod * (1 << QUIRE_FRAC_BITS))


def quire_to_le_u32x8(q_signed):
    q = q_signed & MASK256
    return [(q >> (32 * i)) & 0xFFFFFFFF for i in range(8)]


def le_hex(limbs):
    s = ""
    for limb in limbs:
        for b in range(4):
            s += "%02x" % ((limb >> (8 * b)) & 0xFF)
    return s


# ---- operand matrices (deterministic dyadics in the quire golden zone) --------
# Same dyadic alphabet as the small example, plus structural zeros, extended over
# the larger N x K / K x N shapes. Values stay small so K=64 sums never approach
# the 256-bit quire bounds and every product is exact (E2_a+E2_b+96 in [0,255]).
_DYADIC = [
    Fraction(1), Fraction(1, 2), Fraction(3, 2), Fraction(2),
    Fraction(7, 4), Fraction(1, 8), Fraction(-2), Fraction(9, 2),
    Fraction(11, 4), Fraction(3), Fraction(1, 4), Fraction(5, 2),
    Fraction(-1), Fraction(-1, 2), Fraction(-3, 2), Fraction(4),
    Fraction(-4), Fraction(5, 4), Fraction(3, 4), Fraction(-1, 4),
    Fraction(7, 8), Fraction(-9, 4), Fraction(1), Fraction(-1),
]


def a_val(i, k):
    # deterministic, row-distinct; sprinkle structural zeros for cancellation
    if (i * 7 + k * 5) % 37 == 0:
        return Fraction(0)
    return _DYADIC[(i * 5 + k * 3) % len(_DYADIC)]


def b_val(k, j):
    if (k * 11 + j * 3) % 41 == 0:
        return Fraction(0)
    return _DYADIC[(k * 7 + j * 2 + 1) % len(_DYADIC)]


A = [[C_(a_val(i, k)) for k in range(K)] for i in range(N)]   # N x K codes
B = [[C_(b_val(k, j)) for j in range(N)] for k in range(K)]   # K x N codes

assert len(A) == N and all(len(r) == K for r in A)
assert len(B) == K and all(len(r) == N for r in B)


# ---- choose the S sampled output indices, spread across the grid -------------
def sample_indices():
    idx = []
    # corners + center first (high-value coverage)
    corners = [(0, 0), (0, N - 1), (N - 1, 0), (N - 1, N - 1), (N // 2, N // 2)]
    for (i, j) in corners:
        idx.append(i * N + j)
    # then a strided lattice so the rest of the grid is covered deterministically
    step = max(1, (N * N) // (S * 3))
    o = 0
    seen = set(idx)
    while len(idx) < S:
        o = (o + step + 1) % (N * N)
        if o not in seen:
            seen.add(o)
            idx.append(o)
    return sorted(idx[:S])


SAMPLE = sample_indices()
assert len(SAMPLE) == S


# ---- exact quire dot + readout for the SAMPLED outputs only -------------------
def accumulate(i, j, order):
    q = 0
    for k in order:
        q += exact_prod_to_quire_int(A[i][k], B[k][j])
    return q


sample_readout = []
for o in SAMPLE:
    i, j = o // N, o % N
    q_fwd = accumulate(i, j, range(K))
    q_rev = accumulate(i, j, reversed(range(K)))
    assert q_fwd == q_rev, f"sampled output ({i},{j}) accumulation not order-independent"
    sample_readout.append(quire256_to_bposit16(q_fwd) & 0xFFFF)

# ---- cross-check (1): per-element bposit16_mul of each SAMPLED output's products
for o in SAMPLE:
    i, j = o // N, o % N
    for k in range(K):
        pm = bposit16_mul(A[i][k], B[k][j]) & 0xFFFF
        qi = exact_prod_to_quire_int(A[i][k], B[k][j])
        ri = quire256_to_bposit16(qi) & 0xFFFF
        assert ri == pm, (
            f"per-elem exact readout ({i},{k},{j}) 0x{ri:04x} != bposit16_mul 0x{pm:04x}")

# ---- cross-check (2): against golden/bp16_mul.json where (a,b) overlaps -------
mulj = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../golden/bp16_mul.json")))
mul_map = {(e["a_code"] & 0xFFFF, e["b_code"] & 0xFFFF): e["prod_code"] & 0xFFFF
           for e in mulj["entries"]}
overlap = 0
for o in SAMPLE:
    i, j = o // N, o % N
    for k in range(K):
        key = (A[i][k], B[k][j])
        if key in mul_map:
            pm = bposit16_mul(A[i][k], B[k][j]) & 0xFFFF
            assert pm == mul_map[key], (
                f"product ({i},{k},{j}) {key} oracle=0x{pm:04x} "
                f"!= bp16_mul.json 0x{mul_map[key]:04x}")
            overlap += 1

# ---- one full quire baked for a byte-level gate: a sampled corner output ------
# Use the last output (N-1,N-1): it is in SAMPLE and lives on a far corner core.
GATE_O = (N - 1) * N + (N - 1)
assert GATE_O in SAMPLE
GATE_I, GATE_J = GATE_O // N, GATE_O % N
gate_q = accumulate(GATE_I, GATE_J, range(K))
gate_limbs = quire_to_le_u32x8(gate_q)
gate_hex = le_hex(gate_limbs)
gate_readout = quire256_to_bposit16(gate_q) & 0xFFFF

# ---- emit the header ---------------------------------------------------------
W = sys.stdout.write


def fmt_codes_2d(name, mat, rows, cols):
    W(f"static const int32_t {name}[{rows} * {cols}] = {{\n")
    for r in range(rows):
        W("    " + " ".join("0x%04x," % (mat[r][c] & 0xFFFF) for c in range(cols)))
        W(f"  /* row {r} */\n")
    W("};\n")


W("/* quire_matmul_perf_golden_cases.h — baked golden (SAMPLED) for the exact\n")
W(" * 256-bit b-posit16 quire MULTI-CORE-PERF MATMUL  C[N,N] = A[N,K] . B[K,N]\n")
W(f" * (N={N}, K={K}): {N*N} output elements, each an EXACT-quire dot of length K,\n")
W(f" * distributed across the full ~140-core Blackhole baby-core grid ({N*N*K}\n")
W(" * exact bp16 products total). The device computes the FULL matmul and the host\n")
W(" * TIMES it; correctness is gated on a SAMPLE of S outputs spread across the\n")
W(" * grid (the Fraction oracle is too slow to gate all N*N*K products).\n")
W(" *\n")
W(" * DO NOT hand-edit the numbers — GENERATED from the canonical oracle and\n")
W(" * re-verified by the host program at startup against the reused\n")
W(" * bp16_quire.h / bp16_encode.h headers (belt-and-suspenders, on the sample).\n")
W(" *\n")
W(" * SOURCE OF TRUTH (regenerate, never by hand):\n")
W(" *   oracle: mosyne-bposit/kernels/bposit16_reference.py\n")
W(" *   generator: gen_matmul_perf_golden.py (kept with this commit)\n")
W(" *   cross-checked vs golden/bp16_mul.json (operand pairs that overlap)\n")
W(" *\n")
W(" * Layout: A is N*K row-major, B is K*N row-major, C is N*N row-major.\n")
W(" * C[i,j] = Sum_k A[i*K+k] * B[k*N+j], each product EXACT (no per-product\n")
W(" * rounding) into one quire; q256_add is exact => K-accumulation order-\n")
W(" * independent (forward == reverse, verified on the sample). Only the per-output\n")
W(" * quire->bp16 readout rounds (truncation).\n")
W(" */\n")
W("#ifndef QUIRE_MATMUL_PERF_GOLDEN_CASES_H\n")
W("#define QUIRE_MATMUL_PERF_GOLDEN_CASES_H\n\n")
W("#include <cstdint>\n\n")
W(f"#define QGOLD_MMP_N {N}\n")
W(f"#define QGOLD_MMP_K {K}\n")
W(f"#define QGOLD_MMP_S {S}\n\n")

W("/* A operand matrix, N x K, row-major (16-bit codes) — the REAL device input. */\n")
fmt_codes_2d("QGOLD_MMP_A_CODES", A, N, K)
W("\n/* B operand matrix, K x N, row-major (16-bit codes) — the REAL device input. */\n")
fmt_codes_2d("QGOLD_MMP_B_CODES", B, K, N)

W("\n/* SAMPLE: S flat output indices (sorted) spread across the output grid. */\n")
W(f"static const int32_t QGOLD_MMP_SAMPLE_O[{S}] = {{\n")
for r in range(0, S, 8):
    W("    " + " ".join("%6d," % SAMPLE[r + c] for c in range(min(8, S - r))) + "\n")
W("};\n")

W("\n/* expected bp16 readout for each sampled output (same order as SAMPLE_O). */\n")
W(f"static const int32_t QGOLD_MMP_SAMPLE_READOUT[{S}] = {{\n")
for r in range(0, S, 8):
    W("    " + " ".join("0x%04x," % (sample_readout[r + c] & 0xFFFF)
                        for c in range(min(8, S - r))) + "\n")
W("};\n")

W("\n/* one FULL expected quire baked for a byte-level gate: output (%d,%d),\n" % (GATE_I, GATE_J))
W(" * 8x uint32 little-endian two's-complement. This output is in SAMPLE. */\n")
W(f"#define QGOLD_MMP_GATE_O {GATE_O}\n")
W(f"#define QGOLD_MMP_GATE_I {GATE_I}\n")
W(f"#define QGOLD_MMP_GATE_J {GATE_J}\n")
W(f"#define QGOLD_MMP_GATE_READOUT 0x{gate_readout:04x}\n")
W("static const uint32_t QGOLD_MMP_GATE_QUIRE[8] = {\n")
W("    " + " ".join("0x%08xu," % l for l in gate_limbs[:4]) + "\n")
W("    " + " ".join("0x%08xu," % l for l in gate_limbs[4:]) + "\n")
W("};\n")
W(f'static const char QGOLD_MMP_GATE_QUIRE_HEX[] =\n    "{gate_hex}";\n')
W("\n#endif /* QUIRE_MATMUL_PERF_GOLDEN_CASES_H */\n")

# ---- human-readable summary to stderr (does not pollute the header) ----------
sys.stderr.write("// perf tile: C[%d,%d] = A[%d,%d] . B[%d,%d]  (%d outputs, %d products)\n"
                 % (N, N, N, K, K, N, N * N, N * N * K))
sys.stderr.write("// sampled outputs gated: %d ; bp16_mul.json overlap: %d\n" % (S, overlap))
sys.stderr.write("// gate (%d,%d): quire %s  readout 0x%04x\n"
                 % (GATE_I, GATE_J, gate_hex, gate_readout))
