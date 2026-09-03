#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate the bp16 exact-quire MULTI-CORE MATMUL golden from the canonical oracle.

A correctness tile sized to need MANY cores (one output element per baby core on
an 8x8 grid), NOT a perf kernel:

    C[M,N] = A[M,K] . B[K,N]    with  M = N = 8,  K = 16

so there are M*N = 64 output elements, each an EXACT-quire dot product of length
K=16:  C[i,j] = Sum_k A[i,k] * B[k,j].  Every per-(i,j,k) product is the EXACT
bp16xbp16 product placed into a 256-bit Kulisch quire (QUIRE_FRAC_BITS=96) with NO
per-product rounding (mirrors kernel/bp16_quire.h bp16_prod_to_q256); the K-length
accumulation is exact integer add (associative); the single rounding is in the
per-output quire->bp16 readout (quire256_to_bposit16 == bp16_encode_quire256).

This is the SAME math as gen_matmul_golden.py (single-core 4x16x4 tile) at a
larger M=N=8 so the 64 outputs map one-per-core onto an 8x8 baby-core grid. Same
cross-checks:
  (1) every per-element product vs the oracle bposit16_mul (the rounded mul code);
  (2) any (a,b) pair present in golden/bp16_mul.json must match its prod_code;
  (3) per-output quire is order-independent (forward == reverse == even/odd shuffle).

Emits the header `quire_matmul_mc_golden_cases.h` (operand code matrices A,B in
row-major, per-output expected bp16 readout codes, the full expected quire of one
gated output for a byte-level gate, and per-output-element bposit16_mul cross-check
codes for the load-bearing invariant). Run:

    python3 gen_matmul_mc_golden.py > quire_matmul_mc_golden_cases.h
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

# ---- tile dims (sized so 64 outputs map one-per-core onto an 8x8 grid) -------
M = 8   # rows of A / rows of C
K = 16  # contraction length (matches the proven dot/matmul K=16)
N = 8   # cols of B / cols of C


def C_(x):
    return encode_bposit16(Fraction(x)) & 0xFFFF


def code_fraction(code):
    d = decode_bposit16(code)
    if d.is_special:  # zero or NaR -> contributes 0 (same as bp16_prod_to_q256)
        return Fraction(0)
    return decoded_to_fraction(d)


def exact_prod_to_quire_int(acode, bcode):
    """Exact a*b as a signed quire integer (val * 2^96, truncated toward zero).
    For dyadic bp16 products the *2^96 scaling is exact as long as
    E2_a+E2_b >= -96, which holds for the chosen operands."""
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


# ---- operand matrices --------------------------------------------------------
# Values are exact small dyadics (the quire's golden zone) plus a couple of zero
# edges so the tile exercises sign, variable shift, cancellation, and the readout
# MSB scan across 64 distinct output dots -- without NaR (NaR breaks the
# per-element invariant; see gen_dot/gen_matmul). The 8x16 A and 16x8 B are built
# by a deterministic dyadic pattern so every output row/col differs.
_DYADIC = [
    Fraction(1), Fraction(1, 2), Fraction(3, 2), Fraction(2),
    Fraction(7, 4), Fraction(1, 8), Fraction(-2), Fraction(9, 2),
    Fraction(11, 4), Fraction(3), Fraction(1, 4), Fraction(5, 2),
    Fraction(-1), Fraction(-1, 2), Fraction(-3, 2), Fraction(4),
    Fraction(-4), Fraction(5, 4), Fraction(3, 4), Fraction(-1, 4),
    Fraction(7, 8), Fraction(-9, 4), Fraction(1), Fraction(-1),
]


def a_val(i, k):
    # deterministic, row-distinct; a couple of structural zeros for cancellation
    if (i, k) in ((2, 3), (2, 11), (5, 7), (6, 0)):
        return Fraction(0)
    return _DYADIC[(i * 5 + k * 3) % len(_DYADIC)]


def b_val(k, j):
    if (k, j) in ((4, 4), (9, 1), (13, 6)):
        return Fraction(0)
    return _DYADIC[(k * 7 + j * 2 + 1) % len(_DYADIC)]


A = [[C_(a_val(i, k)) for k in range(K)] for i in range(M)]   # M x K codes
B = [[C_(b_val(k, j)) for j in range(N)] for k in range(K)]   # K x N codes

assert len(A) == M and all(len(r) == K for r in A)
assert len(B) == K and all(len(r) == N for r in B)

# ---- per-output exact quire dot + readout (the matmul) -----------------------


def accumulate(i, j, order):
    q = 0
    for k in order:
        q += exact_prod_to_quire_int(A[i][k], B[k][j])
    return q


out_quires = [[0] * N for _ in range(M)]
out_readout = [[0] * N for _ in range(M)]
for i in range(M):
    for j in range(N):
        q_fwd = accumulate(i, j, range(K))
        q_rev = accumulate(i, j, reversed(range(K)))
        q_shuf = accumulate(i, j, [k for k in range(0, K, 2)] + [k for k in range(1, K, 2)])
        assert q_fwd == q_rev == q_shuf, f"output ({i},{j}) accumulation not order-independent"
        out_quires[i][j] = q_fwd
        out_readout[i][j] = quire256_to_bposit16(q_fwd) & 0xFFFF

# ---- cross-check (1): per-element bposit16_mul of EVERY product -------------
# The load-bearing invariant: the EXACT per-element quire contribution, read back
# ALONE, must equal the standalone oracle bposit16_mul for every (i,k,j) triple.
prod_codes = [[[0] * N for _ in range(K)] for _ in range(M)]  # [i][k][j]
for i in range(M):
    for k in range(K):
        for j in range(N):
            pm = bposit16_mul(A[i][k], B[k][j]) & 0xFFFF
            prod_codes[i][k][j] = pm
            qi = exact_prod_to_quire_int(A[i][k], B[k][j])
            ri = quire256_to_bposit16(qi) & 0xFFFF
            assert ri == pm, (
                f"per-elem exact readout ({i},{k},{j}) 0x{ri:04x} != bposit16_mul 0x{pm:04x}")

# ---- cross-check (2): against golden/bp16_mul.json where (a,b) overlaps ------
mulj = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../golden/bp16_mul.json")))
mul_map = {(e["a_code"] & 0xFFFF, e["b_code"] & 0xFFFF): e["prod_code"] & 0xFFFF
           for e in mulj["entries"]}
overlap = 0
for i in range(M):
    for k in range(K):
        for j in range(N):
            key = (A[i][k], B[k][j])
            if key in mul_map:
                assert prod_codes[i][k][j] == mul_map[key], (
                    f"product ({i},{k},{j}) {key} oracle=0x{prod_codes[i][k][j]:04x} "
                    f"!= bp16_mul.json 0x{mul_map[key]:04x}")
                overlap += 1

# ---- one full quire baked for a byte-level gate: output (M-1,N-1) ------------
# Pick the LAST output so the gate exercises the highest-index core in the grid.
GATE_I, GATE_J = M - 1, N - 1
gate_q = out_quires[GATE_I][GATE_J]
gate_limbs = quire_to_le_u32x8(gate_q)
gate_hex = le_hex(gate_limbs)

# ---- emit the header ---------------------------------------------------------
W = sys.stdout.write


def fmt_codes_2d(name, mat, rows, cols):
    W(f"static const int32_t {name}[{rows} * {cols}] = {{\n")
    for r in range(rows):
        W("    " + " ".join("0x%04x," % (mat[r][c] & 0xFFFF) for c in range(cols)))
        W(f"  /* row {r} */\n")
    W("};\n")


W("/* quire_matmul_mc_golden_cases.h — baked golden for the exact 256-bit\n")
W(" * b-posit16 quire MULTI-CORE MATMUL  C[M,N] = A[M,K] . B[K,N]  (M=N=8, K=16):\n")
W(" * 64 output elements, each an EXACT-quire dot of length K, distributed one per\n")
W(" * baby (RISC-V) core across an 8x8 grid. Same math as the single-core\n")
W(" * quire_matmul_golden_cases.h, sized so the outputs map one-per-core.\n")
W(" *\n")
W(" * DO NOT hand-edit the numbers — GENERATED from the canonical oracle and\n")
W(" * re-verified by the host program at startup against the reused\n")
W(" * bp16_quire.h / bp16_encode.h headers (belt-and-suspenders).\n")
W(" *\n")
W(" * SOURCE OF TRUTH (regenerate, never by hand):\n")
W(" *   oracle: mosyne-bposit/kernels/bposit16_reference.py\n")
W(" *           (encode_bposit16, decode_bposit16, decoded_to_fraction,\n")
W(" *            bposit16_mul, quire256_to_bposit16, QUIRE_FRAC_BITS=96)\n")
W(" *   generator: gen_matmul_mc_golden.py (kept with this commit)\n")
W(" *   cross-checked vs golden/bp16_mul.json (operand pairs that overlap)\n")
W(" *\n")
W(" * Layout: A is M*K row-major, B is K*N row-major, C is M*N row-major.\n")
W(" * C[i,j] = Sum_k A[i*K+k] * B[k*N+j], each product EXACT (no per-product\n")
W(" * rounding) into one quire; q256_add is exact ⇒ K-accumulation order-\n")
W(" * independent (forward == reverse == even/odd shuffle, verified). Only the\n")
W(" * per-output quire->bp16 readout rounds (truncation).\n")
W(" *\n")
W(" * Verified locally (gcc, reusing the EXACT the scalar-kernel C kernel headers the kernel\n")
W(" * #includes — bp16_prod_to_q256 + q256_add + bp16_encode_quire256):\n")
W(" *   - each C[i,j] readout == oracle quire256_to_bposit16 of its exact dot quire\n")
W(" *   - the full output-(M-1,N-1) quire == byte-identical golden below\n")
W(" *   - every per-element exact product == bposit16_mul(A[i,k], B[k,j])\n")
W(f" *   - {overlap} operand pairs == golden/bp16_mul.json prod_code\n")
W(" */\n")
W("#ifndef QUIRE_MATMUL_MC_GOLDEN_CASES_H\n")
W("#define QUIRE_MATMUL_MC_GOLDEN_CASES_H\n\n")
W("#include <cstdint>\n\n")
W(f"#define QGOLD_MMMC_M {M}\n")
W(f"#define QGOLD_MMMC_K {K}\n")
W(f"#define QGOLD_MMMC_N {N}\n\n")

W("/* A operand matrix, M x K, row-major (16-bit codes). */\n")
fmt_codes_2d("QGOLD_MMMC_A_CODES", A, M, K)
W("\n/* B operand matrix, K x N, row-major (16-bit codes). */\n")
fmt_codes_2d("QGOLD_MMMC_B_CODES", B, K, N)

W("\n/* expected C readouts, M x N row-major: C[i*N+j] = truncating bp16 readout\n")
W(" * of the exact quire dot Sum_k A[i,k]*B[k,j]. The host gates all M*N. */\n")
fmt_codes_2d("QGOLD_MMMC_C_READOUT", out_readout, M, N)

W("\n/* per-(i,k,j) standalone bposit16_mul codes, flattened [i*K*N + k*N + j].\n")
W(" * The host self-check re-derives each exact product, reads it back ALONE, and\n")
W(" * asserts it equals this — the load-bearing per-element invariant. */\n")
W(f"static const int32_t QGOLD_MMMC_PROD_CODES[{M} * {K} * {N}] = {{\n")
for i in range(M):
    for k in range(K):
        W("    " + " ".join("0x%04x," % (prod_codes[i][k][j] & 0xFFFF) for j in range(N)))
        W(f"  /* i={i} k={k:2d} */\n")
W("};\n")

W("\n/* one FULL expected quire baked for a byte-level gate: output (%d,%d),\n" % (GATE_I, GATE_J))
W(" * 8x uint32 little-endian two's-complement. */\n")
W(f"#define QGOLD_MMMC_GATE_I {GATE_I}\n")
W(f"#define QGOLD_MMMC_GATE_J {GATE_J}\n")
W("static const uint32_t QGOLD_MMMC_GATE_QUIRE[8] = {\n")
W("    " + " ".join("0x%08xu," % l for l in gate_limbs[:4]) + "\n")
W("    " + " ".join("0x%08xu," % l for l in gate_limbs[4:]) + "\n")
W("};\n")
W(f'static const char QGOLD_MMMC_GATE_QUIRE_HEX[] =\n    "{gate_hex}";\n')
W("\n#endif /* QUIRE_MATMUL_MC_GOLDEN_CASES_H */\n")

# ---- human-readable summary to stderr (does not pollute the header) ----------
sys.stderr.write("// tile: C[%d,%d] = A[%d,%d] . B[%d,%d]  (%d outputs)\n" % (M, N, M, K, K, N, M * N))
sys.stderr.write("// bp16_mul.json overlap checked: %d\n" % overlap)
for i in range(M):
    sys.stderr.write("// C readout row %d: %s\n" % (
        i, " ".join("0x%04x" % out_readout[i][j] for j in range(N))))
sys.stderr.write("// gate quire (%d,%d): %s\n" % (GATE_I, GATE_J, gate_hex))
sys.stderr.write("// gate readout (%d,%d): 0x%04x\n" % (
    GATE_I, GATE_J, out_readout[GATE_I][GATE_J]))
