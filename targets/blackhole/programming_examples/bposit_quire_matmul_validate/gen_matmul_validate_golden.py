#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate the bp16 exact-quire FULL-VALIDATION MATMUL golden from the oracle.

This is the CORRECTNESS-HARDENING golden: instead of one engineered tile (the
single-/multi-core matmul examples) or a large SAMPLED tile (the perf example),
it bakes SEEDS distinct RANDOM matmuls so the on-silicon host can gate EVERY
output of EVERY seed bit-exact vs the exact 256-bit Kulisch-quire oracle:

    for seed in 0..SEEDS-1:
        C_seed[M,N] = A_seed[M,K] . B_seed[K,N]

with M = N = 16, K = 16  ->  M*N = 256 outputs/seed, SEEDS = 8 distinct random
seeds  ->  2048 fully-validated outputs and M*N*K*SEEDS = 32768 EXACT bp16
products. Every per-(i,j,k) product is the EXACT bp16xbp16 product placed into a
256-bit quire (QUIRE_FRAC_BITS=96) with NO per-product rounding (mirrors
kernel/bp16_quire.h bp16_prod_to_q256); the K-length accumulation is exact
integer add (associative); the single rounding is the per-output quire->bp16
readout (quire256_to_bposit16 == bp16_encode_quire256).

The whole point is that the inputs are RANDOM, not hand-picked: a paper reviewer
wants "we ran N random MxKxN matmuls and ALL M*N*SEEDS outputs were bit-exact vs
the exact-quire reference", converting the headline claim from "sampled" to
"fully validated across random data".

REALISTIC DATA MIX (per seed):
  - golden-zone seeds: entries ~ Normal(0, 0.3) (the quire's sweet spot);
  - wide-dynamic-range / cancellation seeds: a mix of large (|.|~2^8), tiny
    (|.|~2^-8) and -large entries, plus structural zeros, to stress the quire's
    256-bit Kulisch range and exercise catastrophic cancellation.
All values are bp16-quantized via the oracle encode_bposit16 before use.

EXACTNESS GUARANTEE (why this stays bit-exact, no caveats):
  The oracle exact_prod_to_quire_int does int(prod*2^96) and the C kernel
  bp16_prod_to_q256 places M_a*M_b at bit (E2_a+E2_b+96), DROPPING the whole
  term if E2_a+E2_b < 0 -> -96, i.e. if E2_a+E2_b < -96. These two agree exactly
  iff every nonzero product has E2_a+E2_b >= -96 (term fully representable in the
  quire) OR is exactly zero. We therefore REJECT-AND-RESEED any (A,B) whose
  product set would land a nonzero term below bit 0 of the quire. With the chosen
  value ranges this essentially never triggers, but the check makes the
  bit-exactness an invariant of the emitted data, not a hope. The host program
  ALSO independently re-derives every output from these codes using the reused C
  headers at startup, so a drift would FAIL loudly before any device run.

Cross-checks (same as gen_matmul_mc_golden.py, applied to every seed):
  (1) every per-element exact product read back ALONE == oracle bposit16_mul;
  (2) any (a,b) pair present in golden/bp16_mul.json matches its prod_code;
  (3) per-output quire is order-independent (forward == reverse == even/odd).

Emits `quire_matmul_validate_golden_cases.h`. Run:

    python3 gen_matmul_validate_golden.py > quire_matmul_validate_golden_cases.h
"""
import json
import random
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

# ---- tile dims + seed count (the prompt's recommended full-validation size) --
# M=N=16, K=16 -> 256 outputs/seed; SEEDS=8 -> 2048 fully-validated outputs,
# 32768 exact products. Sized so the pure-Python Fraction oracle finishes the
# golden gen in a few minutes and the on-silicon run is bounded.
M = 16   # rows of A / rows of C
K = 16   # contraction length (matches the proven dot/matmul K=16)
N = 16   # cols of B / cols of C
SEEDS = 8

# Per-seed value regime: golden-zone vs wide-dynamic-range. Wide seeds stress the
# quire's range and cancellation; golden-zone seeds are the common case.
WIDE_SEEDS = {3, 6}            # which seed indices use the wide / cancellation mix
RANDOM_BASE_SEED = 0x5D575641  # fixed seed — deterministic so the golden is reproducible


def code_e2(code):
    """(M, E2) of a bp16 code: value = +-M * 2^E2 (mirrors bp16_decode.h).
    Returns (0, 0) for zero / NaR. Used only to enforce the quire-exactness
    invariant E2_a+E2_b >= -96 for every nonzero product."""
    d = decode_bposit16(code & 0xFFFF)
    if d.is_special:  # "zero" or "nar" -> contributes 0 to the quire
        return (0, 0)
    fw = d.f_width
    mant = (1 << fw) + d.f_bits if fw > 0 else 1
    e2 = 8 * d.k + d.e - (fw if fw > 0 else 0)
    return (mant, e2)


def C_(x):
    return encode_bposit16(Fraction(x)) & 0xFFFF


def code_fraction(code):
    d = decode_bposit16(code)
    if d.is_special:  # zero or NaR -> contributes 0 (same as bp16_prod_to_q256)
        return Fraction(0)
    return decoded_to_fraction(d)


def exact_prod_to_quire_int(acode, bcode):
    """Exact a*b as a signed quire integer (val * 2^96, truncated toward zero)."""
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


def draw_value(rng, wide):
    """One random realistic operand value, bp16-quantized via the oracle encode."""
    if not wide:
        # golden zone: Normal(0, 0.3), the quire sweet spot.
        return C_(round(rng.gauss(0.0, 0.3), 6))
    # wide / cancellation mix: large, tiny, -large, zero.
    bucket = rng.random()
    if bucket < 0.12:
        return 0x0000  # structural zero
    if bucket < 0.45:
        # large magnitude, both signs (|.|~ up to 2^8)
        return C_(round(rng.uniform(-256.0, 256.0), 4))
    if bucket < 0.78:
        # tiny magnitude (|.|~ 2^-8 .. 2^-2)
        sgn = -1.0 if rng.random() < 0.5 else 1.0
        return C_(round(sgn * rng.uniform(2 ** -8, 2 ** -2), 8))
    # mid golden-zone
    return C_(round(rng.gauss(0.0, 0.5), 6))


def build_seed_matrices(seed_idx):
    """Build (A, B) code matrices for one seed, reseeding until EVERY nonzero
    product satisfies the quire-exactness invariant E2_a+E2_b >= -96 (see header
    docstring). Returns (A, B) as row-major code lists."""
    wide = seed_idx in WIDE_SEEDS
    attempt = 0
    while True:
        rng = random.Random(RANDOM_BASE_SEED + seed_idx * 1009 + attempt * 7919)
        A = [[draw_value(rng, wide) for _ in range(K)] for _ in range(M)]   # M x K
        B = [[draw_value(rng, wide) for _ in range(N)] for _ in range(K)]   # K x N
        # quire-exactness invariant: any nonzero product term must land at bit
        # (E2_a+E2_b+96) >= 0. (zero terms are fine; both oracle and kernel drop
        # them.) Reject-and-reseed otherwise so bit-exactness is guaranteed.
        ok = True
        for i in range(M):
            for k in range(K):
                ma, ea = code_e2(A[i][k])
                if ma == 0:
                    continue
                for j in range(N):
                    mb, eb = code_e2(B[k][j])
                    if mb == 0:
                        continue
                    if ea + eb < -QUIRE_FRAC_BITS:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return A, B, attempt
        attempt += 1
        if attempt > 256:
            raise RuntimeError(
                f"seed {seed_idx}: could not draw a quire-exact random tile in "
                f"{attempt} attempts (tighten value ranges)")


def accumulate(A, B, i, j, order):
    q = 0
    for k in order:
        q += exact_prod_to_quire_int(A[i][k], B[k][j])
    return q


# ---- build every seed + per-output exact quire dot + readout -----------------
all_A = []          # [seed][i][k]
all_B = []          # [seed][k][j]
all_readout = []    # [seed][i][j]
all_prod = []       # [seed][i][k][j] standalone bposit16_mul codes
all_gate_limbs = [] # [seed] 8 limbs of the gated output (M-1,N-1)
overlap_total = 0
reseeds = []

mulj = json.load(open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../golden/bp16_mul.json")))
mul_map = {(e["a_code"] & 0xFFFF, e["b_code"] & 0xFFFF): e["prod_code"] & 0xFFFF
           for e in mulj["entries"]}

GATE_I, GATE_J = M - 1, N - 1  # gate the last output of each seed

for s in range(SEEDS):
    A, B, attempt = build_seed_matrices(s)
    reseeds.append(attempt)

    readout = [[0] * N for _ in range(M)]
    prod_codes = [[[0] * N for _ in range(K)] for _ in range(M)]
    gate_limbs = None
    for i in range(M):
        for j in range(N):
            q_fwd = accumulate(A, B, i, j, range(K))
            q_rev = accumulate(A, B, i, j, reversed(range(K)))
            q_shuf = accumulate(A, B, i, j,
                                [k for k in range(0, K, 2)] + [k for k in range(1, K, 2)])
            assert q_fwd == q_rev == q_shuf, \
                f"seed {s} output ({i},{j}) accumulation not order-independent"
            readout[i][j] = quire256_to_bposit16(q_fwd) & 0xFFFF
            if i == GATE_I and j == GATE_J:
                gate_limbs = quire_to_le_u32x8(q_fwd)

    # cross-check (1): per-element exact product read back ALONE == bposit16_mul.
    for i in range(M):
        for k in range(K):
            for j in range(N):
                pm = bposit16_mul(A[i][k], B[k][j]) & 0xFFFF
                prod_codes[i][k][j] = pm
                qi = exact_prod_to_quire_int(A[i][k], B[k][j])
                ri = quire256_to_bposit16(qi) & 0xFFFF
                assert ri == pm, (
                    f"seed {s} per-elem exact readout ({i},{k},{j}) "
                    f"0x{ri:04x} != bposit16_mul 0x{pm:04x}")
                # cross-check (2): overlap vs golden/bp16_mul.json
                key = (A[i][k], B[k][j])
                if key in mul_map:
                    assert pm == mul_map[key], (
                        f"seed {s} product ({i},{k},{j}) {key} oracle=0x{pm:04x} "
                        f"!= bp16_mul.json 0x{mul_map[key]:04x}")
                    overlap_total += 1

    all_A.append(A)
    all_B.append(B)
    all_readout.append(readout)
    all_prod.append(prod_codes)
    all_gate_limbs.append(gate_limbs)

# ---- emit the header ---------------------------------------------------------
W = sys.stdout.write

W("/* quire_matmul_validate_golden_cases.h — baked golden for the FULL-VALIDATION\n")
W(" * exact 256-bit b-posit16 quire MATMUL: SEEDS distinct RANDOM matmuls\n")
W(" *   C_seed[M,N] = A_seed[M,K] . B_seed[K,N]   (M=N=16, K=16), SEEDS=8\n")
W(" * = 2048 fully-validated outputs and 32768 EXACT bp16 products. The host\n")
W(" * program gates EVERY output of EVERY seed bit-exact vs this golden, plus a\n")
W(" * full 256-bit quire byte-gate on one output per seed.\n")
W(" *\n")
W(" * DO NOT hand-edit the numbers — GENERATED from the canonical oracle and\n")
W(" * re-verified by the host program at startup against the reused\n")
W(" * bp16_quire.h / bp16_encode.h headers (belt-and-suspenders).\n")
W(" *\n")
W(" * SOURCE OF TRUTH (regenerate, never by hand):\n")
W(" *   oracle: mosyne-bposit/kernels/bposit16_reference.py\n")
W(" *           (encode_bposit16, decode_bposit16, decoded_to_fraction,\n")
W(" *            bposit16_mul, quire256_to_bposit16, QUIRE_FRAC_BITS=96)\n")
W(" *   generator: gen_matmul_validate_golden.py (kept with this commit)\n")
W(" *   cross-checked vs golden/bp16_mul.json (operand pairs that overlap)\n")
W(" *\n")
W(" * Random data, bp16-quantized; reject-and-reseed guarantees every nonzero\n")
W(" * product has E2_a+E2_b >= -96 so the quire term is exactly representable and\n")
W(" * the C kernel == oracle bit-for-bit (see generator docstring).\n")
W(" * Layout per seed: A is M*K row-major, B is K*N row-major, C is M*N row-major.\n")
W(" *\n")
W(" * Verified locally (python, oracle) for ALL %d seeds:\n" % SEEDS)
W(" *   - each C[i,j] readout == oracle quire256_to_bposit16 of its exact dot quire\n")
W(" *   - per-output quire order-independent (forward == reverse == even/odd)\n")
W(" *   - every per-element exact product == bposit16_mul(A[i,k], B[k,j])\n")
W(" *   - %d operand pairs == golden/bp16_mul.json prod_code\n" % overlap_total)
W(" */\n")
W("#ifndef QUIRE_MATMUL_VALIDATE_GOLDEN_CASES_H\n")
W("#define QUIRE_MATMUL_VALIDATE_GOLDEN_CASES_H\n\n")
W("#include <cstdint>\n\n")
W(f"#define QGOLD_MMV_M {M}\n")
W(f"#define QGOLD_MMV_K {K}\n")
W(f"#define QGOLD_MMV_N {N}\n")
W(f"#define QGOLD_MMV_SEEDS {SEEDS}\n")
W(f"#define QGOLD_MMV_GATE_I {GATE_I}\n")
W(f"#define QGOLD_MMV_GATE_J {GATE_J}\n\n")


def emit_3d(name, per_seed, rows, cols, rowlabel):
    """Emit a [SEEDS][rows*cols] flattened int32 array."""
    W(f"static const int32_t {name}[{SEEDS}][{rows} * {cols}] = {{\n")
    for s in range(SEEDS):
        W(f"  {{ /* seed {s} */\n")
        mat = per_seed[s]
        for r in range(rows):
            W("    " + " ".join("0x%04x," % (mat[r][c] & 0xFFFF) for c in range(cols)))
            W(f"  /* {rowlabel} {r} */\n")
        W("  },\n")
    W("};\n\n")


W("/* A operand matrices, [seed][M*K] row-major (16-bit codes). */\n")
emit_3d("QGOLD_MMV_A_CODES", all_A, M, K, "row")
W("/* B operand matrices, [seed][K*N] row-major (16-bit codes). */\n")
emit_3d("QGOLD_MMV_B_CODES", all_B, K, N, "row")
W("/* expected C readouts, [seed][M*N] row-major: truncating bp16 readout of the\n")
W(" * exact quire dot Sum_k A[i,k]*B[k,j]. The host gates ALL M*N for ALL seeds. */\n")
emit_3d("QGOLD_MMV_C_READOUT", all_readout, M, N, "row")

# per-(i,k,j) standalone bposit16_mul codes, flattened [seed][i*K*N + k*N + j].
W("/* per-(i,k,j) standalone bposit16_mul codes, [seed][i*K*N + k*N + j]. The host\n")
W(" * self-check re-derives each exact product, reads it back ALONE, and asserts it\n")
W(" * equals this — the load-bearing per-element invariant, across all seeds. */\n")
W(f"static const int32_t QGOLD_MMV_PROD_CODES[{SEEDS}][{M} * {K} * {N}] = {{\n")
for s in range(SEEDS):
    W(f"  {{ /* seed {s} */\n")
    pc = all_prod[s]
    for i in range(M):
        for k in range(K):
            W("    " + " ".join("0x%04x," % (pc[i][k][j] & 0xFFFF) for j in range(N)))
            W(f"  /* i={i:2d} k={k:2d} */\n")
    W("  },\n")
W("};\n\n")

# one FULL expected quire per seed (the gated output (M-1,N-1)), 8x u32 LE.
W("/* one FULL expected quire per seed for a byte-level gate: output (%d,%d),\n" % (GATE_I, GATE_J))
W(" * 8x uint32 little-endian two's-complement, [seed][8]. */\n")
W(f"static const uint32_t QGOLD_MMV_GATE_QUIRE[{SEEDS}][8] = {{\n")
for s in range(SEEDS):
    limbs = all_gate_limbs[s]
    W("    { " + " ".join("0x%08xu," % l for l in limbs) + f" }},  /* seed {s} */\n")
W("};\n\n")

# matching little-endian hex strings for the human-readable gate print.
W(f"static const char QGOLD_MMV_GATE_QUIRE_HEX[{SEEDS}][65] = {{\n")
for s in range(SEEDS):
    W(f'    "{le_hex(all_gate_limbs[s])}",  /* seed {s} */\n')
W("};\n")
W("\n#endif /* QUIRE_MATMUL_VALIDATE_GOLDEN_CASES_H */\n")

# ---- human-readable summary to stderr (does not pollute the header) ----------
total_outputs = SEEDS * M * N
total_products = SEEDS * M * N * K
sys.stderr.write(
    "// FULL-VALIDATION golden: %d seeds x C[%d,%d]=A[%d,%d].B[%d,%d]\n"
    % (SEEDS, M, N, M, K, K, N))
sys.stderr.write("// total fully-validated outputs: %d\n" % total_outputs)
sys.stderr.write("// total EXACT bp16 products:     %d\n" % total_products)
sys.stderr.write("// bp16_mul.json overlap checked: %d\n" % overlap_total)
sys.stderr.write("// reseed attempts per seed:      %s\n" % reseeds)
for s in range(SEEDS):
    tag = "WIDE" if s in WIDE_SEEDS else "golden-zone"
    sys.stderr.write("// seed %d (%s): gate(%d,%d) readout 0x%04x quire %s\n" % (
        s, tag, GATE_I, GATE_J, all_readout[s][GATE_I][GATE_J],
        le_hex(all_gate_limbs[s])))
