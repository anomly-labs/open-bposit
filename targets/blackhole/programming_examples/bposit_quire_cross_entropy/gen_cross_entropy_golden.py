#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate the cross-entropy golden header from the canonical oracle,
cross-checked against the published golden/cross_entropy.json.

cross-entropy (the standard classification training loss):

    H(p, q) = -Sum_i  p_i * log2(q_i)

For a target distribution p and a model/predicted distribution q over the same
alphabet, cross-entropy is the per-sample loss minimized by every softmax
classifier. It is a sum-of-products: each term p_i * log2(q_i) is formed once and
accumulated in the EXACT 256-bit Kulisch quire (no per-product rounding), then the
running sum is negated. The crown-jewel property is that the accumulation is
bit-exact and order-independent -- the tiny-probability tails (where q_i is small
and log2(q_i) is large-negative) are accumulated with FULL fixed-point precision
in the quire, NOT dropped as float would drop them when the partial sum dominates.

Per element the numerics are:
    qlog = bposit16_log2(q_i)              (HOST-precomputed: per-element log2)
    term = bposit16_mul(p_i, qlog)         (EXACT bp16 product = p_i*log2(q_i)) [DEVICE]
    H_q += bposit16_to_quire(term)         (exact 256-bit accumulation, QFRAC=96)[DEVICE]
    H    = -H_q                            (negate the running quire)            [DEVICE]
    H_bp32 = quire256_to_bposit32(H_q)     (the SINGLE rounding, the readout)    [DEVICE]

DEVICE/HOST SPLIT (honest): the per-element log2(q_i) is precomputed on the HOST
(bposit16_log2) and passed to the device as a small bp16 vector. The full
65536-entry log2 LUT does NOT fit the BRISC baby core's tiny local DATA region
(~0x11d0 bytes) and big-L1 is a separate CB/buffer address space, not kernel
.data -- a large static LUT in the kernel TU overflows the .elf segment at load.
So the LOAD-BEARING part -- the EXACT 256-bit quire accumulation of the products
(where tiny-probability tails are summed losslessly, the rounding-stable result)
-- runs 100% ON-DEVICE, while the elementwise log2 is host-precomputed. The device
kernel is then pure exact-quire: term = bp16_mul(p_i, log2q_i) -> quire-accumulate
-> negate -> bposit32 readout, bit-identical on x86, qemu-rv32, and a Tensix BRISC
baby core. The host log2 IS the canonical oracle (bposit16_log2 == the LUT), so the
result is bit-exact vs golden/cross_entropy.json regardless of the split.

SOURCE OF TRUTH (regenerate, never by hand):
  golden : ../../golden/cross_entropy.json
           (p_codes, q_codes, expected quire_le_hex + bp32_code, n)
  oracle : mosyne-bposit/kernels/bposit16_reference.py
           (bposit16_log2, bposit16_mul, bposit16_to_quire, quire256_to_bposit32,
            decode_bposit32, decoded_to_fraction_32)
           (the proven RV32IM-clean reference; the device kernel mirrors its
            exact-quire accumulation, with log2 host-precomputed instead of LUT)
  log2   : mosyne-bposit/kernels/bposit16_reference.py::bposit16_log2 (host-precomputed per
            element; == kernel/bp16_log2_lut.h BP16_LOG2_LUT[q] by construction)
           (build_cross_entropy_golden / _cross_entropy_quire -- canonical producer
            of cross_entropy.json)

Emits the header `cross_entropy_golden_cases.h`. Run:

    python3 gen_cross_entropy_golden.py > cross_entropy_golden_cases.h
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get("BPOSIT16_REFERENCE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../mosyne-bposit/kernels")))  # bposit16_reference.py from github.com/anomly-labs/mosyne-bposit
from bposit16_reference import (  # noqa: E402
    bposit16_log2,
    bposit16_mul,
    bposit16_to_quire,
    quire256_to_bposit32,
    decode_bposit32,
    decoded_to_fraction_32,
)

GOLDEN = "../../golden/cross_entropy.json"
MASK256 = (1 << 256) - 1


def quire_to_le_u32x8(q_signed):
    q = q_signed & MASK256
    return [(q >> (32 * i)) & 0xFFFFFFFF for i in range(8)]


def le_hex(limbs):
    s = ""
    for limb in limbs:
        for b in range(4):
            s += "%02x" % ((limb >> (8 * b)) & 0xFF)
    return s


# ---- load the published golden (single source of operands + expected output) -
gj = json.load(open(GOLDEN))
N = gj["n"]
P = [c & 0xFFFF for c in gj["p_codes"]]              # target distribution codes
Q = [c & 0xFFFF for c in gj["q_codes"]]              # predicted distribution codes
GOLD_QUIRE_HEX = gj["quire_le_hex"]                  # 32-byte LE negated quire
GOLD_BP32 = gj["bp32_code"] & 0xFFFFFFFF             # final bposit32 readout

assert len(P) == N, f"p_codes has {len(P)} != n={N}"
assert len(Q) == N, f"q_codes has {len(Q)} != n*n={N}"

# ---- recompute H(p,q) from the oracle exact-quire path -----------------------
# log2(q_i) is HOST-precomputed (the part the device kernel no longer does, since
# the 65536-entry LUT does not fit the baby core's local DATA region); the EXACT
# quire accumulation of term_i = bposit16_mul(p_i, log2q_i) is what runs on-device.
log2q = [0] * N
terms = [0] * N
H_quire = 0
for i in range(N):
    log2q[i] = bposit16_log2(Q[i]) & 0xFFFF          # log2(q_i)  [HOST precompute]
    term = bposit16_mul(P[i], log2q[i])              # p_i * log2(q_i)  (EXACT) [DEVICE]
    terms[i] = term & 0xFFFF
    H_quire += bposit16_to_quire(term)               # exact 256-bit accumulation [DEVICE]
H_neg = -H_quire                                     # H(p,q) = -Sum p log2 q

# ---- cross-check (A): negated quire LE hex == published golden quire_le_hex ---
neg_limbs = quire_to_le_u32x8(H_neg)
neg_hex = le_hex(neg_limbs)
assert neg_hex == GOLD_QUIRE_HEX, (
    f"oracle negated quire {neg_hex} != golden {GOLD_QUIRE_HEX}")

# ---- cross-check (B): bposit32 readout of the negated quire == golden bp32 ----
# The readout is the quire->bposit32 encode of the NEGATED accumulator (positive
# cross-entropy). H = 2.625 -> 0x45400000.
H_bp32 = quire256_to_bposit32(H_neg) & 0xFFFFFFFF
assert H_bp32 == GOLD_BP32, (
    f"oracle bp32 0x{H_bp32:08x} != golden 0x{GOLD_BP32:08x}")

# ---- decode the readout to a human float for the report ----------------------
gd = decode_bposit32(H_bp32)
H_val = 0.0 if gd.is_special else float(decoded_to_fraction_32(gd))

# ---- full negated quire baked for a byte-level gate --------------------------
# The 32-byte LE negated quire IS the kernel's primary target (golden quire_le_hex).
gate_limbs = neg_limbs
gate_hex = neg_hex


# ---- emit the header ---------------------------------------------------------
Wt = sys.stdout.write


def fmt_codes_1d(name, vec, n, hexw=4, ctype="uint16_t", suffix=""):
    Wt(f"static const {ctype} {name}[{n}] = {{\n    ")
    Wt(" ".join(("0x%0*x%s," % (hexw, v & ((1 << (4 * hexw)) - 1), suffix)) for v in vec))
    Wt("\n};\n")


Wt("/* cross_entropy_golden_cases.h - baked golden for the Layer-1\n")
Wt(" * cross-entropy H(p,q) = -Sum_i p_i*log2(q_i) over bposit16 distributions,\n")
Wt(" * the standard classification training loss. log2(q_i) is HOST-precomputed\n")
Wt(" * (bposit16_log2; QGOLD_CE_LOG2Q below) because the 65536-entry log2 LUT does\n")
Wt(" * not fit the BRISC baby core's local DATA region. Each term p_i*log2(q_i) is\n")
Wt(" * then an EXACT bp16 product accumulated ON-DEVICE in the EXACT 256-bit\n")
Wt(" * b-posit16 quire (QUIRE_FRAC_BITS=96) and negated; the single rounding is the\n")
Wt(" * final quire->bposit32 readout (H=2.625 -> 0x45400000).\n")
Wt(" *\n")
Wt(" * DO NOT hand-edit the numbers - GENERATED from golden/cross_entropy.json and\n")
Wt(" * the canonical oracle, re-verified by the host program at startup against the\n")
Wt(" * reused bp16_mul.h / bp16_quire.h / bp32_encode.h headers + bposit16_log2\n")
Wt(" * (belt-and-suspenders).\n")
Wt(" *\n")
Wt(" * SOURCE OF TRUTH (regenerate, never by hand):\n")
Wt(" *   golden: golden/cross_entropy.json (p_codes, q_codes, quire_le_hex, bp32_code, n)\n")
Wt(" *   oracle: mosyne-bposit/kernels/bposit16_reference.py\n")
Wt(" *           (bposit16_log2, bposit16_mul, bposit16_to_quire,\n")
Wt(" *            quire256_to_bposit32, decode_bposit32, decoded_to_fraction_32)\n")
Wt(" *   kernel: kernel/cross_entropy_kernel.c (the proven RV32IM exact-quire reference)\n")
Wt(" *   log2  : bposit16_log2 host-precomputed == bp16_log2_lut.h BP16_LOG2_LUT[q]\n")
Wt(" *   gen   : gen_cross_entropy_golden.py (kept with this commit)\n")
Wt(" *\n")
Wt(" * Layout: target p[N], predicted q[N], and host-precomputed log2q[N] (bposit16).\n")
Wt(" * The kernel consumes p[] and log2q[] (pure exact-quire on-device), emits the\n")
Wt(" * 32-byte LE negated quire (the primary gate) then the bposit32 readout; the\n")
Wt(" * host gates BOTH byte-/code-identical vs golden.\n")
Wt(" *\n")
Wt(" * Verified locally (gcc/python, reusing the EXACT oracle the kernel C headers\n")
Wt(" * mirror - bposit16_log2 + bposit16_mul + bposit16_to_quire + quire256_to_bposit32):\n")
Wt(" *   - the negated 256-bit quire == golden/cross_entropy.json quire_le_hex\n")
Wt(" *   - the bposit32 readout == golden/cross_entropy.json bp32_code (H=2.625)\n")
Wt(" */\n")
Wt("#ifndef CROSS_ENTROPY_GOLDEN_CASES_H\n")
Wt("#define CROSS_ENTROPY_GOLDEN_CASES_H\n\n")
Wt("#include <cstdint>\n\n")
Wt(f"#define QGOLD_CE_N {N}\n\n")

Wt("/* target distribution p[i] and predicted distribution q[i] (bposit16). */\n")
fmt_codes_1d("QGOLD_CE_P", P, N)
Wt("\n")
fmt_codes_1d("QGOLD_CE_Q", Q, N)

Wt("\n/* HOST-precomputed per-element log2(q_i) = bposit16_log2(q_i) (bposit16). This\n")
Wt(" * is the small vector handed to the device IN PLACE of an on-chip log2 LUT; the\n")
Wt(" * device kernel is then pure exact-quire (term = bp16_mul(p_i, log2q_i)). By\n")
Wt(" * construction QGOLD_CE_LOG2Q[i] == BP16_LOG2_LUT[q_i]. */\n")
fmt_codes_1d("QGOLD_CE_LOG2Q", log2q, N)

Wt("\n/* expected per-element term code term_i = bposit16_mul(p_i, log2q_i) (bp16),\n")
Wt(" * baked for an optional per-element host self-check (not gated on device). */\n")
fmt_codes_1d("QGOLD_CE_TERM", terms, N)

Wt("\n/* expected cross-entropy as the final bposit32 readout of the negated quire.\n")
Wt(f" * H(p,q) = {H_val:g} -> code below. From golden/cross_entropy.json bp32_code. */\n")
Wt(f"#define QGOLD_CE_BP32 0x{H_bp32:08x}u\n")

Wt("\n/* the FULL expected 32-byte little-endian NEGATED quire (the kernel's primary\n")
Wt(" * target): H(p,q) accumulated exactly then negated. From golden quire_le_hex. */\n")
Wt("static const uint32_t QGOLD_CE_QUIRE[8] = {\n")
Wt("    " + " ".join("0x%08xu," % l for l in gate_limbs[:4]) + "\n")
Wt("    " + " ".join("0x%08xu," % l for l in gate_limbs[4:]) + "\n")
Wt("};\n")
Wt(f'static const char QGOLD_CE_QUIRE_HEX[] =\n    "{gate_hex}";\n')
Wt("\n#endif /* CROSS_ENTROPY_GOLDEN_CASES_H */\n")

# ---- human-readable summary to stderr (does not pollute the header) ----------
sys.stderr.write("// cross-entropy: N=%d, H(p,q)=%g (bp32 0x%08x)\n" % (N, H_val, H_bp32))
sys.stderr.write("// negated quire LE: %s\n" % neg_hex)
for i in range(N):
    sys.stderr.write("//  p[%d]=0x%04x q[%d]=0x%04x term=0x%04x\n" % (
        i, P[i], i, Q[i], terms[i]))
