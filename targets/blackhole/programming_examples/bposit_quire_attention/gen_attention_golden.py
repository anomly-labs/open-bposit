#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate the attention-under-massive-activation-outliers golden.

THE RESULT: transformer attention scores s_j = q.k_j are corrupted when a few
activation dimensions are MASSIVE outliers (Sun et al. 2024, arXiv:2402.17762 —
"massive activations" drive attention sinks). A naive IEEE fp32 dot-product lets
the huge outlier PRODUCTS swamp / catastrophically cancel away the many small
but meaningful contributions, distorting the softmax and flipping which key wins.
The EXACT 256-bit Kulisch b-posit16 quire keeps the full dot product EXACTLY
(every bp16xbp16 product placed un-rounded into 96 fraction bits, accumulated
exactly, single readout), so the quire softmax matches the exact softmax and
fp32 does not. (Framing: src/spacetime/attention_precision.py.)

This script bakes the ground truth so the host (bposit_quire_attention.cpp) can
gate the device's exact-quire and fp32 attention scores against it.

Ground truth comes ONLY from exact arithmetic (no float for the quire/exact path):
  * exact score s_j = sum_i Fraction(q_i)*Fraction(k_ji) over the EXACT rationals
    the bp16 codes decode to (decoded_to_fraction) — the oracle's exact value;
  * exact 256-bit quire dot = sum_i int(q_i*k_ji * 2^96) (QUIRE_FRAC_BITS=96),
    matching the on-device kernel (kernel/bp16_quire.h:bp16_prod_to_q256); its
    truncating bp16 readout = quire256_to_bposit16. Because every q,k value is
    bp16-exact and each exact score is itself bp16-representable here, the quire
    readout EQUALS the exact score (quire error == 0).

The fp32 baseline is the genuine IEEE-754 binary32 NAIVE dot: each product
q_i*k_ji is formed exactly (bp16*bp16 fits the fp32 24-bit significand) then
added into a sequential fp32 running sum, in DIM ORDER (outlier dims first) —
exactly what a real fp32 attention kernel does, reproduced on-device in pure
integer arithmetic (kernels/ieee_softfp32.h, host-verified bit-exact vs C float).

THE FLIP (verified below, asserted at gen time): the outlier dims (q*k = +/-2^33)
EXACTLY CANCEL within each key's dot, so the exact score is the small-dim signal;
but in the fp32 running sum the +2^33 lands first, the O(1) signal terms fall
below its ULP (~2^10) and are swallowed, then -2^33 returns the sum to ~0 having
recorded NONE of the signal. fp32 then ranks keys only by a tiny post-cancellation
tail -> picks the WRONG key. exact/quire pick the true signal winner.

Usage:
  python3 gen_attention_golden.py            # prints + writes attention_golden_cases.h
  python3 gen_attention_golden.py out.json   # also writes a machine-readable blob
"""
import json
import math
import struct
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.environ.get("BPOSIT16_REFERENCE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../mosyne-bposit/kernels")))  # bposit16_reference.py from github.com/anomly-labs/mosyne-bposit
from bposit16_reference import (  # noqa: E402
    encode_bposit16,
    decode_bposit16,
    decoded_to_fraction,
    quire256_to_bposit16,
    QUIRE_FRAC_BITS,
)

MASK256 = (1 << 256) - 1
D = 32  # head dimension


# ---- bp16 exact helpers (oracle) -------------------------------------------
def codeval(code: int) -> Fraction:
    d = decode_bposit16(code)
    if d.is_special:               # zero / NaR -> contributes 0
        return Fraction(0)
    return decoded_to_fraction(d)


def C(x) -> int:
    """Encode an exact rational to bp16 and ASSERT it round-trips exactly, so the
    chosen value is genuinely bp16-representable (no encode rounding hides here)."""
    x = Fraction(x)
    code = encode_bposit16(x) & 0xFFFF
    back = codeval(code)
    assert back == x, f"value {x} is not bp16-exact: encodes to 0x{code:04x} = {back}"
    return code


# ---- exact 256-bit quire DOT (matches kernel bp16_prod_to_q256 + readout) --
def exact_quire_dot_int(qc, kc) -> int:
    """sum_i int(q_i*k_ji * 2^96): exact bp16xbp16 product placed into the quire,
    no per-product rounding (== kernel bp16_prod_to_q256 / q256_add)."""
    q = 0
    for a, b in zip(qc, kc):
        q += int(codeval(a) * codeval(b) * (1 << QUIRE_FRAC_BITS))
    return q


def exact_score(qc, kc) -> Fraction:
    return sum((codeval(a) * codeval(b) for a, b in zip(qc, kc)), Fraction(0))


def quire_le_u32x8(q_signed: int):
    q = q_signed & MASK256
    return [(q >> (32 * i)) & 0xFFFFFFFF for i in range(8)]


def le_hex(limbs):
    s = ""
    for limb in limbs:
        for b in range(4):
            s += "%02x" % ((limb >> (8 * b)) & 0xFF)
    return s


# ---- genuine IEEE-754 fp32 NAIVE dot (matches kernel ieee_softfp32 path) ----
def f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def fp32_bits(x: float) -> int:
    return struct.unpack("I", struct.pack("f", f32(x)))[0]


def fp32_naive_dot_bits(qc, kc) -> int:
    """Sequential fp32 naive dot in DIM ORDER. Each product is exact (bp16*bp16
    fits fp32's 24-bit significand), the running ADD is genuine binary32 RTNE.
    Returns the final fp32 BIT pattern — exactly what the on-device soft-fp32
    kernel computes (host-verified bit-exact)."""
    acc = 0.0
    for a, b in zip(qc, kc):
        prod = f32(float(codeval(a)) * float(codeval(b)))  # exact product, fits fp32
        acc = f32(acc + prod)                              # genuine fp32 add (RTNE)
    return struct.unpack("I", struct.pack("f", acc))[0]


def softmax(scores):
    m = max(scores)
    ex = [math.exp(s - m) for s in scores]
    z = sum(ex)
    return [e / z for e in ex]


# ---------------------------------------------------------------------------
# THE QUERY / KEYS — massive-activation outliers that cancel in the exact dot
# but destroy the fp32 dot. D=32. Per dim:
#   dim 0      : +OUTLIER product  (q0=2^17, k0=2^16  -> +2^33)
#   dims 1..29 : the SIGNAL (q_i=1/8 ; k_ji in {+1,-1}); exact dot = sum/8
#   dim 30     : -OUTLIER product  (q30=2^17, k30=-2^16 -> -2^33, exact cancel)
#   dim 31     : per-key TAIL (q31=1/8 ; k31=tail) — small post-cancellation
#                residue that fp32 (having lost the signal) ranks keys by.
# Exact score = +2^33 + signal - 2^33 + tail/8 = signal + tail/8, dominated by
# the signal -> true argmax = key0. fp32: +2^33 then signal swallowed (ULP ~2^10
# >> O(1)) then -2^33 -> ~0, leaving only the tail -> fp32 ranks by tail -> key2.
# All values are bp16-EXACT (asserted in C()).
# ---------------------------------------------------------------------------
def build_problem():
    QHI = Fraction(1 << 17)   # q outlier magnitude (dims 0, 30)
    KBIG = Fraction(1 << 16)  # k outlier magnitude
    QSIG = Fraction(1, 8)     # q weight on the signal/tail dims

    q_vals = [QHI] + [QSIG] * 29 + [QHI] + [QSIG]
    assert len(q_vals) == D

    def make_key(signal29, tail):
        k = [KBIG] + signal29 + [-KBIG] + [tail]
        assert len(k) == D
        return k

    # true signal (dims 1..29; each contributes QSIG*value = value/8):
    sig0 = [Fraction(1)] * 29                        # 29/8 = 3.625  (TRUE best)
    sig1 = [Fraction(1)] * 23 + [Fraction(-1)] * 6   # 17/8 = 2.125
    sig2 = [Fraction(1)] * 21 + [Fraction(-1)] * 8   # 13/8 = 1.625
    # post-cancellation tails (dim31; contribute tail/8): ordered the WRONG way
    # so fp32, which lost the signal, picks key2.
    tail0, tail1, tail2 = Fraction(1, 2), Fraction(2), Fraction(8)

    keys_vals = [make_key(sig0, tail0), make_key(sig1, tail1), make_key(sig2, tail2)]
    key_names = ["key0_strong_signal", "key1_mid_signal", "key2_weak_signal_big_tail"]
    return q_vals, keys_vals, key_names


def emit(json_path=None):
    q_vals, keys_vals, key_names = build_problem()
    J = len(keys_vals)

    qc = [C(v) for v in q_vals]
    kcs = [[C(v) for v in kv] for kv in keys_vals]

    # ---- per-key exact / quire / fp32 ------------------------------------
    exact_scores = [exact_score(qc, kc) for kc in kcs]
    quire_ints = [exact_quire_dot_int(qc, kc) for kc in kcs]
    quire_limbs = [quire_le_u32x8(q) for q in quire_ints]
    quire_readouts = [quire256_to_bposit16(q) & 0xFFFF for q in quire_ints]
    fp32_score_bits = [fp32_naive_dot_bits(qc, kc) for kc in kcs]

    # quire readout MUST equal the exact score (the whole claim)
    for j in range(J):
        rv = codeval(quire_readouts[j])
        assert rv == exact_scores[j], (
            f"key {j}: quire readout {float(rv)} != exact score {float(exact_scores[j])}")

    exact_f = [float(s) for s in exact_scores]
    fp32_f = [struct.unpack("f", struct.pack("I", b))[0] for b in fp32_score_bits]

    # ---- softmax over each score vector ----------------------------------
    sm_exact = softmax(exact_f)
    sm_quire = softmax([float(codeval(r)) for r in quire_readouts])  # == sm_exact
    sm_fp32 = softmax(fp32_f)
    am_exact = max(range(J), key=lambda j: exact_f[j])
    am_fp32 = max(range(J), key=lambda j: fp32_f[j])

    assert sm_quire == sm_exact, "quire softmax must equal exact softmax (bit-exact dot)"
    assert am_exact != am_fp32, (
        f"NO FLIP: exact and fp32 both pick key {am_exact}; make the outliers larger / "
        f"the cancellation deeper")

    # softmax divergence metrics (exact == quire is the reference distribution)
    kl_exact_fp32 = sum(p * math.log(p / q)
                        for p, q in zip(sm_exact, sm_fp32) if p > 0)
    max_w_err = max(abs(a - b) for a, b in zip(sm_exact, sm_fp32))

    # softmax in fixed-point milli-units for an exact integer compare on-device
    # (host re-derives in double; these are just for the printed golden).
    # ---- emit header -----------------------------------------------------
    lines = []
    lines.append("/* Copyright (c) 2026 Anomly, Inc.")
    lines.append(" * SPDX-License-Identifier: Apache-2.0 */")
    lines.append("/* attention_golden_cases.h — GENERATED by gen_attention_golden.py.")
    lines.append(" * DO NOT hand-edit. attention under MASSIVE-ACTIVATION OUTLIERS:")
    lines.append(" * one query q and J keys k_j (head_dim D=32). A few dims are huge")
    lines.append(" * outlier products (+/-2^33) that EXACTLY cancel in each key's exact")
    lines.append(" * dot, so the exact score is the small-dim signal; but a naive IEEE")
    lines.append(" * fp32 dot loses that signal in the cancellation and FLIPS the softmax")
    lines.append(" * argmax. Ground truth = oracle exact Fraction arithmetic")
    lines.append(" * (bposit16_reference.py); exact 256-bit quire dot = bp16_prod_to_q256")
    lines.append(" * accumulation + single readout (kernel/bp16_quire.h, bp16_encode.h);")
    lines.append(" * fp32 baseline = genuine binary32 RTNE naive dot (struct round-trip).")
    lines.append(" *")
    lines.append(" * Per key the host gates: (1) device quire limbs + bp16 readout ==")
    lines.append(" * baked exact quire/readout (quire score error 0 vs exact); (2) device")
    lines.append(" * fp32 naive-dot bits == baked fp32 bits. Then it softmaxes the score")
    lines.append(" * vectors and asserts exact/quire argmax != fp32 argmax (the FLIP).")
    lines.append(" */")
    lines.append("#ifndef ATTENTION_GOLDEN_CASES_H")
    lines.append("#define ATTENTION_GOLDEN_CASES_H")
    lines.append("")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append(f"#define ATTN_D {D}")
    lines.append(f"#define ATTN_NKEYS {J}")
    lines.append("")

    # query codes
    lines.append("/* ---- query q (D codes) ---- */")
    lines.append("static const int32_t ATTN_Q_CODES[ATTN_D] = {")
    for r in range(0, D, 8):
        lines.append("    " + ", ".join("0x%04x" % c for c in qc[r:r + 8]) + ",")
    lines.append("};")
    lines.append("")

    # per-key data
    for j in range(J):
        cap = "K%d" % j
        lines.append(f"/* ---- key {j}: {key_names[j]} ---- */")
        lines.append(f"/*   exact score = {exact_scores[j].numerator}/{exact_scores[j].denominator} "
                     f"= {exact_f[j]!r} ; quire readout 0x{quire_readouts[j]:04x} = "
                     f"{float(codeval(quire_readouts[j]))!r} (EXACT) ; "
                     f"fp32 naive dot = {fp32_f[j]!r} (0x{fp32_score_bits[j]:08x}) -- WRONG */")
        lines.append(f"static const int32_t ATTN_{cap}_CODES[ATTN_D] = {{")
        for r in range(0, D, 8):
            lines.append("    " + ", ".join("0x%04x" % c for c in kcs[j][r:r + 8]) + ",")
        lines.append("};")
        lines.append(f"static const uint32_t ATTN_{cap}_QUIRE[8] = {{")
        lines.append("    " + ", ".join("0x%08xu" % l for l in quire_limbs[j][:4]) + ",")
        lines.append("    " + ", ".join("0x%08xu" % l for l in quire_limbs[j][4:]) + ",")
        lines.append("};")
        lines.append(f"static const char ATTN_{cap}_QUIRE_HEX[] = \"{le_hex(quire_limbs[j])}\";")
        lines.append(f"static const int32_t ATTN_{cap}_READOUT_BP16 = 0x{quire_readouts[j]:04x};")
        lines.append(f"static const uint32_t ATTN_{cap}_FP32_BITS = 0x{fp32_score_bits[j]:08x}u;  "
                     f"/* genuine fp32 naive dot */")
        lines.append(f"static const int64_t ATTN_{cap}_EXACT_NUM = {exact_scores[j].numerator}LL;")
        lines.append(f"static const int64_t ATTN_{cap}_EXACT_DEN = {exact_scores[j].denominator}LL;")
        lines.append(f"static const char ATTN_{cap}_NAME[] = \"{key_names[j]}\";")
        lines.append("")

    # dispatch table
    lines.append("typedef struct {")
    lines.append("    const char* name;")
    lines.append("    const int32_t* codes;        /* D key codes */")
    lines.append("    const uint32_t* quire;       /* 8 limbs LE (exact quire dot q.k) */")
    lines.append("    const char* quire_hex;")
    lines.append("    int32_t readout_bp16;        /* quire dot -> bp16 (== exact score) */")
    lines.append("    uint32_t fp32_bits;          /* genuine fp32 naive dot */")
    lines.append("    int64_t exact_num;")
    lines.append("    int64_t exact_den;")
    lines.append("} attn_key_t;")
    lines.append("")
    lines.append("static const attn_key_t ATTN_KEYS[ATTN_NKEYS] = {")
    for j in range(J):
        cap = "K%d" % j
        lines.append(
            f"    {{ ATTN_{cap}_NAME, ATTN_{cap}_CODES, ATTN_{cap}_QUIRE, ATTN_{cap}_QUIRE_HEX, "
            f"ATTN_{cap}_READOUT_BP16, ATTN_{cap}_FP32_BITS, ATTN_{cap}_EXACT_NUM, "
            f"ATTN_{cap}_EXACT_DEN }},")
    lines.append("};")
    lines.append("")
    # the expected outcome (baked for the host's PASS line / cross-check)
    lines.append(f"static const int ATTN_EXACT_ARGMAX = {am_exact};  /* true winner (quire agrees) */")
    lines.append(f"static const int ATTN_FP32_ARGMAX  = {am_fp32};  /* fp32 picks the WRONG key */")
    lines.append("")
    lines.append("#endif  /* ATTENTION_GOLDEN_CASES_H */")

    text = "\n".join(lines) + "\n"
    here = os.path.dirname(os.path.abspath(__file__))
    with open(here + "/attention_golden_cases.h", "w") as f:
        f.write(text)

    # console summary
    print("// attention under massive-activation outliers — golden")
    print(f"//   D={D}  J={J}")
    for j in range(J):
        print(f"//   {key_names[j]:28s} exact={exact_f[j]:+.4f} "
              f"quire=0x{quire_readouts[j]:04x}({float(codeval(quire_readouts[j])):+.4f}) "
              f"fp32={fp32_f[j]:+.4f}")
    print(f"//   exact softmax = {[round(x,4) for x in sm_exact]}  argmax={am_exact}")
    print(f"//   quire softmax = {[round(x,4) for x in sm_quire]}  argmax={am_exact}  (== exact)")
    print(f"//   fp32  softmax = {[round(x,4) for x in sm_fp32]}  argmax={am_fp32}  (WRONG)")
    print(f"//   FLIP: exact/quire pick key{am_exact}, fp32 picks key{am_fp32}")
    print(f"//   KL(exact||fp32)={kl_exact_fp32:.4f}  max weight err={max_w_err:.4f}")
    print("// wrote attention_golden_cases.h")

    if json_path:
        json.dump({
            "schema": "bp16_quire_attention/1",
            "note": "attention under massive-activation outliers: exact 256-bit "
                    "b-posit16 quire dot vs IEEE fp32 naive dot; softmax argmax FLIP. "
                    "Ground truth from oracle exact Fraction arithmetic.",
            "D": D, "nkeys": J,
            "q_codes": qc,
            "keys": [{
                "name": key_names[j], "codes": kcs[j],
                "quire_limbs_le_u32": quire_limbs[j], "quire_le_hex": le_hex(quire_limbs[j]),
                "readout_bp16": quire_readouts[j],
                "exact_num": exact_scores[j].numerator, "exact_den": exact_scores[j].denominator,
                "exact_float": exact_f[j],
                "fp32_bits": fp32_score_bits[j], "fp32_float": fp32_f[j],
            } for j in range(J)],
            "softmax_exact": sm_exact, "softmax_fp32": sm_fp32,
            "argmax_exact": am_exact, "argmax_fp32": am_fp32,
            "kl_exact_fp32": kl_exact_fp32, "max_weight_err": max_w_err,
        }, open(json_path, "w"), indent=2)
        print(f"// wrote machine-readable blob {json_path}")


if __name__ == "__main__":
    emit(json_path=(sys.argv[1] if len(sys.argv) > 1 else None))
