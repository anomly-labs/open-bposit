#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate the gradient-accumulation golden from the oracle.

THE -for-TRAINING result: in large-batch / many-step training, a parameter's
gradient is a SUM of many tiny per-example / per-step contributions, g = Sum_t d_t.
In low precision, once the running sum grows to magnitude M, a new tiny d_t can fall
below ULP(M) and be LOST entirely (the "swamping" / stale-gradient / gradient-
underflow problem). The EXACT 256-bit Kulisch b-posit16 quire keeps every
contribution exactly, so it recovers the tiny-tail gradient mass that IEEE fp32
drops. This is the TRAINING analogue of the inference cancellation demos already on
silicon (bposit_quire_vs_float).

Both accumulations run on a real Blackhole baby (RISC-V) core; this script bakes the
ground truth so the host can gate both against it.

Ground truth comes ONLY from exact arithmetic (no float):
  * the exact gradient g = sum(Fraction(value_i)) over the EXACT rationals the bp16
    codes decode to (decoded_to_fraction) -- the oracle's exact value;
  * the exact 256-bit quire = sum_i bp16_to_q256(code_i) (QUIRE_FRAC_BITS=96),
    matching the on-device kernel (kernel/bp16_quire.h); its truncating bp16 readout
    = quire256_to_bposit16 (kernel/bp16_encode.h). Every test vector is built from
    bp16-exact contributions whose exact sum is itself bp16-representable, so the
    quire readout EQUALS the exact gradient (quire error == 0).

The fp32 baseline is computed with Python's `struct` round-trip (genuine IEEE-754
binary32, round-to-nearest-ties-to-even) -- exactly what a real fp32 hardware
accumulator does. The on-device float path reproduces this same fp32 add in pure
integer arithmetic (kernels/ieee_softfp32.h, reused verbatim from
bposit_quire_vs_float, host-verified bit-exact vs native C `float`), so the baked
fp32 gradient is the genuine fp32 running sum, not a contrivance.

The construction (per config):
  d_0     = +B           (a large early activation/loss spike, B = 2^k, inflating
                          the running accumulator to magnitude M = B)
  d_1..T  = +tau each    (T tiny per-example gradients, tau = 2^te with te << k-23,
                          so each tau is below ULP(M) once the running sum is at M)
  d_T+1   = -B           (the spike is cancelled by an opposite-sign later batch,
                          restoring the running sum to ~0)
  d_T+2   = +g0          (a coarse surviving gradient component fp32 keeps)
  exact gradient g = g0 + T*tau.
fp32 freezes at g0 (every tau was swamped under M and lost); the quire keeps all of
g0 + T*tau exactly. The fraction fp32 loses is T*tau / (g0 + T*tau), which GROWS with
the step count T -- the more steps you accumulate, the more tiny-gradient mass low
precision throws away.

Usage:
  python3 gen_gradient_golden.py            # prints + writes quire_gradient_cases.h
  python3 gen_gradient_golden.py out.json   # also writes a machine-readable blob
"""
import json
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


def codeval(code: int) -> Fraction:
    d = decode_bposit16(code)
    if d.is_special:               # zero / NaR -> contributes 0 (== bp16_to_q256)
        return Fraction(0)
    return decoded_to_fraction(d)


def C(x: Fraction) -> int:
    """Encode an exact rational to bp16, then ASSERT it round-trips exactly so each
    contribution is genuinely bp16-representable (no encode rounding hides here)."""
    code = encode_bposit16(x) & 0xFFFF
    back = codeval(code)
    assert back == x, f"value {x} is not bp16-exact: encodes to 0x{code:04x} = {back}"
    return code


def exact_quire_int(codes) -> int:
    q = 0
    for c in codes:
        q += int(codeval(c) * (1 << QUIRE_FRAC_BITS))
    return q


def quire_le_u32x8(q_signed: int):
    q = q_signed & MASK256
    return [(q >> (32 * i)) & 0xFFFFFFFF for i in range(8)]


def le_hex(limbs):
    s = ""
    for limb in limbs:
        for b in range(4):
            s += "%02x" % ((limb >> (8 * b)) & 0xFF)
    return s


def f32(x: float) -> float:
    """Round a Python float to genuine IEEE-754 binary32 (RTNE)."""
    return struct.unpack("f", struct.pack("f", x))[0]


def fp32_bits(x: float) -> int:
    return struct.unpack("I", struct.pack("f", f32(x)))[0]


def fp32_running_sum_bits(values):
    """Sequential fp32 running sum, returning the final fp32 BIT pattern. This is
    exactly what the on-device soft-fp32 kernel computes (host-verified bit-exact)."""
    acc = 0.0
    for v in values:
        acc = f32(f32(acc) + f32(float(v)))
    return struct.unpack("I", struct.pack("f", acc))[0]


# ---------------------------------------------------------------------------
# The training-realistic gradient-accumulation vectors.
#
#   M = 2^k is the running-sum magnitude a large early contribution inflates the
#       accumulator to; ULP(M) for fp32 (24-bit significand) is 2^(k-23).
#   tau = 2^te is each tiny per-step gradient; te is chosen << k-23 so every tau is
#       strictly below ULP(M) -> fp32 swallows it once the running sum is at M.
#   g0 is a coarse surviving gradient fp32 keeps. exact g = g0 + T*tau.
#
# We sweep T to show the lost tail mass (and lost fraction) GROW with the step count.
# k=14 -> M=16384, ULP(M)=2^-9~=0.00195; tau=2^-14~=6.1e-5 << ULP(M); g0=2^-7=1/128.
# ---------------------------------------------------------------------------
K_EXP = 14            # M = 2^14 = 16384 (running-sum magnitude)
TAU_EXP = -14         # tau = 2^-14 (each tiny per-step gradient), << ULP(M)=2^-9
G0 = Fraction(1, 128)  # 2^-7 coarse surviving gradient (bp16-exact)
T_SWEEP = [64, 128, 192, 256]


def build_cases():
    cases = []
    B = Fraction(1 << K_EXP)
    tau = Fraction(1, 1 << (-TAU_EXP)) if TAU_EXP < 0 else Fraction(1 << TAU_EXP)
    for T in T_SWEEP:
        # d_0=+B (spike), T x +tau (tiny per-step grads), -B (spike removed), +g0.
        vals = [B] + [tau] * T + [-B, G0]
        name = f"grad_accum_T{T}"
        desc = (f"+2^{K_EXP} spike, {T}x +2^{TAU_EXP} tiny per-step grads, -2^{K_EXP}, "
                f"+{G0} coarse grad  (running sum M=2^{K_EXP}; each tau<ULP(M); "
                f"exact g={float(G0 + T * tau)})")
        cases.append((name, vals, desc, T))
    return cases


def emit(cases, json_path=None):
    blob_cases = []
    lines = []
    lines.append("/* Copyright (c) 2026 Anomly, Inc.")
    lines.append(" * SPDX-License-Identifier: Apache-2.0 */")
    lines.append("/* quire_gradient_cases.h — GENERATED by gen_gradient_golden.py.")
    lines.append(" * DO NOT hand-edit. Training-realistic gradient-accumulation vectors")
    lines.append(" * g = Sum_t d_t where a large early contribution inflates the running")
    lines.append(" * accumulator to magnitude M, so each subsequent tiny per-step gradient")
    lines.append(" * falls below ULP(M) and is LOST by an IEEE fp32 running sum (the")
    lines.append(" * swamping / gradient-underflow problem). The EXACT 256-bit b-posit16")
    lines.append(" * quire keeps every contribution and recovers the tiny-tail mass.")
    lines.append(" *")
    lines.append(" * Ground truth from the oracle's exact Fraction arithmetic")
    lines.append(" * (bposit16_reference.py); fp32 baseline = genuine IEEE-754 binary32 RTNE")
    lines.append(" * running sum (struct round-trip).")
    lines.append(" *")
    lines.append(" * Per case the host gates: (1) device quire limbs + bp16 readout ==")
    lines.append(" * baked exact quire/readout (quire error 0 vs exact gradient); (2) device")
    lines.append(" * fp32 running-sum bits == baked fp32 bits, and != the exact gradient")
    lines.append(" * bits (fp32 genuinely loses tail mass). Self-checked against the reused C")
    lines.append(" * headers at startup before any device run.")
    lines.append(" */")
    lines.append("#ifndef QUIRE_GRADIENT_CASES_H")
    lines.append("#define QUIRE_GRADIENT_CASES_H")
    lines.append("")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append(f"#define QGRAD_NCASES {len(cases)}")
    lines.append("")

    case_meta = []  # (cap, name, n, desc)
    for ci, (name, vals, desc, T) in enumerate(cases):
        codes = [C(v) for v in vals]
        n = len(codes)
        q = exact_quire_int(codes)
        limbs = quire_le_u32x8(q)
        readout = quire256_to_bposit16(q) & 0xFFFF
        exact_sum = sum(vals, Fraction(0))
        readout_val = codeval(readout)
        assert readout_val == exact_sum, (
            f"case {name}: quire readout {float(readout_val)} != exact gradient "
            f"{float(exact_sum)}")
        # fp32 baseline running sum
        fvals = [float(v) for v in vals]
        fp32_final_bits = fp32_running_sum_bits(fvals)
        fp32_final = struct.unpack("f", struct.pack("I", fp32_final_bits))[0]
        exact_f = float(exact_sum)
        exact_fp32_bits = fp32_bits(exact_f)
        # fp32 MUST lose tail mass (drift below the exact gradient) -- else not a demo
        float_drifts = (fp32_final_bits != exact_fp32_bits)
        assert float_drifts, (
            f"case {name}: fp32 running sum did NOT lose the tail (={fp32_final}); "
            f"make tau smaller / M larger / T bigger")
        # lost fraction: (exact - fp32)/exact (the gradient mass fp32 dropped)
        lost_frac = (exact_f - fp32_final) / exact_f if exact_f != 0 else 0.0
        assert lost_frac > 0.0, f"case {name}: fp32 did not lose gradient mass"

        num = exact_sum.numerator
        den = exact_sum.denominator
        # encode lost fraction as an exact rational (exact - fp32_final) is dyadic too,
        # but we bake it as a double for the host print; the host re-derives it anyway.
        cap = name.upper()
        lines.append(f"/* ---- case {ci}: {name} ---- */")
        lines.append(f"/*   {desc} */")
        lines.append(f"/*   exact g = {num}/{den} = {exact_f!r} ; "
                     f"quire readout 0x{readout:04x} = {float(readout_val)!r} (EXACT) ; "
                     f"fp32 running sum = {fp32_final!r} (0x{fp32_final_bits:08x}) ; "
                     f"fp32 LOSES {lost_frac * 100.0:.1f}% of g */")
        lines.append(f"static const int32_t QGRAD_{cap}_CODES[] = {{")
        for r in range(0, n, 8):
            row = ", ".join("0x%04x" % c for c in codes[r:r + 8])
            lines.append(f"    {row},")
        lines.append("};")
        lines.append(f"static const int QGRAD_{cap}_N = "
                     f"(int)(sizeof(QGRAD_{cap}_CODES)/sizeof(QGRAD_{cap}_CODES[0]));")
        lines.append(f"static const int QGRAD_{cap}_T = {T};  /* number of tiny per-step grads */")
        lines.append(f"static const uint32_t QGRAD_{cap}_QUIRE[8] = {{")
        lines.append("    " + ", ".join("0x%08xu" % l for l in limbs[:4]) + ",")
        lines.append("    " + ", ".join("0x%08xu" % l for l in limbs[4:]) + ",")
        lines.append("};")
        lines.append(f"static const char QGRAD_{cap}_QUIRE_HEX[] = \"{le_hex(limbs)}\";")
        lines.append(f"static const int32_t QGRAD_{cap}_READOUT_BP16 = 0x{readout:04x};")
        lines.append(f"static const uint32_t QGRAD_{cap}_FP32_BITS = 0x{fp32_final_bits:08x}u;  "
                     f"/* genuine fp32 running sum (loses tail) */")
        lines.append(f"static const uint32_t QGRAD_{cap}_EXACT_FP32_BITS = 0x{exact_fp32_bits:08x}u;  "
                     f"/* fp32(exact gradient) -- the target fp32 missed */")
        lines.append(f"static const int64_t QGRAD_{cap}_EXACT_NUM = {num}LL;")
        lines.append(f"static const int64_t QGRAD_{cap}_EXACT_DEN = {den}LL;")
        lines.append(f"static const char QGRAD_{cap}_NAME[] = \"{name}\";")
        lines.append("")
        case_meta.append((cap, name, n, desc))

        blob_cases.append({
            "name": name, "desc": desc, "T": T,
            "codes": codes, "n": n,
            "quire_limbs_le_u32": limbs, "quire_le_hex": le_hex(limbs),
            "readout_bp16": readout,
            "exact_num": num, "exact_den": den, "exact_float": exact_f,
            "fp32_running_bits": fp32_final_bits, "fp32_running_float": fp32_final,
            "exact_fp32_bits": exact_fp32_bits, "lost_frac": lost_frac,
        })

    lines.append("/* ---- dispatch table: pointers into the per-case arrays ---- */")
    lines.append("typedef struct {")
    lines.append("    const char* name;")
    lines.append("    const int32_t* codes;")
    lines.append("    int n;")
    lines.append("    int t_steps;                  /* number of tiny per-step grads */")
    lines.append("    const uint32_t* quire;        /* 8 limbs LE */")
    lines.append("    const char* quire_hex;")
    lines.append("    int32_t readout_bp16;")
    lines.append("    uint32_t fp32_bits;           /* genuine fp32 running sum */")
    lines.append("    uint32_t exact_fp32_bits;     /* fp32(exact gradient) */")
    lines.append("    int64_t exact_num;")
    lines.append("    int64_t exact_den;")
    lines.append("} qgrad_case_t;")
    lines.append("")
    lines.append("static const qgrad_case_t QGRAD_CASES[] = {")
    for cap, name, n, desc in case_meta:
        lines.append(
            f"    {{ QGRAD_{cap}_NAME, QGRAD_{cap}_CODES, QGRAD_{cap}_N, QGRAD_{cap}_T, "
            f"QGRAD_{cap}_QUIRE, QGRAD_{cap}_QUIRE_HEX, QGRAD_{cap}_READOUT_BP16, "
            f"QGRAD_{cap}_FP32_BITS, QGRAD_{cap}_EXACT_FP32_BITS, "
            f"QGRAD_{cap}_EXACT_NUM, QGRAD_{cap}_EXACT_DEN }},")
    lines.append("};")
    lines.append("")
    lines.append("#endif  /* QUIRE_GRADIENT_CASES_H */")

    text = "\n".join(lines) + "\n"
    here = os.path.dirname(os.path.abspath(__file__))
    with open(here + "/quire_gradient_cases.h", "w") as f:
        f.write(text)

    for bc in blob_cases:
        print(f"// case {bc['name']} (T={bc['T']}): exact_g={bc['exact_float']!r} "
              f"quire_readout=0x{bc['readout_bp16']:04x} "
              f"fp32_g={bc['fp32_running_float']!r} "
              f"fp32_loses={bc['lost_frac'] * 100.0:.1f}%")
    print(f"// wrote quire_gradient_cases.h with {len(cases)} cases")

    if json_path:
        json.dump({"schema": "bp16_quire_gradient/1",
                   "note": "training-realistic gradient-accumulation vectors: exact "
                           "256-bit b-posit16 quire vs IEEE fp32 running sum. Ground "
                           "truth from oracle exact Fraction arithmetic; fp32 = genuine "
                           "binary32 RTNE. fp32 loses tiny-tail gradient mass swamped "
                           "under the running-sum magnitude; quire recovers it exactly.",
                   "cases": blob_cases},
                  open(json_path, "w"), indent=2)
        print(f"// wrote machine-readable blob {json_path}")


if __name__ == "__main__":
    emit(build_cases(), json_path=(sys.argv[1] if len(sys.argv) > 1 else None))
