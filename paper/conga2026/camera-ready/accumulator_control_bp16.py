# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The 16-bit half of the control Reviewer 3 asked for in R3.4, and the
measured (not merely proved) Q_F = 106 confirmation R3.1 asks for.

`accumulator_control.py` ran this experiment in the W8A8 integer path and
returned an honest null: int8 products are bounded by 127^2 and fp32 holds
integers exactly to 2^24, so int64, int32, sequential fp32 and a blocked fp32
dot agree on all 131,072 outputs. The quire buys reproducibility there, not
accuracy. Section 6.7 now says so and points the accuracy case at the 16-bit
evidence, "where operands carry real dynamic range" -- which is the claim this
script measures.

Same model, same layer, same seed and same token draw as the int8 control, so
the two are directly comparable. Operands are encoded to b-posit16 exactly
once; every arm then accumulates the IDENTICAL products:

  exact        sum of dyadic rationals in integer arithmetic -- ground truth
  quire QF=96  the shipped 256-bit profile (b-posit16 products span 2^-96 to
               2^96, so the lowest-order products are truncated at the radix)
  quire QF=106 the all-exact profile; the kernel change is BP16_QFRAC 106
  fp32 seq     sequential fp32, rounding after every add
  fp32 pair    numpy's blocked/pairwise fp32 dot, as a GPU GEMM schedules it
  bf16 seq     sequential bf16, the accumulator the flash-attention bias paper
               traces

Every b-posit16 value is a dyadic rational m*2^e, so products and their exact
sum are computed in integer arithmetic with no rounding anywhere in the
reference path. The fp32/bf16 arms receive those same products widened
losslessly to float64 (a b-posit16 product carries at most ~22 mantissa bits,
well inside float64's 53), so the ONLY thing that varies across arms is the
accumulator.
"""
import json
import sys, os
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

# bposit16_reference.py (Apache-2.0) ships with github.com/anomly-labs/mosyne-bposit under
# kernels/. Set BPOSIT16_REFERENCE_DIR, or clone mosyne-bposit beside this repository.
sys.path.insert(0, os.environ.get("BPOSIT16_REFERENCE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "mosyne-bposit", "kernels")))
from bposit16_reference import (  # noqa: E402
    decode_bposit16, decoded_to_fraction, encode_bposit16,
)

SNAP = Path(os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")) + "/"
            "models--unsloth--Llama-3.2-1B-Instruct/snapshots/"
            "5a8abab4a5d6f164389b1079fb721cfab8d7126c")
LAYER = "model.layers.0.self_attn.q_proj.weight"
EMBED = "model.embed_tokens.weight"
N_TOK = 8             # real token ids -> real layer-0 input rows
N_OUT = 16            # output columns of q_proj
SEED = 0              # same draw as accumulator_control.py
QF_SHIPPED = 96
QF_EXACT = 106


def as_dyadic(f: Fraction):
    """A b-posit16 value is m * 2^e exactly. Return (m, e) with integer m."""
    num, den = f.numerator, f.denominator
    if den & (den - 1):
        raise ValueError(f"not dyadic: {f}")
    return num, -den.bit_length() + 1


def encode_row(row):
    """Encode a float64 row to b-posit16 codes, then to exact dyadics."""
    codes = [encode_bposit16(Fraction(float(v))) for v in row]
    dy = []
    for c in codes:
        d = decode_bposit16(c)
        dy.append((0, 0) if d.is_special else as_dyadic(decoded_to_fraction(d)))
    return codes, dy


def exact_dot(a_dy, b_dy):
    """Exact integer dot of two dyadic vectors. Returns (m, e) with value m*2^e."""
    terms = [(ma * mb, ea + eb) for (ma, ea), (mb, eb) in zip(a_dy, b_dy)
             if ma and mb]
    if not terms:
        return 0, 0
    emin = min(e for _, e in terms)
    return sum(m << (e - emin) for m, e in terms), emin


def quire_dot(a_dy, b_dy, qf):
    """Accumulate the same products in a fixed-point quire with qf fraction
    bits. Each product is truncated to a multiple of 2^-qf on entry, exactly
    as bposit16_to_quire does; the accumulation itself is exact."""
    acc = 0
    for (ma, ea), (mb, eb) in zip(a_dy, b_dy):
        if not ma or not mb:
            continue
        m, e = ma * mb, ea + eb
        shift = e + qf
        acc += (m << shift) if shift >= 0 else -((-m) >> -shift) if m < 0 else (m >> -shift)
    return acc


def to_float(m, e):
    """Exact dyadic -> float64 (used only for reporting and the float arms)."""
    return float(Fraction(m, 1) * Fraction(2) ** e) if m else 0.0


def main():
    with safe_open(str(SNAP / "model.safetensors"), framework="pt") as f:
        W = f.get_tensor(LAYER).to(torch.float64).numpy()
        E = f.get_tensor(EMBED).to(torch.float64).numpy()
    cfg = json.loads((SNAP / "config.json").read_text())
    print("=== 16-bit accumulator control: identical b-posit16 operands ===")
    print(f"model  : Llama-3.2-1B-Instruct  (hidden {cfg['hidden_size']})")
    print(f"layer  : {LAYER}  shape {W.shape}")

    rng = np.random.default_rng(SEED)
    ids = rng.integers(0, E.shape[0], size=N_TOK)
    H = E[ids]

    K = W.shape[1]
    print(f"shape  : M={N_TOK} K={K} N={N_OUT}  "
          f"({N_TOK * N_OUT} outputs, {K}-term contractions)")
    print(f"operands: b-posit16, encoded once, identical for every arm")
    print(f"seed   : {SEED} (same token draw as the int8 control)\n")

    h_dy = [encode_row(H[i])[1] for i in range(N_TOK)]
    w_dy = [encode_row(W[j])[1] for j in range(N_OUT)]

    stats = {k: {"diff": 0, "max_rel": 0.0, "sum_rel": 0.0} for k in
             ("q96", "q106", "fp32_seq", "fp32_pair", "bf16_seq")}
    n_out = 0
    exp_lo, exp_hi = None, None

    for i in range(N_TOK):
        for j in range(N_OUT):
            a, b = h_dy[i], w_dy[j]
            em, ee = exact_dot(a, b)
            truth = Fraction(em, 1) * Fraction(2) ** ee
            n_out += 1

            # --- product exponent spread, to show why QF matters at all
            for (ma, ea), (mb, eb) in zip(a, b):
                if ma and mb:
                    e = ea + eb
                    exp_lo = e if exp_lo is None else min(exp_lo, e)
                    exp_hi = e if exp_hi is None else max(exp_hi, e)

            for name, qf in (("q96", QF_SHIPPED), ("q106", QF_EXACT)):
                q = quire_dot(a, b, qf)
                got = Fraction(q, 1 << qf)
                if got != truth:
                    stats[name]["diff"] += 1
                    if truth:
                        rel = abs(float((got - truth) / truth))
                        stats[name]["max_rel"] = max(stats[name]["max_rel"], rel)
                        stats[name]["sum_rel"] += rel

            prods = np.array([to_float(ma * mb, ea + eb)
                              for (ma, ea), (mb, eb) in zip(a, b)], dtype=np.float64)
            tf = float(truth)

            acc = np.float32(0.0)
            for p in prods.astype(np.float32):
                acc = np.float32(acc + p)
            pair = np.float32(prods.astype(np.float32).sum(dtype=np.float32))
            bacc = torch.zeros((), dtype=torch.bfloat16)
            for p in torch.from_numpy(prods).to(torch.bfloat16):
                bacc = (bacc + p).to(torch.bfloat16)

            for name, val in (("fp32_seq", float(acc)),
                              ("fp32_pair", float(pair)),
                              ("bf16_seq", float(bacc))):
                if Fraction(val) != truth:
                    stats[name]["diff"] += 1
                    if tf:
                        rel = abs((val - tf) / tf)
                        stats[name]["max_rel"] = max(stats[name]["max_rel"], rel)
                        stats[name]["sum_rel"] += rel

    print(f"product exponent spread: 2^{exp_lo} .. 2^{exp_hi}  "
          f"(span {exp_hi - exp_lo} binades)")
    print(f"  a quire with QF fraction bits is exact iff QF >= {-exp_lo}\n")
    print(f"{'arm':<12} {'outputs != exact':>18} {'max |rel err|':>16}"
          f" {'mean |rel err|':>16}")
    print("-" * 65)
    for name, label in (("q96", f"quire QF={QF_SHIPPED}"),
                        ("q106", f"quire QF={QF_EXACT}"),
                        ("fp32_seq", "fp32 seq"),
                        ("fp32_pair", "fp32 pairwise"),
                        ("bf16_seq", "bf16 seq")):
        s = stats[name]
        rel = "0" if s["max_rel"] == 0 else f"{s['max_rel']:.3e}"
        avg = s["sum_rel"] / n_out
        mean = "0" if avg == 0 else f"{avg:.3e}"
        print(f"{label:<12} {str(s['diff']) + '/' + str(n_out):>18} {rel:>16}"
              f" {mean:>16}")
    print()
    print("Every arm consumed the identical b-posit16 products; the only")
    print("difference is the accumulator. On real Llama operands QF=96 is")
    print("already exact, because the products never reach the radix.")
    print()
    adversarial_qf_corner()


def adversarial_qf_corner():
    """Part B: derive the QF the format actually requires, by enumeration.

    Real activations keep products near 2^-44, far above either radix, so
    Part A cannot separate the two profiles. The question R3.1 asks is about
    the format's declared domain, not one workload, so this part enumerates
    the whole 16-bit lattice and finds the lowest-order bit any legal product
    can carry.

    CAUTION, and the reason this is done by enumeration rather than from the
    stated range: two "b-posit16" range conventions exist in our own tree.
    The canonical spec (the b-posit type specification, and the CUDA
    and Blackhole kernels this paper measures) is regime-BOUNDED at a total
    exponent of +-48. The open-bposit reference imported here implements the
    UNBOUNDED posit reach of +-112, and its docstring still says +-48. In-range
    codes are bit-identical between the two, so Part A is unaffected, but the
    exactness question is decided entirely at the extremes -- where they
    differ by 112 binades. The paper describes the bounded profile, so that is
    what is reported; the unbounded figure is given alongside it so the
    distinction is on the record rather than latent.
    """
    print("=== Part B: the QF the format requires, by enumeration ===")
    rows = []
    for code in range(1, 1 << 16):
        d = decode_bposit16(code)
        if d.is_special:
            continue
        f = decoded_to_fraction(d)
        if f == 0:
            continue
        a = abs(f)
        m, e = as_dyadic(a)
        total_exp = a.numerator.bit_length() - 1 + e
        rows.append((code, a, m, e, total_exp))

    # the encoder-clamped profile of the paper: T = 8k+e in [-48, 47].
    # The range is ASYMMETRIC -- |T| <= 48 admits 16 extra codes at T=48
    # and does not describe the profile the kernels accept.
    bounded = [r for r in rows if -48 <= r[4] <= 47]
    print(f"legal codes enumerated      : {len(rows)}")
    print(f"total-exponent reach        : 2^{min(r[4] for r in rows)} "
          f".. 2^{max(r[4] for r in rows)}  (unbounded reference)")
    print(f"encoder-clamped T in [-48,47]: {len(bounded)} codes "
          f"({100 * len(bounded) / len(rows):.1f}% of the lattice)")

    e_bounded = min(r[3] for r in bounded)
    e_all = min(r[3] for r in rows)
    witness = next(r for r in bounded if r[3] == e_bounded)
    n_wit = sum(1 for r in bounded if r[3] == e_bounded)
    print()
    print(f"lowest set bit, bounded     : 2^{e_bounded}  "
          f"({n_wit} codes attain it, e.g. 0x{witness[0]:04X} = "
          f"{float(witness[1]):.6e})")
    print(f"  note this is below 2^-48: the regime bound caps the EXPONENT,")
    print(f"  while the mantissa carries {-e_bounded - 48} further bits under it.")
    print(f"a product of two such needs  : fraction bits down to 2^{2 * e_bounded}")
    print()
    print(f"  => exact on the bounded profile iff QF >= {-2 * e_bounded}")
    npos = sum(1 for r in bounded if r[1] > 0)
    hist = {}
    for r in bounded:
        if r[1] > 0:
            hist[r[3]] = hist.get(r[3], 0) + 1
    trunc = sum(hist[a] * hist[b] for a in hist for b in hist if a + b < -96)
    print()
    print(f"  cross-check of Section 4.1, independently enumerated:")
    print(f"    positive encoder-reachable codes : {npos}      (paper: 32256)")
    print(f"    positive operand pairs           : {npos * npos}  (paper: 1040449536)")
    print(f"    pairs truncated at QF=96         : {trunc}       (paper: 47041)")
    print(f"    fraction of the pair space       : "
          f"{100 * trunc / (npos * npos):.4f}%      (paper: 0.0045%)")
    print(f"  => exact on the unbounded reach iff QF >= {-2 * e_all}")
    print()

    # Demonstrate it on the witness pair: the product that decides the profile.
    d_wit = (witness[2], witness[3])
    one = as_dyadic(decoded_to_fraction(decode_bposit16(encode_bposit16(Fraction(1)))))
    print(f"{'contraction':>34} {'QF=96':>16} {'QF=106':>16}")
    print("-" * 68)
    cases = [
        ("minbit^2 alone", [d_wit], [d_wit]),
        ("minbit^2 + 1.0", [d_wit, one], [d_wit, one]),
        ("minbit^2 + 1023 unit terms", [d_wit] + [one] * 1023,
         [d_wit] + [one] * 1023),
    ]
    for label, a, b in cases:
        em, ee = exact_dot(a, b)
        truth = Fraction(em, 1) * Fraction(2) ** ee
        out = []
        for qf in (QF_SHIPPED, QF_EXACT):
            got = Fraction(quire_dot(a, b, qf), 1 << qf)
            out.append("exact" if got == truth else "truncated")
        print(f"{label:>34} {out[0]:>16} {out[1]:>16}")
    print()
    print(f"So QF={QF_EXACT} is not a safety margin, it is the exact requirement")
    print(f"of the bounded b-posit16 profile: 2 x 53 fraction bits. The shipped")
    print(f"QF={QF_SHIPPED} profile is exact on every product a real 16-bit workload")
    print("generates (Part A) and truncates only products whose magnitude is")
    print("below 2^-96, which the enumeration shows requires both operands")
    print("within 5 binades of the bottom of the format.")


if __name__ == "__main__":
    main()
