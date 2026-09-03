#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden vectors for the quire's sub-binary-point products (CoNGA'26 §4).

Finding (verify_paper_math.py): with the binary point at bit 96, bounded
b-posit16 products of two values |a|,|b| <~ 2^-37 have LSBs below 2^-96
(47,041 of the 1.04e9 positive operand pairs). An earlier revision of the
kernel (bp16_quire.h `if (shift < 0) return;`) DROPPED the whole product while
the Python oracle TRUNCATED it toward zero; the two disagreed. The published
kernel truncates toward zero exactly like the oracle (check_quire_patch.py is
the host gate for that), and the paper scopes its exactness claim to products
whose LSB lies at or above 2^-96. These vectors document the gap and the
resolution.

This script (CPU only, real reference codec) emits:

  1. pair_vectors   — every positive bounded operand pair whose product LSB is
                      below 2^-96, with {exact, oracle_truncate, prepatch_kernel_drop}
                      values in 2^-96 units (exact as a rational string).
  2. dot_vectors    — dot products built ONLY from such pairs whose exact sum
                      is >= 2^-96, i.e. representable in the quire, yet both
                      pre-patch kernel and oracle return 0. This is the sharp version of
                      the finding: it is not "below resolution", it is a lost
                      representable contribution.
  3. frac106 check  — the same enumeration with QUIRE_FRAC_BITS = 106: proves
                      every bounded pair is placeable and reports the headroom.

Usage: tiny_product_vectors.py [--out vectors.json] [--frac 96]
The JSON can be fed to the Blackhole validate tests (targets/blackhole).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction

# bposit16_reference.py (Apache-2.0) ships with github.com/anomly-labs/mosyne-bposit
# under kernels/.  Set BPOSIT16_REFERENCE_DIR, or clone mosyne-bposit beside this repo.
HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
for c in (os.environ.get("BPOSIT16_REFERENCE_DIR", ""),
          os.path.join(HERE, "../../../../mosyne-bposit/kernels"),
          os.path.join(HOME, "development/mosyne-bposit/kernels"),
          os.path.join(HOME, "development/mosyne-bposit-public/kernels")):
    if c and os.path.exists(os.path.join(c, "bposit16_reference.py")):
        sys.path.insert(0, os.path.abspath(c))
        break
else:
    sys.exit("bposit16_reference.py not found: set BPOSIT16_REFERENCE_DIR or clone "
             "https://github.com/anomly-labs/mosyne-bposit beside this repository")
import bposit16_reference as R  # noqa: E402


def bounded_codes():
    """(code, E2, M) for every positive bounded (k in [-6,5]) b-posit16."""
    out = []
    for p in range(1, 1 << 15):
        d = R.decode_bposit16(p)
        if d.is_special or not (-6 <= d.k <= 5):
            continue
        E2 = 8 * d.k + d.e - d.f_width
        M = (1 << d.f_width) + d.f_bits if d.f_width else 1
        out.append((p, E2, M))
    return out


def place(Ma, Mb, ea, eb, frac):
    """Return (exact Fraction in units of 2^-frac, oracle_trunc int, pre-patch kernel int)."""
    sh = ea + eb + frac
    exact = Fraction(Ma * Mb) * Fraction(2) ** sh
    if sh >= 0:
        v = (Ma * Mb) << sh
        return exact, v, v
    return exact, (Ma * Mb) >> (-sh), 0     # oracle truncates, pre-patch kernel dropped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--frac", type=int, default=96)
    a = ap.parse_args(argv)

    codes = bounded_codes()
    minE2 = min(e for _, e, _ in codes)
    # only pairs that can reach below the binary point matter
    small = [x for x in codes if x[1] + max(e for _, e, _ in codes) + a.frac < 0
             or x[1] <= -(a.frac // 2) + 5]
    pairs = []
    n_inexact = n_diverge = 0
    for pa, ea, Ma in small:
        for pb, eb, Mb in small:
            if pb < pa:
                continue
            exact, orc, ker = place(Ma, Mb, ea, eb, a.frac)
            if ea + eb + a.frac >= 0:
                continue
            inexact = exact != orc
            diverge = orc != ker
            n_inexact += inexact * (1 if pa == pb else 2)
            n_diverge += diverge * (1 if pa == pb else 2)
            if inexact or diverge:
                pairs.append(dict(a=f"0x{pa:04x}", b=f"0x{pb:04x}", E2a=ea, E2b=eb,
                                  Ma=Ma, Mb=Mb, shift=ea + eb + a.frac,
                                  exact_units=str(exact), oracle_truncate=orc,
                                  prepatch_kernel_drop=ker))
    n = len(codes)
    print(f"[tiny] frac={a.frac}: {n} bounded positive codes, min E2 = {minE2} "
          f"(product LSB down to 2^{2*minE2})")
    print(f"[tiny] ordered pairs with product LSB below 2^-{a.frac}: "
          f"inexact (oracle != exact) {n_inexact:,} / {n*n:,} = {100*n_inexact/(n*n):.4f} %; "
          f"pre-patch kernel != oracle {n_diverge:,}")

    # headroom with this binary point
    maxv = max(R.decoded_to_fraction(R.decode_bposit16(p)) for p, _, _ in codes)
    top = maxv.numerator.bit_length() - maxv.denominator.bit_length()  # floor(log2) +-1
    prod_top_bit = 2 * (top + 1) + a.frac
    print(f"[tiny] max |value| ~ 2^{top} -> max product < 2^{2*(top+1)} at fixed-point bit "
          f"{prod_top_bit}; same-sign headroom 2^{255 - prod_top_bit} MACs "
          f"({2**(255-prod_top_bit)/(16e9)/86400:.1f} days at 16 lanes / 1 GHz)")

    # dot products of tiny pairs whose exact sum IS representable
    dots = []
    if pairs:
        # repeat a pre-patch-kernel-dropped pair K = denominator times so the exact sum is
        # an integer number of 2^-frac units, i.e. exactly representable.
        # Pick the one with the largest exact value (worst absolute loss).
        best = max(pairs, key=lambda r: Fraction(r["exact_units"]))
        ex = Fraction(best["exact_units"])
        K = ex.denominator
        dots.append(dict(a=[best["a"]] * K, b=[best["b"]] * K, K=K,
                         exact_units=str(ex * K),
                         exact_is_representable=(ex * K).denominator == 1,
                         oracle_truncate=best["oracle_truncate"] * K,
                         prepatch_kernel_drop=0))
        print(f"[tiny] dot: {K} x ({best['a']}*{best['b']}) exact = {ex*K} units of 2^-{a.frac} "
              f"(integer -> representable); oracle {best['oracle_truncate']*K}, "
              f"pre-patch kernel 0  -> a REPRESENTABLE sum was lost by the pre-patch kernel and is mis-summed by the oracle")
        # and the smallest-magnitude case (largest K): the 0x0101 * 0x0101 corner
        corner = min(pairs, key=lambda r: Fraction(r["exact_units"]))
        exc = Fraction(corner["exact_units"]); Kc = exc.denominator
        dots.append(dict(a=[corner["a"]] * Kc, b=[corner["b"]] * Kc, K=Kc,
                         exact_units=str(exc * Kc), exact_is_representable=True,
                         oracle_truncate=corner["oracle_truncate"] * Kc, prepatch_kernel_drop=0))
        print(f"[tiny] dot: {Kc} x ({corner['a']}*{corner['b']}) exact = {exc*Kc} units; "
              f"oracle {corner['oracle_truncate']*Kc}, pre-patch kernel 0")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(frac_bits=a.frac, n_bounded_codes=n, min_E2=minE2,
                           n_inexact_ordered_pairs=n_inexact,
                           n_prepatch_kernel_vs_oracle_divergent=n_diverge,
                           pair_vectors=pairs, dot_vectors=dots), fh, indent=1)
        print(f"[tiny] wrote {len(pairs)} pair vectors + {len(dots)} dot vectors -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
