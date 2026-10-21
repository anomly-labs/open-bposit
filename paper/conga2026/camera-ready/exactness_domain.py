# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact-domain analysis of the 256-bit quire for the bounded b-posit16 profile.

Answers CoNGA'26 Review 3's main concern with exact rational arithmetic:
  - what fractional resolution makes the quire exact over the WHOLE profile,
  - what accumulation headroom that costs,
  - and (for the shipped QF=96 design) exactly which operand pairs truncate.
No floating point is used anywhere in the derivation.
"""
import sys, os
from fractions import Fraction
# bposit16_reference.py (Apache-2.0) ships with github.com/anomly-labs/mosyne-bposit under
# kernels/. Set BPOSIT16_REFERENCE_DIR, or clone mosyne-bposit beside this repository.
sys.path.insert(0, os.environ.get("BPOSIT16_REFERENCE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "mosyne-bposit", "kernels")))
import bposit16_reference as R

ZERO, NAR = 0x0000, 0x8000

def lsb_exp(fr: Fraction) -> int:
    """value = odd * 2^L  ->  return L (exact, fr != 0)."""
    num, den = abs(fr.numerator), fr.denominator
    L = 0
    while num % 2 == 0:
        num //= 2; L += 1
    while den % 2 == 0:
        den //= 2; L -= 1
    assert den == 1, f"non-dyadic value {fr}"
    return L

def msb_exp(fr: Fraction) -> int:
    """smallest n with |value| < 2^n."""
    a = abs(fr)
    n = 0
    while Fraction(2) ** n <= a:
        n += 1
    while Fraction(2) ** (n - 1) > a:
        n -= 1
    return n

# ---- enumerate the bounded (encoder-reachable) profile ---------------------
vals = {}          # code -> Fraction (positive codes only; symmetric)
T_seen = set()
for p in range(1 << 16):
    if p in (ZERO, NAR):
        continue
    d = R.decode_bposit16(p)
    T = 8 * d.k + d.e
    T_seen.add(T)
    if not (-48 <= T <= 47):
        continue                      # encoder clamps these away
    v = R.decoded_to_fraction(d)
    if v > 0:
        vals[p] = v

print(f"decoder T range            : [{min(T_seen)}, {max(T_seen)}]")
print(f"bounded positive codes     : {len(vals)}")

Ls = {p: lsb_exp(v) for p, v in vals.items()}
Lmin = min(Ls.values())
pmin = min(Ls, key=lambda p: Ls[p])
Vmax = max(vals.values())
pmax = max(vals, key=lambda p: vals[p])
Vmin = min(vals.values())

print(f"min value LSB exponent     : 2^{Lmin}   (code 0x{pmin:04x}, value {vals[pmin]})")
print(f"max magnitude              : 2^{msb_exp(Vmax)}-  (code 0x{pmax:04x})")
print(f"min magnitude              : {Vmin} = 2^{lsb_exp(Vmin)}")

# ---- what the PRODUCT range demands --------------------------------------
prod_lsb_min = 2 * Lmin                      # LSB(a*b) = LSB(a)+LSB(b)
prod_msb_max = msb_exp(Vmax * Vmax)
frac_needed = -prod_lsb_min
int_needed = prod_msb_max
print()
print(f"product LSB floor          : 2^{prod_lsb_min}  -> {frac_needed} fractional bits needed")
print(f"product MSB ceiling        : < 2^{prod_msb_max} -> {int_needed} integer bits needed")
print(f"total bits for exactness   : {frac_needed + int_needed}  (quire is 256)")
# The quire is a SIGNED two's-complement fixed-point register, so one bit of the
# 256 is the sign: magnitude growth headroom above the product field is
#   256 - frac - int - 1.
for QF in (96, 106):
    g = 256 - QF - int_needed - 1
    sec = (2 ** g) / (16 * 1e9)
    yrs = sec / (365.2425 * 86400)
    exact = "EXACT over whole profile" if QF >= frac_needed else \
            "truncates the tiny-product corner"
    print(f"  QF={QF:3d}: product field bits {QF+int_needed}, guard {g} bits -> "
          f"depth 2^{g} same-sign MACs = {sec:.4g} s = {sec/86400:.3g} days "
          f"= {yrs:.4g} years   [{exact}]")

# ---- shipped QF=96: exactly which pairs truncate -------------------------
# LSB(a*b) < 2^-96  <=>  L(a)+L(b) < -96.  Count over positive pairs.
from collections import Counter
hist = Counter(Ls.values())
keys = sorted(hist)
total = bad = 0
for la in keys:
    for lb in keys:
        n = hist[la] * hist[lb]
        total += n
        if la + lb < -96:
            bad += n
print()
print(f"QF=96 positive operand pairs        : {total:,}")
print(f"  pairs with product LSB < 2^-96    : {bad:,}  ({100.0*bad/total:.4f} %)")
thr = max(l for l in keys if 2 * l < -96)
print(f"  both operands need LSB <= 2^{thr//1} for truncation (paper: |a|,|b| ~< 2^-37)")
smallest = min(vals.values())
print(f"  worked example: {hex(pmin)}^2 = ({vals[pmin]})^2 = "
      f"{vals[pmin]**2} , LSB 2^{2*Ls[pmin]}")

# R3's own example: a = (33/32)*2^-48
a = Fraction(33, 32) * Fraction(1, 2 ** 48)
print(f"  R3's example a=(33/32)2^-48: a^2 = {a*a} = (1089/1024)*2^-96, "
      f"LSB 2^{lsb_exp(a*a)}; representable at QF=106? "
      f"{lsb_exp(a*a) >= -106}; at QF=96? {lsb_exp(a*a) >= -96}")
enc = R.encode_bposit16(a)
print(f"    is a itself a bounded b-posit16 code? encode -> 0x{enc:04x}, "
      f"decodes to {R.decoded_to_fraction(R.decode_bposit16(enc))}, exact={R.decoded_to_fraction(R.decode_bposit16(enc))==a}")

# ---- QF=106 exactness over the whole profile (exact argument + check) ----
worst = min(Ls.values()) * 2
print()
print(f"QF=106 verdict: worst product LSB over the profile is 2^{worst}; "
      f"106 fractional bits place it exactly -> {worst >= -106}")
