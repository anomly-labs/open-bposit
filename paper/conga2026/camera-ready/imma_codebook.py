# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Characterise the bposit8 -> int8 codebook that the IMMA path (Path C) uses.

The deployed LUT is value * 2^96 arithmetic-shifted right by (96 - SCALE_BITS),
i.e. truncation toward -inf onto multiples of 2^-SCALE_BITS, then saturated into
int8.  This script states, from the reference codec, exactly which bposit8 codes
survive that map without loss -- the "supported codebook" the reviewer asked for.
"""
import sys, os
from fractions import Fraction
# bposit16_reference.py (Apache-2.0) ships with github.com/anomly-labs/mosyne-bposit under
# kernels/. Set BPOSIT16_REFERENCE_DIR, or clone mosyne-bposit beside this repository.
sys.path.insert(0, os.environ.get("BPOSIT16_REFERENCE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "mosyne-bposit", "kernels")))
from bposit16_reference import decode_bposit8, decoded_to_fraction_8

SCALE_BITS = 6
S = Fraction(1 << SCALE_BITS)

exact, trunc, sat, special = [], [], [], []
for c in range(256):
    d = decode_bposit8(c)
    if d.is_special == "nar":
        special.append(c); continue
    v = decoded_to_fraction_8(d)
    x = v * S                      # the int8 the LUT wants to store
    if x > 127 or x < -128:
        sat.append((c, float(v))); continue
    if x.denominator == 1:
        exact.append((c, float(v), int(x)))
    else:
        trunc.append((c, float(v), float(x)))

print(f"SCALE_BITS={SCALE_BITS}  (bposit8 1.0 -> int8 {int(S)})")
print(f"total codes            : 256")
print(f"NaR / special          : {len(special)}  {special}")
print(f"exactly representable  : {len(exact)}")
print(f"truncated (sub-2^-6)   : {len(trunc)}")
print(f"saturated (|v| too big): {len(sat)}")
vals = [v for _, v, _ in exact]
print(f"exact value range      : [{min(vals)}, {max(vals)}]")
nz = [abs(v) for v in vals if v != 0]
print(f"exact nonzero magnitude: [{min(nz)}, {max(nz)}]")
if sat:
    print(f"smallest saturating |v|: {min(abs(v) for _, v in sat)}")
if trunc:
    print(f"largest truncated |v|  : {max(abs(v) for _, v, _ in trunc)}")

# int32 overflow bound for a K-term int8 dot product
P = 127 * 127
K = (2**31 - 1) // P
print()
print(f"int8 product bound     : |a*b| <= {P}")
print(f"INT32 exact while K <= : {K}")
print(f"  shapes in the paper   : K=2048 ok, K=4096 ok, K=14336 ok")
