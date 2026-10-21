# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prove the QF=106 quire layout is well-formed over the whole bounded b-posit16
profile, using the kernel's own (sign, M, E2) decode convention.

Invariants the kernel relies on (bp16_quire.h):
  products: shift = Ea+Eb+QFRAC must satisfy 0 <= shift and shift+21 <= 255
  singles : shift = E2+QFRAC      must satisfy 0 <= shift and shift+10 <= 255
"""
import sys, os
# bposit16_reference.py (Apache-2.0) ships with github.com/anomly-labs/mosyne-bposit under
# kernels/. Set BPOSIT16_REFERENCE_DIR, or clone mosyne-bposit beside this repository.
sys.path.insert(0, os.environ.get("BPOSIT16_REFERENCE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "mosyne-bposit", "kernels")))
import bposit16_reference as R

def kernel_decode(p):
    """Transcription of bp16_decode (bp16_decode.h) -> (sign, M, E2)."""
    d = R.decode_bposit16(p)
    if d.is_special:
        return None
    M = (1 << d.f_width) + d.f_bits if d.f_width > 0 else 1
    E2 = 8 * d.k + d.e - (d.f_width if d.f_width > 0 else 0)
    return d.sign, M, E2

bounded = []
for p in range(1 << 16):
    r = kernel_decode(p)
    if r is None:
        continue
    sign, M, E2 = r
    d = R.decode_bposit16(p)
    if not (-48 <= 8 * d.k + d.e <= 47):
        continue                      # not encoder-reachable
    bounded.append((p, M, E2))

Es = sorted({e for _, _, e in bounded})
Ms = {m for _, m, _ in bounded}
print(f"bounded non-special codes : {len(bounded)}")
print(f"E2 range                  : [{min(Es)}, {max(Es)}]")
print(f"M range                   : [{min(Ms)}, {max(Ms)}]  (M < 2^11 = {max(Ms) < 2048})")

for QF in (96, 106):
    # products
    lo = 2 * min(Es) + QF
    hi = 2 * max(Es) + QF
    under = sum(1 for a in Es for b in Es if a + b + QF < 0)
    over = sum(1 for a in Es for b in Es if a + b + QF + 21 > 255)
    # singles
    slo, shi = min(Es) + QF, max(Es) + QF
    print(f"\nQF={QF}:")
    print(f"  product shift range   : [{lo}, {hi}]   "
          f"(need >=0 and +21<=255 i.e. <=234)")
    print(f"  E2-pair combos under 0: {under} / {len(Es)**2}; "
          f"over 234: {over} / {len(Es)**2}")
    print(f"  single shift range    : [{slo}, {shi}]   (need >=0 and +10<=245)")
    ok = lo >= 0 and hi <= 234 and slo >= 0 and shi <= 245
    print(f"  WELL-FORMED, no truncation, no overflow: {ok}")
