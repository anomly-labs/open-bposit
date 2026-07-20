# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
# Round-trip completeness: every finite code must satisfy encode(value(code)) == code.
# Guards the encoder-clamp fix (the per-format min/maxpos clamp must sit at the true
# reachable exponent range (2^es)*(nbits-2), not a too-small value that makes the
# extreme-regime codes unreachable and collapses them to min/maxpos).
from __future__ import annotations
from fractions import Fraction

import bposit_ref as bp


def _val16(c: int):
    d = bp.decode_bposit16(c)
    if d.is_special:
        return None
    m = Fraction((1 << d.f_width) + d.f_bits, 1 << d.f_width)
    v = m * (Fraction(256) ** d.k) * (Fraction(2) ** d.e)   # useed_16 = 2^(2^3) = 256
    return -v if d.sign else v


def main() -> int:
    fails = 0

    # small formats: exhaustive via code_value
    for fmt, nb, enc in (("bp4", 4, bp.encode_bposit4),
                         ("aip5", 5, bp.encode_aiposit5),
                         ("bp8", 8, bp.encode_bposit8)):
        bad = [c for c in range(1 << nb)
               if bp.code_value(c, fmt) != 0 and enc(bp.code_value(c, fmt)) != c]
        print(f"  {fmt}: {(1 << nb) - len(bad)}/{1 << nb} codes round-trip"
              + ("" if not bad else f"  FAIL {[hex(c) for c in bad][:12]}"))
        fails += len(bad)

    # bp16: exhaustive over all 65536 codes
    bad16 = 0
    for c in range(1 << 16):
        v = _val16(c)
        if v is None or v == 0:
            continue
        if bp.encode_bposit16(v) != c:
            bad16 += 1
    print(f"  bp16: {(1 << 16) - bad16}/65536 codes round-trip" + ("" if not bad16 else "  FAIL"))
    fails += bad16

    # bp32: extremes (where the clamp bug lived) + a large random sample
    import random
    rng = random.Random(20260720)
    codes = list(range(1, 64)) + list(range(0x7FFFFFFF - 64, 0x7FFFFFFF)) + \
            [rng.randrange(1, 1 << 32) for _ in range(200000)]
    bad32 = 0
    for c in codes:
        d = bp.decode_bposit32(c)
        if d.is_special:
            continue
        v = bp.decoded_to_fraction_32(d)
        if v == 0:
            continue
        if bp.encode_bposit32(v) != c:
            bad32 += 1
    print(f"  bp32: {len(codes) - bad32}/{len(codes)} codes round-trip (extremes + sample)"
          + ("" if not bad32 else "  FAIL"))
    fails += bad32

    print("roundtrip_test:", "ALL COMPLETE" if fails == 0 else f"FAILED ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
