# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent exact reference for the Blackhole Q_F = 106 confirmation (R3.1).

The SFPU example validates the device against a scalar host chain, but both
share one `BP16_QFRAC` define, so agreement proves implementation consistency
and nothing about exactness. This script supplies the third, independent
reference: the exact rational dot product, rendered into the same 8 x uint32
little-endian two's-complement quire layout at a chosen Q_F, computed in
Python from the b-posit16 reference rather than from either C path.

Test vectors are chosen to straddle the Q_F = 96 radix. Code 0x0101 is
33 * 2^-53, the bounded profile's minimum least-significant bit, so
0x0101 * 0x0101 has its lowest product bit at 2^-106: placement shift -10 at
Q_F = 96 (truncated) and 0 at Q_F = 106 (exact). Code 0x4000 is 1.0, whose
products stay far above either radix and act as the control.

Expected outcome:
  Q_F = 96  -- lanes 0..23 differ from exact; lanes 24..31 agree.
  Q_F = 106 -- all 32 lanes agree with exact.
"""
import sys, os
from fractions import Fraction

# bposit16_reference.py (Apache-2.0) ships with github.com/anomly-labs/mosyne-bposit under
# kernels/. Set BPOSIT16_REFERENCE_DIR, or clone mosyne-bposit beside this repository.
sys.path.insert(0, os.environ.get("BPOSIT16_REFERENCE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "mosyne-bposit", "kernels")))
from bposit16_reference import decode_bposit16, decoded_to_fraction  # noqa: E402

LANES = 32
K = 4
N_LIMB = 8
LOW = 0x0101      # 33 * 2^-53, minimum lsb of the bounded profile
LOW2 = 0x0103     # 35 * 2^-53
ONE = 0x4000      # 1.0


def make_inputs():
    """The same vectors the patched host program builds. Four lane groups."""
    A = [[0] * LANES for _ in range(K)]
    B = [[0] * LANES for _ in range(K)]
    for k in range(K):
        for lane in range(LANES):
            if lane < 8:
                a, b = LOW, LOW                      # both below the radix
            elif lane < 16:
                a, b = LOW, LOW2                     # both below the radix
            elif lane < 24:
                a, b = (LOW, LOW) if k % 2 == 0 else (LOW, ONE)   # mixed
            else:
                a = ((lane * 2129) + (k * 911) + 3) & 0xFFFF      # control
                b = ((lane * 1471) + (k * 1303) + 7) & 0xFFFF
            A[k][lane], B[k][lane] = a, b
    return A, B


def value(code):
    d = decode_bposit16(code)
    if d.is_special:
        return Fraction(0)
    return decoded_to_fraction(d)


def trunc_toward_zero(f, qf):
    """Place f in fixed point with qf fraction bits, truncating toward zero --
    the rule bposit16_to_quire and the SFPU kernel both apply."""
    scaled = f * (1 << qf)
    n, d = scaled.numerator, scaled.denominator
    q = abs(n) // d
    return -q if n < 0 else q


def limbs(i, n_limb=N_LIMB):
    """Signed integer -> n_limb little-endian uint32 two's complement."""
    mask = (1 << (32 * n_limb)) - 1
    u = i & mask
    return [(u >> (32 * j)) & 0xFFFFFFFF for j in range(n_limb)]


def main():
    qfs = [int(a) for a in sys.argv[1:]] or [96, 106]
    A, B = make_inputs()

    for qf in qfs:
        print(f"=== Q_F = {qf} ===")
        n_diff = 0
        for lane in range(LANES):
            # exact: sum of exact rational products, no rounding at all
            exact = sum(value(A[k][lane]) * value(B[k][lane]) for k in range(K))
            # quire: each product truncated at the radix on entry, then summed
            quire = sum(trunc_toward_zero(value(A[k][lane]) * value(B[k][lane]), qf)
                        for k in range(K))
            # The question is whether the quire's VALUE equals the exact
            # rational sum -- NOT whether it equals that sum truncated to the
            # same Q_F, which would compare the quire against itself and hide
            # every truncation.
            same = (Fraction(quire, 1 << qf) == exact)
            n_diff += (not same)
            if lane in (0, 8, 16, 24) or not same:
                tag = "exact" if same else "TRUNCATED"
                print(f"  lane {lane:2d}  {tag:>9}  quire limbs = "
                      + " ".join(f"{x:08x}" for x in limbs(quire)))
        print(f"  lanes differing from exact: {n_diff}/{LANES}\n")

    print("Device/host limbs from the SFPU run must equal the 'quire limbs'")
    print("above at the matching Q_F. Exactness is the separate question of")
    print("whether those limbs also equal the exact rational sum, which is")
    print("what the 'TRUNCATED' tag reports.")


if __name__ == "__main__":
    main()
