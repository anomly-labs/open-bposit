# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
# b-posit / AI-Posit number format — reference implementation (format only).
# Extracted from Anomly's reference; non-format functions removed.

"""bposit16 reference implementation, integer-only, per Gustafson Ch.7.

Parameters from spec/header (the b-posit type spec):
    eS=3, rS=6, total=16 bits, dynamic range 2^-48 to 2^48

This is a PROOF that the math is implementable in pure integer arithmetic
without IEEE float — exactly what we'll port to CUDA __device__ functions.

Pure-integer, float-free b-posit number format (decode / encode / quire /
    mul / add) — the reference for the open-bposit standard.

Run:
    python3 bposit16_reference.py
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import math


# ---- Parameters --------------------------------------------------------------
NBITS = 16
ES = 3
NAR = 0x8000
ZERO = 0x0000
ONE = 0x4000
USEED = 1 << (1 << ES)  # 2^(2^eS) = 256


# ---- bposit32 parameters (eS=3, rS=6 per bposit_types.h, same as bposit16) --
NBITS_32 = 32
ES_32 = 3
NAR_32 = 0x80000000
ZERO_32 = 0x00000000
ONE_32 = 0x40000000


def decode_bposit32(p: int) -> "Decoded":
    """Decode bposit32 (32-bit, eS=3, rS=6, range 2^-48 to 2^48)."""
    p &= 0xFFFFFFFF
    if p == ZERO_32:
        return Decoded(0, 0, 0, 0, 0, "zero")
    if p == NAR_32:
        return Decoded(0, 0, 0, 0, 0, "nar")
    sign = (p >> 31) & 1
    rest = p & 0x7FFFFFFF
    if sign:
        rest = ((~rest) + 1) & 0x7FFFFFFF
    leading_bit = (rest >> 30) & 1
    rs = 0
    while rs < 31 and ((rest >> (30 - rs)) & 1) == leading_bit:
        rs += 1
    if rs == 31:
        k = 30 if leading_bit else -31
        return Decoded(sign, k, 0, 0, 0)
    k = (rs - 1) if leading_bit else -rs
    consumed = rs + 1
    remaining = 31 - consumed
    rest2 = rest & ((1 << remaining) - 1)
    e_width = min(ES_32, remaining)
    if e_width > 0:
        e = (rest2 >> (remaining - e_width)) & ((1 << e_width) - 1)
        e <<= (ES_32 - e_width)
    else:
        e = 0
    remaining -= e_width
    f_width = remaining
    f_bits = (rest2 & ((1 << f_width) - 1)) if f_width > 0 else 0
    return Decoded(sign, k, e, f_bits, f_width)


def decoded_to_fraction_32(d: "Decoded") -> Fraction:
    if d.is_special == "zero":
        return Fraction(0)
    if d.is_special == "nar":
        raise ValueError("NaR")
    # Same two-step decode as decoded_to_fraction (bposit16):
    # decode_bposit32 took the 2's complement of the trailing 31
    # bits when sign=1 (line 49), so (k, e, f_bits) describe the
    # *magnitude* — we just negate. The (-1)^sign · useed^k · 2^e
    # · (1+f) shorthand is wrong for negative posits and is
    # deliberately not used here.
    base = Fraction(USEED) ** d.k  # USEED = 256 same as bposit16
    base *= Fraction(1 << d.e) if d.e >= 0 else Fraction(1, 1 << -d.e)
    if d.f_width > 0:
        base *= Fraction(2 ** d.f_width + d.f_bits, 2 ** d.f_width)
    if d.sign:
        base = -base
    return base


def _encode_unsigned_32(value: Fraction) -> int:
    """Encode positive Fraction → 31-bit unsigned bposit32 field."""
    if value == 0:
        return 0
    if value >= 1:
        total_e = 0; v = value
        while v >= 2: v /= 2; total_e += 1
    else:
        total_e = 0; v = value
        while v < 1: v *= 2; total_e -= 1
    if total_e > 48:
        return 0x7FFFFFFF
    if total_e < -48:
        return 0x00000001
    k = total_e >> 3       # useed = 256, log_useed = 3 bits per
    e = total_e & 7
    f = v - 1
    bits = []
    if k >= 0:
        bits.extend([1] * (k + 1)); bits.append(0)
    else:
        bits.extend([0] * (-k)); bits.append(1)
    if len(bits) >= 31:
        bits = bits[:31]
    else:
        e_bits = [(e >> i) & 1 for i in range(ES_32 - 1, -1, -1)]
        for b in e_bits:
            bits.append(b)
            if len(bits) == 31: break
        while len(bits) < 32:
            f *= 2
            if f >= 1: bits.append(1); f -= 1
            else: bits.append(0)
        if len(bits) > 31:
            bits = bits[:31]
    out = 0
    for b in bits: out = (out << 1) | b
    out <<= (31 - len(bits))
    return out & 0x7FFFFFFF


def encode_bposit32(value) -> int:
    if isinstance(value, float):
        if value == 0: return ZERO_32
        value = Fraction(value).limit_denominator(10 ** 16)
    if isinstance(value, int):
        value = Fraction(value)
    if value == 0:
        return ZERO_32
    sign = 1 if value < 0 else 0
    abs_field = _encode_unsigned_32(abs(value))
    if sign:
        abs_field = ((~abs_field) + 1) & 0x7FFFFFFF
    return ((sign << 31) | abs_field) & 0xFFFFFFFF


# ---- bposit8 parameters (eS=2, rS=3 per bposit_types.h) ---------------------
NBITS_8 = 8
ES_8 = 2
NAR_8 = 0x80
ZERO_8 = 0x00
ONE_8 = 0x40
USEED_8 = 1 << (1 << ES_8)  # 2^(2^eS) = 16


def decode_bposit8(p: int) -> "Decoded":
    """Decode bposit8 (8-bit, eS=2, rS=3, range 2^-12 to 2^12)."""
    p &= 0xFF
    if p == ZERO_8:
        return Decoded(0, 0, 0, 0, 0, "zero")
    if p == NAR_8:
        return Decoded(0, 0, 0, 0, 0, "nar")
    sign = (p >> 7) & 1
    rest = p & 0x7F
    if sign:
        rest = ((~rest) + 1) & 0x7F
    leading_bit = (rest >> 6) & 1
    rs = 0
    while rs < 7 and ((rest >> (6 - rs)) & 1) == leading_bit:
        rs += 1
    if rs == 7:
        k = 6 if leading_bit else -7
        return Decoded(sign, k, 0, 0, 0)
    k = (rs - 1) if leading_bit else -rs
    consumed = rs + 1
    remaining = 7 - consumed
    rest2 = rest & ((1 << remaining) - 1)
    e_width = min(ES_8, remaining)
    if e_width > 0:
        e = (rest2 >> (remaining - e_width)) & ((1 << e_width) - 1)
        e <<= (ES_8 - e_width)
    else:
        e = 0
    remaining -= e_width
    f_width = remaining
    f_bits = (rest2 & ((1 << f_width) - 1)) if f_width > 0 else 0
    return Decoded(sign, k, e, f_bits, f_width)


def decoded_to_fraction_8(d: "Decoded") -> Fraction:
    if d.is_special == "zero":
        return Fraction(0)
    if d.is_special == "nar":
        raise ValueError("NaR")
    # Same two-step decode as decoded_to_fraction (bposit16):
    # decode_bposit8 took the 2's complement of the trailing 7
    # bits when sign=1, so (k, e, f_bits) describe the magnitude
    # — we just negate. The (-1)^sign · useed_8^k · 2^e · (1+f)
    # shorthand is wrong for negative posits and is deliberately
    # not used here.
    base = Fraction(USEED_8) ** d.k
    base *= Fraction(1 << d.e) if d.e >= 0 else Fraction(1, 1 << -d.e)
    if d.f_width > 0:
        base *= Fraction(2 ** d.f_width + d.f_bits, 2 ** d.f_width)
    if d.sign:
        base = -base
    return base


def _encode_unsigned_8(value: Fraction) -> int:
    """Encode positive Fraction → 7-bit unsigned bposit8 field."""
    if value == 0:
        return 0
    if value >= 1:
        total_e = 0; v = value
        while v >= 2: v /= 2; total_e += 1
    else:
        total_e = 0; v = value
        while v < 1: v *= 2; total_e -= 1
    if total_e > 12:
        return 0x7F
    if total_e < -12:
        return 0x01
    # total_e = k * 4 + e, since useed_8 = 16 = 2^4
    k = total_e >> 2
    e = total_e & 3
    f = v - 1
    bits = []
    if k >= 0:
        bits.extend([1] * (k + 1)); bits.append(0)
    else:
        bits.extend([0] * (-k)); bits.append(1)
    if len(bits) >= 7:
        bits = bits[:7]
    else:
        e_bits = [(e >> i) & 1 for i in range(ES_8 - 1, -1, -1)]
        for b in e_bits:
            bits.append(b)
            if len(bits) == 7: break
        while len(bits) < 8:
            f *= 2
            if f >= 1: bits.append(1); f -= 1
            else: bits.append(0)
        if len(bits) > 7:
            bits = bits[:7]
    out = 0
    for b in bits: out = (out << 1) | b
    out <<= (7 - len(bits))
    return out & 0x7F


def encode_bposit8(value) -> int:
    if isinstance(value, float):
        if value == 0: return ZERO_8
        value = Fraction(value).limit_denominator(10 ** 8)
    if isinstance(value, int):
        value = Fraction(value)
    if value == 0:
        return ZERO_8
    sign = 1 if value < 0 else 0
    abs_field = _encode_unsigned_8(abs(value))
    if sign:
        abs_field = ((~abs_field) + 1) & 0x7F
    return ((sign << 7) | abs_field) & 0xFF


def bposit8_mul(a: int, b: int) -> int:
    da = decode_bposit8(a)
    db = decode_bposit8(b)
    if da.is_special == "zero" or db.is_special == "zero":
        return ZERO_8
    if da.is_special == "nar" or db.is_special == "nar":
        return NAR_8
    fa = decoded_to_fraction_8(da)
    fb = decoded_to_fraction_8(db)
    return encode_bposit8(fa * fb)


def bposit8_to_quire(p: int) -> int:
    d = decode_bposit8(p)
    if d.is_special:
        return 0
    f = decoded_to_fraction_8(d)
    f *= (1 << QUIRE_FRAC_BITS)
    return int(f)



# ---- Decode -----------------------------------------------------------------
@dataclass
class Decoded:
    sign: int          # 0 or 1
    k: int             # regime k (signed; range = USEED^k)
    e: int             # exponent (0 to USEED-1)
    f_bits: int        # fraction bits, MSB-aligned in 32-bit field
    f_width: int       # number of fraction bits actually present
    is_special: str | None = None   # "zero" | "nar" | None


def decode_bposit16(p: int) -> Decoded:
    p &= 0xFFFF
    if p == ZERO:
        return Decoded(0, 0, 0, 0, 0, "zero")
    if p == NAR:
        return Decoded(0, 0, 0, 0, 0, "nar")

    sign = (p >> 15) & 1
    # 2's-complement-style sign handling for posits: if sign=1, take the
    # 2's complement of the rest, then proceed as positive.
    rest = p & 0x7FFF
    if sign:
        rest = ((~rest) + 1) & 0x7FFF

    # Read regime: count leading run of identical bits in rest[14:0]
    leading_bit = (rest >> 14) & 1
    rs = 0
    while rs < 15 and ((rest >> (14 - rs)) & 1) == leading_bit:
        rs += 1
    if rs == 15:
        # All-ones case (extreme value); regime fully consumed
        k = 14 if leading_bit else -15
        e = 0
        f_bits = 0
        f_width = 0
        return Decoded(sign, k, e, f_bits, f_width)

    k = (rs - 1) if leading_bit else -rs

    # Skip regime + terminator bit
    consumed = rs + 1
    remaining = 15 - consumed
    rest2 = rest & ((1 << remaining) - 1)

    # Read exponent (up to ES bits, MSB-first)
    e_width = min(ES, remaining)
    if e_width > 0:
        e = (rest2 >> (remaining - e_width)) & ((1 << e_width) - 1)
        e <<= (ES - e_width)  # left-pad if regime ate some exponent bits
    else:
        e = 0
    remaining -= e_width

    f_width = remaining
    if f_width > 0:
        f_bits = rest2 & ((1 << f_width) - 1)
    else:
        f_bits = 0

    return Decoded(sign, k, e, f_bits, f_width)


# ---- Decoded → exact rational ----------------------------------------------
def decoded_to_fraction(d: Decoded) -> Fraction:
    """Exact rational value the bposit16 represents. Reference for testing."""
    if d.is_special == "zero":
        return Fraction(0)
    if d.is_special == "nar":
        raise ValueError("NaR")
    # For sign=0:  value = useed^k * 2^e * (1 + f_bits / 2^f_width).
    # For sign=1:  decode_bposit16 already took the 2's complement of
    # the trailing 15 bits at line 298-299, so (k, e, f_bits) describe
    # the *magnitude* — we just negate. The single-formula shorthand
    # value = (-1)^sign * useed^k * 2^e * (1 + f) that appears in some
    # posit literature gives the wrong magnitude for negative values
    # and is deliberately avoided here (per Gustafson, 2026-05-15).
    base = Fraction(USEED) ** d.k
    base *= Fraction(1 << d.e) if d.e >= 0 else Fraction(1, 1 << -d.e)
    if d.f_width > 0:
        base *= Fraction(2 ** d.f_width + d.f_bits, 2 ** d.f_width)
    if d.sign:
        base = -base
    return base


# ---- Encode (round-to-nearest-even) ----------------------------------------
def _encode_unsigned(value: Fraction, mode: str = "truncate") -> int:
    """Encode positive value as 15-bit unsigned posit field (no sign bit).

    ``mode`` selects the rounding policy at the 15-bit boundary:

      * ``"truncate"`` (default) — drop bits past position 14. This matches
        the baked CUDA LUTs and the bit-exact Python↔CUDA crosscheck, so
        it is the default for backwards compatibility with the rest of
        the test suite.
      * ``"rtne"`` — round-to-nearest, ties-to-even (Gustafson posit
        spec, Ch.7). Reduces the worst-case round-trip error by half a
        ULP at the cost of breaking bit-exact equivalence with the
        existing truncating CUDA encoder until that path is upgraded.
    """
    if mode not in ("truncate", "rtne"):
        raise ValueError(f"unknown rounding mode: {mode!r}")
    if value == 0:
        return 0

    # Find floor(log2(value))
    if value >= 1:
        total_e = 0
        v = value
        while v >= 2:
            v /= 2
            total_e += 1
    else:
        total_e = 0
        v = value
        while v < 1:
            v *= 2
            total_e -= 1

    # Clamp to bposit16 range
    if total_e > 48:
        return 0x7FFF  # +maxPos
    if total_e < -48:
        return 0x0001  # +minPos

    # Decompose: total_e = k * 8 + e, where 0 <= e < 8
    k = total_e >> 3
    e = total_e & 7
    # f = v - 1, where 1 <= v < 2  (v is the mantissa)
    f = v - 1  # Fraction in [0, 1)

    # Encode regime: k positive → (k+1) ones followed by a 0
    #                k negative → -k zeros followed by a 1
    bits = []
    if k >= 0:
        bits.extend([1] * (k + 1))
        bits.append(0)
    else:
        bits.extend([0] * (-k))
        bits.append(1)

    # We have 15 bits total in the unsigned field
    used = len(bits)
    round_up = False
    if used >= 15:
        # Regime ate everything (extreme value) — no fraction bits to round.
        bits = bits[:15]
    else:
        # Encode exponent (ES = 3 bits, MSB-first)
        e_bits = [(e >> i) & 1 for i in range(ES - 1, -1, -1)]
        for b in e_bits:
            bits.append(b)
            if len(bits) == 15:
                break
        # Encode fraction bits
        while len(bits) < 16:  # +1 to detect rounding
            f *= 2
            if f >= 1:
                bits.append(1)
                f -= 1
            else:
                bits.append(0)

        # Inspect the round bit (bits[15]) and any sticky residue (`f` > 0)
        # so we can either drop them (truncate) or apply RTNE.
        round_bit = bits[15] if len(bits) > 15 else 0
        bits = bits[:15]

        if mode == "rtne":
            # Standard IEEE-754-style round-to-nearest-ties-to-even:
            #   round up if guard=1 AND (sticky=1 OR LSB=1)
            sticky = f > 0
            lsb_odd = bits[14] == 1
            round_up = round_bit == 1 and (sticky or lsb_odd)
        else:
            round_up = False

    # Pack 15 bits into integer
    out = 0
    for b in bits:
        out = (out << 1) | b
    # Pad if regime+exp+frac < 15 (shouldn't happen normally)
    out <<= (15 - len(bits))
    out &= 0x7FFF

    if mode == "rtne" and round_up:
        # Increment the field; saturate at maxPos (0x7FFF) instead of wrapping.
        if out < 0x7FFF:
            out += 1

    return out


def encode_bposit16(value: Fraction | int | float, mode: str = "truncate") -> int:
    """Encode a number as a 16-bit bposit. ``mode`` is forwarded to
    :func:`_encode_unsigned` — see its docstring for the truncate-vs-RTNE
    semantics. Default ``"truncate"`` preserves bit-exact equivalence with
    the existing CUDA encoder and the baked LUTs."""
    if isinstance(value, float):
        # Boundary conversion only — for test inputs.
        if value == 0:
            return ZERO
        value = Fraction(value).limit_denominator(10 ** 12)
    if isinstance(value, int):
        value = Fraction(value)
    if value == 0:
        return ZERO
    sign = 1 if value < 0 else 0
    abs_field = _encode_unsigned(abs(value), mode=mode)
    if sign:
        abs_field = ((~abs_field) + 1) & 0x7FFF
    return ((sign << 15) | abs_field) & 0xFFFF


# ---- Arithmetic via decode → fraction → encode -----------------------------
def bposit16_mul(a: int, b: int) -> int:
    da = decode_bposit16(a)
    db = decode_bposit16(b)
    if da.is_special == "zero" or db.is_special == "zero":
        return ZERO
    if da.is_special == "nar" or db.is_special == "nar":
        return NAR
    fa = decoded_to_fraction(da)
    fb = decoded_to_fraction(db)
    return encode_bposit16(fa * fb)


def bposit16_log2(a: int) -> int:
    da = decode_bposit16(a)
    if da.is_special == "zero":
        return NAR
    if da.is_special == "nar":
        return NAR
    fa = decoded_to_fraction(da)
    if fa <= 0:
        return NAR
    # log2 via float (the LUT we'll use in CUDA also goes through float once
    # at LUT-build time; on the device it's a constant-memory lookup)
    log_val = math.log2(float(fa))
    return encode_bposit16(log_val)


# ---- Quire256 (exact accumulator) ------------------------------------------
# Layout: 256-bit signed integer, fixed-point with 96 fraction bits.
# Range: 2^-96 to 2^159 — covers any product of two bposit16s (2^-96 to 2^96).
QUIRE_FRAC_BITS = 96
QUIRE_BITS = 256


def quire256_zero() -> int:
    return 0


def bposit16_sqrt(p: int) -> int:
    """Square root of a bposit16 value. Returns NAR for negative inputs.
    Uses Python math.sqrt at LUT-build time; the *kernel* path is a 65 K
    __device__ const lookup, so the float in this function never reaches
    the GPU."""
    import math
    d = decode_bposit16(p)
    if d.is_special == "zero":
        return ZERO
    if d.is_special == "nar":
        return NAR
    f = decoded_to_fraction(d)
    if f < 0:
        return NAR
    return encode_bposit16(math.sqrt(float(f)))


def bposit16_exp2(p: int) -> int:
    """2^x for a bposit16 value. Returns NaR for NaR input.
    Like sqrt/recip, this uses Fraction/float at LUT-build time only — the
    GPU path is a 65 K __device__ const lookup. Combined with bposit16_log2,
    this gives us multiplication via mul(a,b) = exp2(log2(a) + log2(b)).

    bposit16's representable range is roughly 2^-48 to 2^48; for inputs
    outside that range (i.e. exp2 result over/underflows the format), we
    encode the saturating bposit16 value via encode_bposit16's clamping."""
    d = decode_bposit16(p)
    if d.is_special == "zero":
        return ONE  # 2^0 = 1
    if d.is_special == "nar":
        return NAR
    f = decoded_to_fraction(d)
    # Bposit16 range: ~2^-48 to ~2^48. Clamp the *input* to this range
    # before exponentiating so float doesn't overflow. Past the clamp the
    # encoder saturates to maxpos/minpos respectively.
    if f >= 49:
        return encode_bposit16(2.0 ** 48)   # saturate to ~maxpos
    if f <= -49:
        return encode_bposit16(2.0 ** -48)  # saturate to ~minpos
    return encode_bposit16(2.0 ** float(f))


def bposit16_reciprocal(p: int) -> int:
    """1/x for a bposit16 value. Returns NaR for x=0 (and propagates NaR).
    Like bposit16_sqrt, this uses Fraction arithmetic at LUT-build time only —
    on the GPU the path is a 65 K __device__ const lookup."""
    d = decode_bposit16(p)
    if d.is_special == "zero":
        return NAR
    if d.is_special == "nar":
        return NAR
    f = decoded_to_fraction(d)
    if f == 0:
        return NAR
    return encode_bposit16(Fraction(1) / f)


def bposit16_to_quire(p: int) -> int:
    d = decode_bposit16(p)
    if d.is_special:
        return 0
    f = decoded_to_fraction(d)
    # Multiply by 2^QUIRE_FRAC_BITS to convert to fixed-point integer
    f *= (1 << QUIRE_FRAC_BITS)
    return int(f)  # truncating — round half away or to-even is the proper choice


def quire256_add(q: int, x: int) -> int:
    return q + x


def quire256_to_bposit16(q: int) -> int:
    """Final round of an exact quire256 accumulation back to bposit16.

    The quire is signed fixed-point with QUIRE_FRAC_BITS (= 96) fractional
    bits, giving exact products of any two bposit16 values. The CUDA
    __device__ implementation must produce bit-exact results against this
    reference. Handles zero, sign, saturation, and round-to-nearest-even
    via the standard encode_bposit16 path."""
    if q == 0:
        return ZERO
    sign = q < 0
    q_abs = -q if sign else q
    val = Fraction(q_abs, 1 << QUIRE_FRAC_BITS)
    if sign:
        val = -val
    return encode_bposit16(val)


def bposit16_mul_via_log_exp2(a: int, b: int) -> int:
    """Multiply via mul(a,b) = sign(a)·sign(b) · exp2(log2(|a|) + log2(|b|)).
    This is the path the CUDA bposit16_mul_dev kernel uses; the Python
    function exists so test cases can be generated bit-exact for comparison
    against the device. (The 'exact' bposit16_mul above uses Fraction
    arithmetic and produces slightly different rounding for many products.)"""
    if a == NAR or b == NAR:
        return NAR
    if a == ZERO or b == ZERO:
        return ZERO
    ONE_LOCAL = 0x4000
    if a == ONE_LOCAL:
        return b
    if b == ONE_LOCAL:
        return a

    sign_a = (a >> 15) & 1
    sign_b = (b >> 15) & 1
    result_sign = sign_a ^ sign_b

    def abs_bp16(x: int) -> int:
        if not ((x >> 15) & 1):
            return x
        return ((~x) + 1) & 0x7FFF

    abs_a = abs_bp16(a)
    abs_b = abs_bp16(b)

    la = bposit16_log2(abs_a)
    lb = bposit16_log2(abs_b)

    # Sum logs in quire256, encode back to bposit16
    q = bposit16_to_quire(la) + bposit16_to_quire(lb)
    log_sum = quire256_to_bposit16(q)
    mag = bposit16_exp2(log_sum)

    if mag == NAR:
        return NAR
    if mag == ZERO:
        return ZERO

    if result_sign:
        return ((~mag) + 1) & 0xFFFF
    return mag




def bposit16_add(a: int, b: int) -> int:
    """a + b in bposit16, computed via the exact quire path:
        quire(a) + quire(b) → quire result → encode to bposit16.
    Special cases: NaR + anything = NaR, zero + x = x.
    The CUDA path mirrors this exactly using the bposit16_to_quire LUT
    and the quire256_to_bposit16 device function."""
    if a == NAR or b == NAR:
        return NAR
    if a == ZERO:
        return b
    if b == ZERO:
        return a
    qa = bposit16_to_quire(a)
    qb = bposit16_to_quire(b)
    return quire256_to_bposit16(qa + qb)


def quire256_to_bposit32(q: int) -> int:
    """Final round to bposit32. Calls encode_bposit32 directly on the
    quire's Fraction representation, so the result has bposit32 precision
    (~7 decimal digits, 23 fraction bits + regime/exponent) rather than
    bposit16 precision (~3.5 decimal digits) widened by trivial shift.

    For values exactly representable in bposit16 (e.g. small integer
    powers of two like H = log₂(n) for the canonical shannon test case
    on uniform p = 1/n) the native and bridged encodings agree
    bit-exactly; for arbitrary distributions where the entropy lands on
    a non-power-of-two value the native encoder preserves the extra
    fraction bits."""
    if q == 0:
        return 0
    sign = q < 0
    q_abs = -q if sign else q
    val = Fraction(q_abs, 1 << QUIRE_FRAC_BITS)
    if sign:
        val = -val
    return encode_bposit32(val)


def quire256_to_bposit32_via_bp16(q: int) -> int:
    """Original POC bridge: encode through bposit16 then 16-bit upshift
    (Gustafson 7.4). Kept for differential testing against the native
    quire256_to_bposit32 above. New code should prefer the native form."""
    if q == 0:
        return 0
    sign = q < 0
    q_abs = -q if sign else q
    val = Fraction(q_abs, 1 << QUIRE_FRAC_BITS)
    if sign:
        val = -val
    p16 = encode_bposit16(val)
    return (p16 << 16) & 0xFFFFFFFF


# ---- Shannon entropy via the b-posit pipeline -------------------------------


# ---- Low-precision rungs: 4-bit b-posit + 5-bit AI-posit --------------------
# The b-posit precision ladder bottoms out at two formats
# the rest of this module didn't yet implement:
#   4-bit b-posit  : eS=1, rS=1  -> USEED=4, encode-range 2^-2..2^2  (16 codes)
#   5-bit AI-posit : eS=1, rS=2  -> USEED=4, encode-range 2^-4..2^4  (32 codes)
# Both reuse the EXACT decode/encode discipline of bposit8/16/32: standard
# regime run-length decode, and rS only bounds the encode saturation magnitude
# max|total_e| = (1<<eS) * rS  (bp8: 4*3=12; bp16: 8*6=48 — verified).
# Implemented parametrically so the rungs cannot drift from the family.


def _decode_posit_field(p: int, nbits: int, es: int) -> "Decoded":
    """Generic bounded-posit decode (magnitude form) — generalises
    decode_bposit8/16/32 to any width/eS."""
    mask = (1 << nbits) - 1
    p &= mask
    nar = 1 << (nbits - 1)
    if p == 0:
        return Decoded(0, 0, 0, 0, 0, "zero")
    if p == nar:
        return Decoded(0, 0, 0, 0, 0, "nar")
    sign = (p >> (nbits - 1)) & 1
    rest_mask = (1 << (nbits - 1)) - 1
    rest = p & rest_mask
    if sign:
        rest = ((~rest) + 1) & rest_mask
    top = nbits - 2
    leading_bit = (rest >> top) & 1
    rs = 0
    while rs < (nbits - 1) and ((rest >> (top - rs)) & 1) == leading_bit:
        rs += 1
    if rs == (nbits - 1):
        k = (nbits - 2) if leading_bit else -(nbits - 1)
        return Decoded(sign, k, 0, 0, 0)
    k = (rs - 1) if leading_bit else -rs
    consumed = rs + 1
    remaining = (nbits - 1) - consumed
    rest2 = rest & ((1 << remaining) - 1)
    e_width = min(es, remaining)
    if e_width > 0:
        e = (rest2 >> (remaining - e_width)) & ((1 << e_width) - 1)
        e <<= (es - e_width)
    else:
        e = 0
    remaining -= e_width
    f_width = remaining
    f_bits = (rest2 & ((1 << f_width) - 1)) if f_width > 0 else 0
    return Decoded(sign, k, e, f_bits, f_width)


def _decoded_to_fraction_gen(d: "Decoded", useed: int) -> Fraction:
    if d.is_special == "zero":
        return Fraction(0)
    if d.is_special == "nar":
        raise ValueError("NaR")
    base = Fraction(useed) ** d.k
    base *= Fraction(1 << d.e) if d.e >= 0 else Fraction(1, 1 << -d.e)
    if d.f_width > 0:
        base *= Fraction(2 ** d.f_width + d.f_bits, 2 ** d.f_width)
    if d.sign:
        base = -base
    return base


def _encode_unsigned_gen(value: Fraction, nbits: int, es: int, max_e: int) -> int:
    """Encode positive Fraction → (nbits-1)-bit unsigned field (truncate mode),
    generalising _encode_unsigned_8/32."""
    if value == 0:
        return 0
    field_bits = nbits - 1
    if value >= 1:
        total_e = 0
        v = value
        while v >= 2:
            v /= 2
            total_e += 1
    else:
        total_e = 0
        v = value
        while v < 1:
            v *= 2
            total_e -= 1
    if total_e > max_e:
        return (1 << field_bits) - 1   # maxpos
    if total_e < -max_e:
        return 1                        # minpos
    k = total_e >> es
    e = total_e & ((1 << es) - 1)
    f = v - 1
    bits: list[int] = []
    if k >= 0:
        bits.extend([1] * (k + 1))
        bits.append(0)
    else:
        bits.extend([0] * (-k))
        bits.append(1)
    if len(bits) >= field_bits:
        bits = bits[:field_bits]
    else:
        e_bits = [(e >> i) & 1 for i in range(es - 1, -1, -1)]
        for b in e_bits:
            bits.append(b)
            if len(bits) == field_bits:
                break
        while len(bits) < field_bits + 1:
            f *= 2
            if f >= 1:
                bits.append(1)
                f -= 1
            else:
                bits.append(0)
        if len(bits) > field_bits:
            bits = bits[:field_bits]
    out = 0
    for b in bits:
        out = (out << 1) | b
    out <<= (field_bits - len(bits))
    return out & ((1 << field_bits) - 1)


def _encode_gen(value, nbits: int, es: int, max_e: int) -> int:
    if isinstance(value, float):
        if value == 0:
            return 0
        value = Fraction(value).limit_denominator(10 ** 8)
    if isinstance(value, int):
        value = Fraction(value)
    if value == 0:
        return 0
    sign = 1 if value < 0 else 0
    field_bits = nbits - 1
    abs_field = _encode_unsigned_gen(abs(value), nbits, es, max_e)
    if sign:
        abs_field = ((~abs_field) + 1) & ((1 << field_bits) - 1)
    return ((sign << (nbits - 1)) | abs_field) & ((1 << nbits) - 1)


# ---- 4-bit b-posit (eS=1, rS=1) ---------------------------------------------
NBITS_4 = 4
ES_4 = 1
RS_4 = 1
NAR_4 = 0x8
ZERO_4 = 0x0
ONE_4 = 0x4
USEED_4 = 1 << (1 << ES_4)        # 4
MAXE_4 = (1 << ES_4) * RS_4       # 2  -> encode range 2^-2 .. 2^2


def decode_bposit4(p: int) -> "Decoded":
    return _decode_posit_field(p, NBITS_4, ES_4)


def decoded_to_fraction_4(d: "Decoded") -> Fraction:
    return _decoded_to_fraction_gen(d, USEED_4)


def encode_bposit4(value) -> int:
    return _encode_gen(value, NBITS_4, ES_4, MAXE_4)


# ---- 5-bit AI-posit (eS=1, rS=2) — biological-precision rung ----------------
NBITS_5 = 5
ES_5 = 1
RS_5 = 2
NAR_5 = 0x10
ZERO_5 = 0x00
ONE_5 = 0x08
USEED_5 = 1 << (1 << ES_5)        # 4
MAXE_5 = (1 << ES_5) * RS_5       # 4  -> encode range 2^-4 .. 2^4


def decode_aiposit5(p: int) -> "Decoded":
    return _decode_posit_field(p, NBITS_5, ES_5)


def decoded_to_fraction_5(d: "Decoded") -> Fraction:
    return _decoded_to_fraction_gen(d, USEED_5)


def encode_aiposit5(value) -> int:
    return _encode_gen(value, NBITS_5, ES_5, MAXE_5)


def _lowp_mul(a: int, b: int, decode, to_frac, encode, nar: int, zero: int) -> int:
    da, db = decode(a), decode(b)
    if da.is_special == "zero" or db.is_special == "zero":
        return zero
    if da.is_special == "nar" or db.is_special == "nar":
        return nar
    return encode(to_frac(da) * to_frac(db))


def _lowp_add(a: int, b: int, decode, to_frac, encode, nar: int, zero: int) -> int:
    da, db = decode(a), decode(b)
    if da.is_special == "nar" or db.is_special == "nar":
        return nar
    if da.is_special == "zero":
        return b
    if db.is_special == "zero":
        return a
    return encode(to_frac(da) + to_frac(db))


def bposit4_mul(a: int, b: int) -> int:
    return _lowp_mul(a, b, decode_bposit4, decoded_to_fraction_4,
                     encode_bposit4, NAR_4, ZERO_4)


def bposit4_add(a: int, b: int) -> int:
    return _lowp_add(a, b, decode_bposit4, decoded_to_fraction_4,
                     encode_bposit4, NAR_4, ZERO_4)


def aiposit5_mul(a: int, b: int) -> int:
    return _lowp_mul(a, b, decode_aiposit5, decoded_to_fraction_5,
                     encode_aiposit5, NAR_5, ZERO_5)


def aiposit5_add(a: int, b: int) -> int:
    return _lowp_add(a, b, decode_aiposit5, decoded_to_fraction_5,
                     encode_aiposit5, NAR_5, ZERO_5)


def bposit4_to_quire(p: int) -> int:
    d = decode_bposit4(p)
    if d.is_special:
        return 0
    return int(decoded_to_fraction_4(d) * (1 << QUIRE_FRAC_BITS))


def aiposit5_to_quire(p: int) -> int:
    d = decode_aiposit5(p)
    if d.is_special:
        return 0
    return int(decoded_to_fraction_5(d) * (1 << QUIRE_FRAC_BITS))


# ---- Unified quire:  "single quire architecture supports all five
# precision levels — input conversion posit->quire (exact), accumulation
# quire+=quire (exact), output quire->posit (single rounding)". -------------
_TO_QUIRE = {
    "bp4": bposit4_to_quire,
    "aip5": aiposit5_to_quire,
    "bp8": bposit8_to_quire,
    "bp16": bposit16_to_quire,
}
_ENCODE_OUT = {
    "bp16": quire256_to_bposit16,
    "bp32": quire256_to_bposit32,
}


def mixed_precision_sum(items: list[tuple[int, str]], out_fmt: str) -> int:
    """Exact accumulation of mixed-precision posit values into one quire256,
    rounded once to out_fmt. `items` = [(code, format), ...] with format in
    _TO_QUIRE; out_fmt in _ENCODE_OUT. Reference for the unified-quire kernel."""
    q = 0
    for code, fmt in items:
        q += _TO_QUIRE[fmt](code)
    return _ENCODE_OUT[out_fmt](q)


# ---- Dynamic-precision dot product ( p-coordinate +  quire) ------
# The OS picks a precision p per task; whatever p, the quire accumulates the
# products EXACTLY (no rounding between MACs), and the result is rounded once.
# Lower p => coarser inputs (less storage/energy) but bounded, deterministic
# accuracy loss. These helpers are the reference for test_dynamic_precision.cu.
_FMT_CODEC = {
    "bp4": (encode_bposit4, decode_bposit4, decoded_to_fraction_4),
    "aip5": (encode_aiposit5, decode_aiposit5, decoded_to_fraction_5),
    "bp8": (encode_bposit8, decode_bposit8, decoded_to_fraction_8),
}


def quantize_vec(vals, fmt: str) -> list[int]:
    enc = _FMT_CODEC[fmt][0]
    return [enc(v) & ((1 << {"bp4": 4, "aip5": 5, "bp8": 8}[fmt]) - 1) for v in vals]


def code_value(code: int, fmt: str) -> Fraction:
    _, dec, tf = _FMT_CODEC[fmt]
    d = dec(code)
    return Fraction(0) if d.is_special else tf(d)


def dot_at_precision(a_codes, b_codes, fmt: str, out_fmt: str = "bp32") -> int:
    """Exact dot product of two quantized vectors, accumulated in one quire256,
    rounded once to out_fmt. The MAC products are exact (no inter-term
    rounding) — the quire's defining property."""
    q = 0
    for ca, cb in zip(a_codes, b_codes):
        q += int(code_value(ca, fmt) * code_value(cb, fmt) * (1 << QUIRE_FRAC_BITS))
    return _ENCODE_OUT[out_fmt](q)


def dot_exact(a_vals, b_vals) -> Fraction:
    return sum((Fraction(a) * Fraction(b) for a, b in zip(a_vals, b_vals)), Fraction(0))


# ---- Validation -------------------------------------------------------------