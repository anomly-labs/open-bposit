/* Copyright (c) 2026 Anomly, Inc.
 * SPDX-License-Identifier: Apache-2.0 */
/* ieee_softfp32.h — IEEE-754 binary32 (fp32) ADD in PURE INTEGER arithmetic,
 * round-to-nearest-ties-to-even, RV32IM-clean (no `float`, no libgcc soft-float).
 *
 * WHY pure integer (and not native `float`): the tt-metal data-movement baby
 * cores (BRISC/NCRISC) have no hardware FPU, so a native `float +` would compile
 * to a libgcc soft-float call (__addsf3) whose presence in the kernel-compile
 * sandbox is not guaranteed — exactly the dependency the reused bp16_quire.h /
 * bp16_encode.h headers deliberately avoid by staying 32-bit-integer-only. This
 * header reproduces the SAME semantics __addsf3 would, in plain uint32 ops, so
 * the on-device fp32 running sum is genuine IEEE-754 fp32 (round-to-nearest,
 * ties-to-even) — the honest baseline the exact quire is being compared against.
 *
 * Correctness is gated host-side (bposit_quire_vs_float.cpp): for every value in
 * every test vector, fp32_add(...) is cross-checked bit-for-bit against the
 * host's native C `float` addition before the device run. A mismatch fails the
 * build-time self-check, so the on-device float baseline can NEVER be more
 * generous (or more pessimistic) than real fp32 — the float drift we report is
 * exactly what an IEEE fp32 accumulator does, not a contrivance.
 *
 * Scope: finite operands and the running-sum that the demo needs (the test
 * vectors are all finite). Inf/NaN propagation is implemented (so a sum that
 * overflows to Inf is reported honestly), but subnormal *inputs* are not part of
 * the demo vectors; subnormal *results* are produced correctly via the gradual-
 * underflow path below.
 */
#ifndef IEEE_SOFTFP32_H
#define IEEE_SOFTFP32_H

#include <cstdint>

/* fields of an IEEE-754 binary32 bit pattern */
#define SFP32_EXPMASK 0xFFu
#define SFP32_MANTMASK 0x007FFFFFu
#define SFP32_HIDDEN 0x00800000u  /* implicit leading 1 for normals */

static inline uint32_t sfp32_abs(uint32_t a) { return a & 0x7FFFFFFFu; }
static inline uint32_t sfp32_sign(uint32_t a) { return a >> 31; }
static inline uint32_t sfp32_exp(uint32_t a) { return (a >> 23) & SFP32_EXPMASK; }
static inline uint32_t sfp32_mant(uint32_t a) { return a & SFP32_MANTMASK; }

static inline int sfp32_is_nan(uint32_t a) {
    return (sfp32_exp(a) == 0xFFu) && (sfp32_mant(a) != 0u);
}
static inline int sfp32_is_inf(uint32_t a) {
    return (sfp32_exp(a) == 0xFFu) && (sfp32_mant(a) == 0u);
}

/* canonical quiet NaN */
#define SFP32_QNAN 0x7FC00000u

/* Round a 26-bit "extended" significand back to a 24-bit significand using the
 * guard/round/sticky in the low 2 bits + the sticky flag, ties-to-even. `sig`
 * holds the significand left-justified so that bit (shift+2) is the unit; here
 * we pass sig already in [unit<<2 | guard<<1 | round-or-sticky-collapsed].
 * To keep this simple and provably correct we instead implement add directly
 * below with an explicit 3-bit GRS, matching __addsf3. */

/* IEEE-754 binary32 add (round-to-nearest, ties-to-even). Pure integer. */
static uint32_t fp32_add(uint32_t a, uint32_t b) {
    /* NaN propagation */
    if (sfp32_is_nan(a) || sfp32_is_nan(b)) return SFP32_QNAN;

    /* Inf handling */
    if (sfp32_is_inf(a) || sfp32_is_inf(b)) {
        if (sfp32_is_inf(a) && sfp32_is_inf(b)) {
            if (a == b) return a;        /* same-sign Inf */
            return SFP32_QNAN;           /* Inf - Inf */
        }
        return sfp32_is_inf(a) ? a : b;
    }

    uint32_t sa = sfp32_sign(a), sb = sfp32_sign(b);
    int32_t ea = (int32_t)sfp32_exp(a), eb = (int32_t)sfp32_exp(b);
    uint32_t ma = sfp32_mant(a), mb = sfp32_mant(b);

    /* build 24-bit significands with the hidden bit; normals get the implicit 1,
     * subnormals (exp==0) have no hidden bit and an effective exponent of 1. */
    uint32_t siga = (ea == 0) ? ma : (ma | SFP32_HIDDEN);
    uint32_t sigb = (eb == 0) ? mb : (mb | SFP32_HIDDEN);
    if (ea == 0) ea = 1;
    if (eb == 0) eb = 1;

    /* handle +/-0 explicitly (both fields zero -> value is +/-0) */
    int a_zero = (sfp32_abs(a) == 0);
    int b_zero = (sfp32_abs(b) == 0);
    if (a_zero && b_zero) {
        /* -0 + -0 = -0; otherwise +0 (round-to-nearest) */
        return (sa && sb) ? 0x80000000u : 0x00000000u;
    }
    if (a_zero) return b;
    if (b_zero) return a;

    /* Align to the larger exponent. ALL of this stays in uint32_t (no 64-bit
     * variable shifts) so the kernel pulls in NO libgcc soft-64-bit helpers on
     * RV32 — the same discipline the reused bp16_*.h headers keep. A 24-bit
     * significand shifted up by 3 GRS bits is 27 bits; their same-sign sum is
     * <= 28 bits, comfortably inside 32. The shift `d` is a NATIVE rv32 32-bit
     * shift (>> on uint32_t), and we collapse all bits shifted past bit 0 into a
     * single sticky bit — exact round-to-nearest information, no wide field. */
    uint32_t fa = siga << 3;   /* 27-bit: significand in bits [26..3], GRS = [2..0] */
    uint32_t fb = sigb << 3;
    int32_t exp;
    if (ea >= eb) {
        exp = ea;
        int32_t d = ea - eb;
        if (d >= 32) {                       /* fb entirely below the sticky bit */
            if (fb) fb = 1; else fb = 0;     /* any nonzero -> pure sticky */
        } else if (d > 0) {
            uint32_t lost = fb & ((1u << d) - 1u);
            fb >>= d;
            if (lost) fb |= 1u;              /* sticky */
        }
    } else {
        exp = eb;
        int32_t d = eb - ea;
        if (d >= 32) {
            if (fa) fa = 1; else fa = 0;
        } else if (d > 0) {
            uint32_t lost = fa & ((1u << d) - 1u);
            fa >>= d;
            if (lost) fa |= 1u;
        }
    }

    uint32_t rsign;
    uint32_t f;
    if (sa == sb) {
        f = fa + fb;                          /* <= 28 bits, fits uint32 */
        rsign = sa;
    } else {
        if (fa >= fb) { f = fa - fb; rsign = sa; }
        else          { f = fb - fa; rsign = sb; }
        if (f == 0) return 0x00000000u;      /* exact cancellation -> +0 (RTNE) */
    }

    /* Normalize. The hidden-bit (unit) position is bit (23+3) = 26.
     * If addition overflowed into bit 27, shift right one (preserving sticky). */
    if (f & (1u << 27)) {
        uint32_t lost = f & 1u;
        f >>= 1;
        if (lost) f |= 1u;                   /* preserve sticky on the shift */
        exp += 1;
    }
    /* If subtraction left the unit below bit 26, shift left to renormalize. */
    while (!(f & (1u << 26)) && f != 0 && exp > 1) {
        f <<= 1;
        exp -= 1;
    }

    /* round-to-nearest, ties-to-even using the low 3 bits as guard/round/sticky.
     * keep = bits [26..3] (the 24-bit significand incl hidden bit). */
    uint32_t guard = (f >> 2) & 1u;
    uint32_t round = (f >> 1) & 1u;
    uint32_t stick = f & 1u;
    uint32_t rs = round | stick;
    uint32_t sig = f >> 3;                    /* 24-bit (+maybe carry) significand */

    if (guard && (rs || (sig & 1))) {
        sig += 1;
        if (sig & 0x01000000u) {             /* rounding carried into bit 24 */
            sig >>= 1;
            exp += 1;
        }
    }

    /* overflow to infinity */
    if (exp >= 0xFF) {
        return (rsign << 31) | 0x7F800000u;  /* +/-Inf */
    }

    /* subnormal / normal pack. If the hidden bit is set, exp>=1 is a normal. */
    if (sig & SFP32_HIDDEN) {
        uint32_t mant = sig & SFP32_MANTMASK;
        return (rsign << 31) | ((uint32_t)exp << 23) | mant;
    } else {
        /* result is subnormal (exp underflowed during normalize loop's guard
         * exp>1, so the only way here is a genuine subnormal): exp field 0. */
        uint32_t mant = sig & SFP32_MANTMASK;
        return (rsign << 31) | (0u << 23) | mant;
    }
}

#endif  /* IEEE_SOFTFP32_H */
