/* Copyright (c) 2026 Anomly, Inc.
 * SPDX-License-Identifier: Apache-2.0 */
/* bp16_encode.h — bposit16 encode of a dyadic value sign·mag·2^e2, pure integer
 * (RV32IM-clean), faithful to the Apache-2.0 oracle encode_bposit16 +
 * _encode_unsigned (ES=3, useed=256, TRUNCATION, max |total_e| = 48).
 *
 * Encoding from the (mag, e2) dyadic form (rather than a Fraction) is what an
 * exact bposit16 multiply needs: the product of two bp16 values is
 * sign·(M_a·M_b)·2^(E2_a+E2_b), with M_a·M_b ≤ 2^22 (fits 32 bits). The mantissa
 * fraction bits are read straight out of `mag` below its MSB, so there is no
 * division or 64-bit variable shift (no libgcc on rv32). Single source of truth
 * shared by the x86/qemu-rv32 kernel and the BRISC-on-ttsim kernel.
 */
#ifndef BP16_ENCODE_H
#define BP16_ENCODE_H

#ifndef BP16_ES
#define BP16_ES 3
#endif

/* encode positive mag·2^e2 (mag > 0) into the 15-bit unsigned bposit16 field. */
static unsigned bp16_encode_mag_e2_mode(unsigned mag, int e2, int rtne) {
    if (mag == 0) return 0;
    int msbm = 31;
    while (msbm >= 0 && !((mag >> msbm) & 1u)) msbm--;
    int total_e = e2 + msbm;                  /* floor(log2(mag·2^e2)) */
    if (total_e > 48) return 0x7FFF;          /* maxpos */
    if (total_e < -48) return 0x0001;         /* minpos */

    int k = total_e >> 3;                      /* arithmetic floor /8 */
    int e = total_e & 7;
    int bits[24]; int nb = 0;
    if (k >= 0) { for (int i = 0; i < k + 1; i++) bits[nb++] = 1; bits[nb++] = 0; }
    else        { for (int i = 0; i < -k; i++)    bits[nb++] = 0; bits[nb++] = 1; }
    int round_bit = 0, sticky = 0, did_frac = 0;
    if (nb < 15) {
        for (int i = BP16_ES - 1; i >= 0 && nb < 15; i--) bits[nb++] = (e >> i) & 1;
        int fb = msbm - 1;                     /* fraction = bits below the MSB */
        while (nb < 16) { bits[nb++] = (fb >= 0) ? (int)((mag >> fb) & 1u) : 0; fb--; }
        round_bit = bits[15];                  /* the dropped guard bit */
        did_frac = 1;                          /* nb reached 16: a round bit exists */
        if (fb >= 0) sticky = (mag & (((unsigned)1 << (fb + 1)) - 1u)) != 0;
    }
    if (nb > 15) nb = 15;
    unsigned out = 0;
    for (int i = 0; i < nb; i++) out = (out << 1) | (unsigned)bits[i];
    out <<= (15 - nb);
    out &= 0x7FFF;
    if (rtne && did_frac && round_bit && (sticky || (out & 1u))) {
        if (out < 0x7FFF) out += 1;            /* round up, ties-to-even; saturate at maxpos */
    }
    return out;
}

/* encode positive mag·2^e2 (mag > 0) into the 15-bit unsigned bposit16 field.
 * `rtne` selects the rounding policy at the 15-bit boundary:
 *   0 — TRUNCATION: drop bits past position 14 (the default; matches the baked
 *       CUDA LUTs and every existing golden, so callers using bp16_encode_mag_e2
 *       are byte-for-byte unchanged).
 *   1 — round-to-nearest, ties-to-even (Gustafson posit spec): round up iff the
 *       guard bit is 1 AND (sticky residue OR the kept LSB is odd). Halves the
 *       worst-case round-trip error (docs/fast_emulation_and_rtne_2026-06-06.md).
 * Faithful to the oracle _encode_unsigned(mode=...). */
static unsigned bp16_encode_mag_e2(unsigned mag, int e2) {
    return bp16_encode_mag_e2_mode(mag, e2, 0);
}

/* encode signed dyadic value sign·mag·2^e2 into a full 16-bit bposit16 code.
 * `rtne` forwards to bp16_encode_mag_e2_mode (0 = truncate, 1 = ties-to-even). */
static int bp16_encode_signed_mode(int sign, unsigned mag, int e2, int rtne) {
    if (mag == 0) return 0x0000;               /* ZERO */
    unsigned field = bp16_encode_mag_e2_mode(mag, e2, rtne);
    if (sign) field = ((~field) + 1) & 0x7FFF;  /* 2's complement of 15-bit field */
    return (int)((((unsigned)sign << 15) | field) & 0xFFFF);
}

static int bp16_encode_signed(int sign, unsigned mag, int e2) {
    return bp16_encode_signed_mode(sign, mag, e2, 0);
}

/* ---- quire256 -> bposit16 (eS=3, useed=256, truncation) -------------------
 * The bp16-output analog of bp32_encode_quire256 (bp32_encode.h): rounds an
 * exact 256-bit Kulisch quire (signed fixed-point, 96 frac bits, 8x uint32 LE)
 * back to a bposit16 code. Faithful to the oracle quire256_to_bposit16 =
 * encode_bposit16(Fraction(|q|, 2^96)) with sign. Mantissa bits are read straight
 * out of the quire below its MSB — no division / variable shift. This is the
 * bp16-output stage for exact-quire reductions that stay at bp16 precision
 * (e.g. bposit16_add, Regge-action sums). */
#define BP16_QUIRE_FRAC 96

static int bp16_q256_getbit(const unsigned q[8], int i) {
    return (int)((q[i >> 5] >> (i & 31)) & 1u);
}

static int bp16_encode_quire256(const unsigned qin[8]) {
    int sign = (int)((qin[7] >> 31) & 1u);
    unsigned q[8];
    if (sign) {                                   /* magnitude = -qin (256-bit) */
        unsigned long long carry = 1;
        for (int i = 0; i < 8; i++) { unsigned long long t = (unsigned long long)(~qin[i]) + carry; q[i] = (unsigned)t; carry = t >> 32; }
    } else {
        for (int i = 0; i < 8; i++) q[i] = qin[i];
    }
    unsigned orall = 0;
    for (int i = 0; i < 8; i++) orall |= q[i];
    if (orall == 0) return 0x0000;                /* zero */

    int msb = 255;
    while (msb >= 0 && !bp16_q256_getbit(q, msb)) msb--;
    int total_e = msb - BP16_QUIRE_FRAC;          /* floor(log2(value)) */

    unsigned field;
    if (total_e > 48) {
        field = 0x7FFF;
    } else if (total_e < -48) {
        field = 0x0001;
    } else {
        int k = total_e >> 3;
        int e = total_e & 7;
        int bits[24]; int nb = 0;
        if (k >= 0) { for (int i = 0; i < k + 1; i++) bits[nb++] = 1; bits[nb++] = 0; }
        else        { for (int i = 0; i < -k; i++)    bits[nb++] = 0; bits[nb++] = 1; }
        if (nb < 15) {
            for (int i = BP16_ES - 1; i >= 0 && nb < 15; i--) bits[nb++] = (e >> i) & 1;
            int fb = msb - 1;
            while (nb < 16) { bits[nb++] = (fb >= 0) ? bp16_q256_getbit(q, fb) : 0; fb--; }
        }
        if (nb > 15) nb = 15;
        unsigned out = 0;
        for (int i = 0; i < nb; i++) out = (out << 1) | (unsigned)bits[i];
        out <<= (15 - nb);
        field = out & 0x7FFF;
    }
    if (sign) field = ((~field) + 1) & 0x7FFF;
    return (int)((((unsigned)sign << 15) | field) & 0xFFFF);
}

#endif
