/* Copyright (c) 2026 Anomly, Inc.
 * SPDX-License-Identifier: Apache-2.0 */
/* bp32_encode.h — int32 -> bposit32 encode (eS=3, useed=256, truncation),
 * faithful transcription of the oracle encode_bposit32. Pure integer; single
 * source of truth shared by the x86/rv32 kernel and the BRISC-on-ttsim
 * kernel. The pack-baby Stage-4 primitive. */
#ifndef BP32_ENCODE_H
#define BP32_ENCODE_H

typedef long long bp32_i64;
#define BP32_ES 3

static unsigned bp32_encode_unsigned_int(bp32_i64 value) {
    if (value == 0) return 0;
    int total_e = 0;
    bp32_i64 v = value;
    while (v >= 2) { v >>= 1; total_e++; }
    if (total_e > 48) return 0x7FFFFFFF;
    int k = total_e >> 3;
    int e = total_e & 7;
    bp32_i64 f_thr = (bp32_i64)1 << total_e;
    bp32_i64 f_num = value - f_thr;

    int bits[40]; int nb = 0;
    if (k >= 0) { for (int i = 0; i < k + 1; i++) bits[nb++] = 1; bits[nb++] = 0; }
    else        { for (int i = 0; i < -k; i++)    bits[nb++] = 0; bits[nb++] = 1; }

    if (nb < 31) {
        for (int i = BP32_ES - 1; i >= 0 && nb < 31; i--) bits[nb++] = (e >> i) & 1;
        while (nb < 32) {
            f_num <<= 1;
            if (f_num >= f_thr) { bits[nb++] = 1; f_num -= f_thr; }
            else                  bits[nb++] = 0;
        }
    }
    if (nb > 31) nb = 31;

    unsigned out = 0;
    for (int i = 0; i < nb; i++) out = (out << 1) | (unsigned)bits[i];
    out <<= (31 - nb);
    return out & 0x7FFFFFFF;
}

static unsigned bp32_encode_int(bp32_i64 value) {
    if (value == 0) return 0x00000000u;
    int sign = value < 0 ? 1 : 0;
    bp32_i64 mag = sign ? -value : value;
    unsigned field = bp32_encode_unsigned_int(mag);
    if (sign) field = ((~field) + 1) & 0x7FFFFFFF;
    return ((unsigned)sign << 31) | field;
}

/* ---- quire256 -> bposit32 (eS=3, useed=256, truncation) -------------------
 * The final-encode stage: the exact 256-bit Kulisch quire (signed
 * fixed-point, 96 fractional bits, 8x uint32 little-endian — q[0] = bits 0..31)
 * encoded to a bposit32 code. value = quire / 2^96. Faithful to the oracle's
 * quire256_to_bposit32 = encode_bposit32(Fraction(|q|, 2^96)) with sign. The
 * mantissa fraction bits are read straight out of the quire below its MSB, so
 * no 256-bit division/variable shift is needed (rv32-libgcc-free). */
#define BP32_QFRAC 96

static int bp32_q256_getbit(const unsigned q[8], int i) {
    return (int)((q[i >> 5] >> (i & 31)) & 1u);
}

static unsigned bp32_encode_quire256(const unsigned qin[8]) {
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
    if (orall == 0) return 0x00000000u;           /* zero */

    int msb = 255;
    while (msb >= 0 && !bp32_q256_getbit(q, msb)) msb--;
    int total_e = msb - BP32_QFRAC;               /* floor(log2(value)) */

    unsigned field;
    if (total_e > 48) {
        field = 0x7FFFFFFFu;                       /* saturate maxpos */
    } else if (total_e < -48) {
        field = 0x00000001u;                       /* saturate minpos */
    } else {
        int k = total_e >> 3;                      /* arithmetic floor /8 */
        int e = total_e & 7;
        int bits[40]; int nb = 0;
        if (k >= 0) { for (int i = 0; i < k + 1; i++) bits[nb++] = 1; bits[nb++] = 0; }
        else        { for (int i = 0; i < -k; i++)    bits[nb++] = 0; bits[nb++] = 1; }
        if (nb < 31) {
            for (int i = BP32_ES - 1; i >= 0 && nb < 31; i--) bits[nb++] = (e >> i) & 1;
            int fb = msb - 1;                      /* fraction = bits below msb */
            while (nb < 32) { bits[nb++] = (fb >= 0) ? bp32_q256_getbit(q, fb) : 0; fb--; }
        }
        if (nb > 31) nb = 31;
        unsigned out = 0;
        for (int i = 0; i < nb; i++) out = (out << 1) | (unsigned)bits[i];
        out <<= (31 - nb);
        field = out & 0x7FFFFFFFu;
    }
    if (sign) field = ((~field) + 1) & 0x7FFFFFFFu;
    return ((unsigned)sign << 31) | field;
}

#endif
