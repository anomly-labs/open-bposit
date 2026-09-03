/* Copyright (c) 2026 Anomly, Inc.
 * SPDX-License-Identifier: Apache-2.0 */
/* bp16_decode.h — the bposit-16 decode primitive, pure integer (RV32IM-clean).
 *
 * Single source of truth shared by every build target so the SAME code is
 * proven to run identically on x86, qemu-rv32, and a BRISC baby core inside
 * ttsim's golden model. Faithful transcription of the Apache-2.0 oracle
 * decode_bposit16 + decoded_to_fraction (ES=3, useed=256=2^8).
 *
 * bp16 values are dyadic: value = (sign?-1:1) * M * 2^E2 with M = 2^fw + f_bits
 * (so 1 <= M < 2^(fw+1) <= 2^11, fits 32 bits) and E2 = 8*k + e - fw. We return
 * (sign, M, E2) rather than num/den because bp16 magnitudes reach 2^48..2^112,
 * which would overflow a 32-bit num/den and force 64-bit variable shifts
 * (__ashldi3 — the rv32 toolchain has no multilib libgcc for it). The (sign,M,E2)
 * form keeps M small and pushes the only large shift to the exact-quire
 * placement, which is done with 32-bit shifts in the consumer. M == 0 signals a
 * special (zero or NaR), which both contribute 0 to the quire.
 */
#ifndef BP16_DECODE_H
#define BP16_DECODE_H

#define BP16_ES   3
#define BP16_ZERO 0x0000
#define BP16_NAR  0x8000

static void bp16_decode(int p, int *sign_out, unsigned *M_out, int *E2_out) {
    p &= 0xFFFF;
    *sign_out = 0; *M_out = 0; *E2_out = 0;
    if (p == BP16_ZERO || p == BP16_NAR) return;     /* special -> M=0 */

    int sign = (p >> 15) & 1;
    int rest = p & 0x7FFF;
    if (sign) rest = ((~rest) + 1) & 0x7FFF;          /* 2's complement of 15-bit */

    int leading_bit = (rest >> 14) & 1;
    int rs = 0;
    while (rs < 15 && ((rest >> (14 - rs)) & 1) == leading_bit) rs++;

    int k, e = 0, f_bits = 0, f_width = 0;
    if (rs == 15) {
        k = leading_bit ? 14 : -15;                   /* regime saturates the field */
    } else {
        k = leading_bit ? (rs - 1) : -rs;
        int consumed = rs + 1;
        int remaining = 15 - consumed;
        int rest2 = rest & ((1 << remaining) - 1);
        int e_width = BP16_ES < remaining ? BP16_ES : remaining;
        if (e_width > 0) {
            e = (rest2 >> (remaining - e_width)) & ((1 << e_width) - 1);
            e <<= (BP16_ES - e_width);
        }
        remaining -= e_width;
        f_width = remaining;
        f_bits = (f_width > 0) ? (rest2 & ((1 << f_width) - 1)) : 0;
    }

    unsigned M = (f_width > 0) ? ((1u << f_width) + (unsigned)f_bits) : 1u;
    int E2 = 8 * k + e - (f_width > 0 ? f_width : 0);
    *sign_out = sign; *M_out = M; *E2_out = E2;
}

#endif
