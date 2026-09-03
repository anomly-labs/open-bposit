/* Copyright (c) 2026 Anomly, Inc.
 * SPDX-License-Identifier: Apache-2.0 */
/* bp16_quire.h — bposit16 → exact 256-bit Kulisch quire accumulation, the
 * shared bandwidth-bound engine of every exact-quire reduction operator
 * (entropy / cross-entropy / KL, tensor-network contractions, curvature sums). Pure RV32IM
 * integer (32-bit shifts only — no libgcc), faithful to the oracle
 * bposit16_to_quire + quire256_add + quire256 negate (QUIRE_FRAC_BITS = 96).
 *
 * A bp16 value is dyadic (sign·M·2^E2 from bp16_decode); its quire contribution
 * is M placed at bit (E2+96), exact whenever that bit is in [0,255] — which it
 * always is for the small terms these operators produce. Same discipline as
 * quire256_reduce_kernel.c / the shannon kernels (which predate this header).
 */
#ifndef BP16_QUIRE_H
#define BP16_QUIRE_H

#include "bp16_decode.h"

#define BP16_QFRAC 96

/* Debug-only invariant guard. The dyadic placement is exact only while the term's
 * top significant bit index stays in [0,255]; past that the high word lands in the
 * (non-existent) word 8 and is silently dropped — the "exact" quire quietly wraps.
 * All operators here obey the invariant for the small terms they produce, so this is
 * a latent-misuse tripwire, not a live bug. Defining BP16Q_CHECK (host TB / debug
 * builds) makes a future out-of-range term fail loudly; the freestanding RV32 kernel
 * builds without it, so the guard is compiled out (no libc dependency, zero cost). */
#ifdef BP16Q_CHECK
#include <assert.h>
#define BP16Q_ASSERT(c) assert(c)
#else
#define BP16Q_ASSERT(c) ((void)0)
#endif

/* Two's complement of a 256-bit quire (8x uint32 LE) in place. */
static void q256_negate(unsigned q[8]) {
    unsigned long long carry = 1;
    for (int i = 0; i < 8; i++) { unsigned long long t = (unsigned long long)(~q[i]) + carry; q[i] = (unsigned)t; carry = t >> 32; }
}

/* bp16 code -> its 256-bit quire contribution (8x uint32 LE, two's complement). */
static void bp16_to_q256(int code, unsigned out[8]) {
    for (int i = 0; i < 8; i++) out[i] = 0;
    int sign; unsigned M; int E2;
    bp16_decode(code, &sign, &M, &E2);
    if (M == 0) return;                          /* zero / NaR -> contributes 0 */
    int shift = E2 + BP16_QFRAC;
    if (shift < 0) {                              /* below the binary point: truncate */
        if (-shift >= 11) return;
        M >>= -shift;
        if (M == 0u) return;
        shift = 0;
    }
    int wi = shift / 32, bit = shift % 32;
    if (wi >= 8) return;                          /* overflows the quire -> 0 */
    BP16Q_ASSERT(shift + 10 <= 255);              /* M < 2^11 -> top bit at shift+10 */
    out[wi] = M << bit;                           /* M < 2^11, spans <= 2 words */
    if (bit && wi + 1 < 8) out[wi + 1] = M >> (32 - bit);
    if (sign) q256_negate(out);                   /* two's complement over 256 bits */
}

/* EXACT bp16×bp16 product placed into the quire (no per-product rounding). The
 * product of two bp16 dyadics is exact: sign_a⊕sign_b · (M_a·M_b)·2^(E2_a+E2_b),
 * with M_a·M_b < 2^22 (fits the two-word placement). Using this instead of
 * bp16_to_q256(bposit16_mul(a,b)) keeps a contraction exact through the
 * accumulation — only the final quire→bp32 encode rounds. Lifts MERA / holographic
 * contractions on non-representable inputs from ~3 (per-product bp16) to ~8 digits. */
static void bp16_prod_to_q256(int acode, int bcode, unsigned out[8]) {
    for (int i = 0; i < 8; i++) out[i] = 0;
    int sa, sb, Ea, Eb; unsigned Ma, Mb;
    bp16_decode(acode, &sa, &Ma, &Ea);
    bp16_decode(bcode, &sb, &Mb, &Eb);
    if (Ma == 0u || Mb == 0u) return;            /* zero / NaR -> 0 */
    unsigned M = Ma * Mb;                          /* < 2^22 */
    int shift = Ea + Eb + BP16_QFRAC;
    if (shift < 0) {                               /* product LSB below the binary point:
                                                     bounded bp16 keeps 5 fraction bits at
                                                     k=-6, so E2 reaches -53 and Ea+Eb+96
                                                     reaches -10. Truncate toward zero like
                                                     the golden oracle instead of dropping
                                                     the whole product (minpos*minpos =
                                                     2^-96 exactly was returned as 0). */
        if (-shift >= 22) return;                  /* everything below 2^-QFRAC -> 0 */
        M >>= -shift;
        if (M == 0u) return;
        shift = 0;
    }
    int wi = shift / 32, bit = shift % 32;
    if (wi >= 8) return;
    BP16Q_ASSERT(shift + 21 <= 255);              /* M < 2^22 -> top bit at shift+21 */
    out[wi] = M << bit;                            /* M(22b)+bit(<=31) = <=53 < 64 -> 2 words */
    if (bit && wi + 1 < 8) out[wi + 1] = M >> (32 - bit);
    if (sa ^ sb) q256_negate(out);
}

static void q256_add(unsigned q[8], const unsigned x[8]) {
    unsigned long long carry = 0;
    for (int i = 0; i < 8; i++) { unsigned long long t = (unsigned long long)q[i] + x[i] + carry; q[i] = (unsigned)t; carry = t >> 32; }
}

/* q += (or -=, when negate) a 1-2 word term (hi:lo) placed at quire words wi:wi+1,
 * with a short carry/borrow ripple that stops as soon as it clears. The single
 * windowed-arithmetic routine shared by both fused MADDs — the one place to
 * bit-exact-check, so a one-sided edit can't silently break the equivalence the
 * two MADD entry points promise. */
static void q256_addsub_window(unsigned q[8], int wi, unsigned lo, unsigned hi, int negate) {
    if (!negate) {                                    /* q += hi:lo */
        unsigned long long t = (unsigned long long)q[wi] + lo;
        q[wi] = (unsigned)t;
        unsigned long long carry = t >> 32;
        for (int i = wi + 1; i < 8; i++) {
            unsigned add = (i == wi + 1) ? hi : 0u;
            if (!carry && add == 0u) break;
            t = (unsigned long long)q[i] + add + carry;
            q[i] = (unsigned)t; carry = t >> 32;
        }
    } else {                                          /* q -= hi:lo */
        unsigned long long t = (unsigned long long)q[wi] - lo;
        q[wi] = (unsigned)t;
        unsigned long long borrow = (t >> 32) & 1ull;
        for (int i = wi + 1; i < 8; i++) {
            unsigned sub = (i == wi + 1) ? hi : 0u;
            if (!borrow && sub == 0u) break;
            t = (unsigned long long)q[i] - sub - borrow;
            q[i] = (unsigned)t; borrow = (t >> 32) & 1ull;
        }
    }
}

/* Fused exact MADD: q += (signed) product of two bp16 codes, written DIRECTLY into the 1-2 quire
 * words the product touches (wi, wi+1) with a short carry/borrow ripple that stops as soon as it
 * clears. Bit-exact-equivalent to bp16_prod_to_q256()+q256_add() (subtract for opposite signs ==
 * adding the two's-complement product mod 2^256) but avoids the per-MAC 8-word zero + 8-word negate
 * + 8-word add — the matmul inner-loop hot path. */
static void bp16_madd_q256(unsigned q[8], int acode, int bcode) {
    int sa, sb, Ea, Eb; unsigned Ma, Mb;
    bp16_decode(acode, &sa, &Ma, &Ea);
    bp16_decode(bcode, &sb, &Mb, &Eb);
    if (Ma == 0u || Mb == 0u) return;                 /* zero / NaR -> 0 */
    unsigned M = Ma * Mb;                              /* < 2^22 */
    int shift = Ea + Eb + BP16_QFRAC;
    if (shift < 0) {                                  /* below the binary point: truncate toward zero (= oracle) */
        if (-shift >= 22) return;
        M >>= -shift;
        if (M == 0u) return;
        shift = 0;
    }
    int wi = shift / 32, bit = shift % 32;
    if (wi >= 8) return;                              /* above quire -> drop (matches scalar path) */
    BP16Q_ASSERT(shift + 21 <= 255);                  /* M < 2^22 -> top bit at shift+21 */
    unsigned lo = M << bit;                            /* product = hi:lo at words wi+1:wi */
    unsigned hi = bit ? (M >> (32 - bit)) : 0u;
    q256_addsub_window(q, wi, lo, hi, sa ^ sb);       /* same sign -> +, opposite -> - */
}

/* Packed bp16 operand (for LUT fast path): bit0 = sign, bits1..12 = significand M (<2^11),
 * bits13..31 = (E2 + 2048) biased exponent. bp16_pack(code) decodes once; bp16_madd_q256_packed
 * does the exact windowed MADD from two pre-decoded packed operands — lets a bf16->packed LUT
 * replace BOTH the per-element encode and the in-MADD decode (the inner-loop codec is the hot cost).
 * Bit-exact-equivalent to bp16_madd_q256: same M, shift, sign, same windowed add/sub. */
static unsigned bp16_pack(int code) {
    int sign; unsigned M; int E2;
    bp16_decode(code, &sign, &M, &E2);
    if (M == 0u) return 0u;                               /* zero / NaR -> 0 (madd checks M==0) */
    return ((unsigned)(E2 + 2048) << 13) | ((M & 0xFFFu) << 1) | (unsigned)(sign & 1);
}

static void bp16_madd_q256_packed(unsigned q[8], unsigned pa, unsigned pb) {
    unsigned Ma = (pa >> 1) & 0xFFFu;
    unsigned Mb = (pb >> 1) & 0xFFFu;
    if (Ma == 0u || Mb == 0u) return;
    unsigned M = Ma * Mb;                                 /* < 2^22 */
    int shift = (int)(pa >> 13) - 2048 + (int)(pb >> 13) - 2048 + BP16_QFRAC;
    if (shift < 0) {                                  /* below the binary point: truncate toward zero (= oracle) */
        if (-shift >= 22) return;
        M >>= -shift;
        if (M == 0u) return;
        shift = 0;
    }
    int wi = shift / 32, bit = shift % 32;
    if (wi >= 8) return;
    BP16Q_ASSERT(shift + 21 <= 255);                     /* M < 2^22 -> top bit at shift+21 */
    unsigned lo = M << bit;
    unsigned hi = bit ? (M >> (32 - bit)) : 0u;
    q256_addsub_window(q, wi, lo, hi, (pa ^ pb) & 1u);   /* sign bit differs -> subtract */
}

#endif
