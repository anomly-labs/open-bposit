/* Copyright (c) 2026 Anomly, Inc.
 * SPDX-License-Identifier: Apache-2.0 */
/* cross_entropy_golden_cases.h - baked golden for the Layer-1
 * cross-entropy H(p,q) = -Sum_i p_i*log2(q_i) over bposit16 distributions,
 * the standard classification training loss. log2(q_i) is HOST-precomputed
 * (bposit16_log2; QGOLD_CE_LOG2Q below) because the 65536-entry log2 LUT does
 * not fit the BRISC baby core's local DATA region. Each term p_i*log2(q_i) is
 * then an EXACT bp16 product accumulated ON-DEVICE in the EXACT 256-bit
 * b-posit16 quire (QUIRE_FRAC_BITS=96) and negated; the single rounding is the
 * final quire->bposit32 readout (H=2.625 -> 0x45400000).
 *
 * DO NOT hand-edit the numbers - GENERATED from golden/cross_entropy.json and
 * the canonical oracle, re-verified by the host program at startup against the
 * reused bp16_mul.h / bp16_quire.h / bp32_encode.h headers + bposit16_log2
 * (belt-and-suspenders).
 *
 * SOURCE OF TRUTH (regenerate, never by hand):
 *   golden: golden/cross_entropy.json (p_codes, q_codes, quire_le_hex, bp32_code, n)
 *   oracle: mosyne-bposit/kernels/bposit16_reference.py
 *           (bposit16_log2, bposit16_mul, bposit16_to_quire,
 *            quire256_to_bposit32, decode_bposit32, decoded_to_fraction_32)
 *   kernel: kernel/cross_entropy_kernel.c (the proven RV32IM exact-quire reference)
 *   log2  : bposit16_log2 host-precomputed == bp16_log2_lut.h BP16_LOG2_LUT[q]
 *   gen   : gen_cross_entropy_golden.py (kept with this commit)
 *
 * Layout: target p[N], predicted q[N], and host-precomputed log2q[N] (bposit16).
 * The kernel consumes p[] and log2q[] (pure exact-quire on-device), emits the
 * 32-byte LE negated quire (the primary gate) then the bposit32 readout; the
 * host gates BOTH byte-/code-identical vs golden.
 *
 * Verified locally (gcc/python, reusing the EXACT oracle the kernel C headers
 * mirror - bposit16_log2 + bposit16_mul + bposit16_to_quire + quire256_to_bposit32):
 *   - the negated 256-bit quire == golden/cross_entropy.json quire_le_hex
 *   - the bposit32 readout == golden/cross_entropy.json bp32_code (H=2.625)
 */
#ifndef CROSS_ENTROPY_GOLDEN_CASES_H
#define CROSS_ENTROPY_GOLDEN_CASES_H

#include <cstdint>

#define QGOLD_CE_N 4

/* target distribution p[i] and predicted distribution q[i] (bposit16). */
static const uint16_t QGOLD_CE_P[4] = {
    0x3c00, 0x3800, 0x3400, 0x3400,
};

static const uint16_t QGOLD_CE_Q[4] = {
    0x3400, 0x3400, 0x3800, 0x3c00,
};

/* HOST-precomputed per-element log2(q_i) = bposit16_log2(q_i) (bposit16). This
 * is the small vector handed to the device IN PLACE of an on-chip log2 LUT; the
 * device kernel is then pure exact-quire (term = bp16_mul(p_i, log2q_i)). By
 * construction QGOLD_CE_LOG2Q[i] == BP16_LOG2_LUT[q_i]. */
static const uint16_t QGOLD_CE_LOG2Q[4] = {
    0xba00, 0xba00, 0xbc00, 0xc000,
};

/* expected per-element term code term_i = bposit16_mul(p_i, log2q_i) (bp16),
 * baked for an optional per-element host self-check (not gated on device). */
static const uint16_t QGOLD_CE_TERM[4] = {
    0xbe00, 0xc200, 0xc800, 0xcc00,
};

/* expected cross-entropy as the final bposit32 readout of the negated quire.
 * H(p,q) = 2.625 -> code below. From golden/cross_entropy.json bp32_code. */
#define QGOLD_CE_BP32 0x45400000u

/* the FULL expected 32-byte little-endian NEGATED quire (the kernel's primary
 * target): H(p,q) accumulated exactly then negated. From golden quire_le_hex. */
static const uint32_t QGOLD_CE_QUIRE[8] = {
    0x00000000u, 0x00000000u, 0xa0000000u, 0x00000002u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
};
static const char QGOLD_CE_QUIRE_HEX[] =
    "0000000000000000000000a00200000000000000000000000000000000000000";

#endif /* CROSS_ENTROPY_GOLDEN_CASES_H */
