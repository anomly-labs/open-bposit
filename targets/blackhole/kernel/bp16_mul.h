/* Copyright (c) 2026 Anomly, Inc.
 * SPDX-License-Identifier: Apache-2.0 */
/* bp16_mul.h — exact bposit16 multiply, pure integer (RV32IM-clean), faithful to
 * the Apache-2.0 oracle bposit16_mul (decode → exact Fraction product → encode).
 *
 * Because every bposit16 value is dyadic (sign·M·2^E2 from bp16_decode), the
 * product is exact: sign_a⊕sign_b · (M_a·M_b)·2^(E2_a+E2_b), with M_a·M_b ≤ 2^22.
 * No 64-bit arithmetic, no division — the only encode rounding is the truncation
 * inside bp16_encode. This is the elementwise multiply behind two-input Layer-1
 * operators (cross-entropy / KL: term = p·log₂(q)) and bp16-precision Layer-2
 * tensor-network contractions.
 *
 * Special handling: inputs are assumed non-special (valid bp16 probabilities /
 * weights); a zero operand yields ZERO (matching the oracle). NaR is not
 * exercised on this path.
 */
#ifndef BP16_MUL_H
#define BP16_MUL_H

#include "bp16_decode.h"
#include "bp16_encode.h"

static int bposit16_mul(int a, int b) {
    int sa, sb, Ea, Eb; unsigned Ma, Mb;
    bp16_decode(a, &sa, &Ma, &Ea);
    bp16_decode(b, &sb, &Mb, &Eb);
    if (Ma == 0 || Mb == 0) return 0x0000;     /* zero operand -> ZERO */
    unsigned mag = Ma * Mb;                      /* <= 2^22, exact in 32 bits */
    int e2 = Ea + Eb;
    int sign = sa ^ sb;
    return bp16_encode_signed(sign, mag, e2);
}

#endif
