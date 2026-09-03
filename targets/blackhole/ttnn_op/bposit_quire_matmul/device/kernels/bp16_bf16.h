// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bp16_bf16.h — bf16 <-> bposit16-code conversion, pure integer (RV32IM-clean), for the
// ON-DEVICE codec that makes the exact-quire op host-round-trip-free / trace-compatible
// (task #18). bf16 normals are exactly dyadic, so the conversion is exact decomposition +
// the proven bp16_encode_signed / bp16_decode primitives. Reused by the op's reader (encode
// float operands) and writer (decode readout to float), and host-validated first (these
// headers are byte-identical x86 == qemu-rv32 == silicon).
#pragma once
#include "../../../../kernel/bp16_encode.h"  // bp16_encode_signed
#include "../../../../kernel/bp16_decode.h"  // bp16_decode

// bf16 bit pattern (uint16) -> bposit16 code (int, low 16 bits). Round-to-nearest-even.
static inline int bf16_to_bp16_code(unsigned u16) {
    unsigned sign = (u16 >> 15) & 1u;
    unsigned exp = (u16 >> 7) & 0xFFu;
    unsigned mant = u16 & 0x7Fu;
    if (exp == 0u) {
        if (mant == 0u) {
            return 0;  // +/-0 -> bposit16 zero code
        }
        return bp16_encode_signed_mode((int)sign, mant, -126 - 7, 1);  // subnormal: mant * 2^(-133)
    }
    if (exp == 0xFFu) {
        return 0x8000;  // inf/nan -> NaR
    }
    unsigned mag = 128u | mant;          // implicit 1 + 7 mantissa bits
    int e2 = (int)exp - 127 - 7;         // unbias and account for the 7-bit mantissa shift
    return bp16_encode_signed_mode((int)sign, mag, e2, 1);  // rtne
}

// bposit16 code (int) -> bf16 bit pattern (uint16). value = sign * M * 2^E2 -> normalize to bf16.
static inline unsigned bp16_code_to_bf16(int p) {
    int sign = 0;
    unsigned M = 0u;
    int E2 = 0;
    bp16_decode(p, &sign, &M, &E2);
    if (M == 0u) {
        return (unsigned)sign << 15;     // zero (NaR also decodes to M==0 here -> 0; acceptable)
    }
    int msb = 0;                          // highest set bit index of M
    unsigned t = M;
    while (t > 1u) {
        t >>= 1;
        msb++;
    }
    int exp_b = E2 + msb + 127;           // biased bf16 exponent
    unsigned mant7;
    if (msb >= 7) {
        mant7 = (M >> (unsigned)(msb - 7)) & 0x7Fu;
    } else {
        mant7 = (M << (unsigned)(7 - msb)) & 0x7Fu;
    }
    if (exp_b <= 0) {
        return (unsigned)sign << 15;      // underflow -> zero (simplification)
    }
    if (exp_b >= 0xFF) {
        return ((unsigned)sign << 15) | (0xFEu << 7) | 0x7Fu;  // overflow -> max finite
    }
    return ((unsigned)sign << 15) | ((unsigned)(exp_b & 0xFF) << 7) | mant7;
}
