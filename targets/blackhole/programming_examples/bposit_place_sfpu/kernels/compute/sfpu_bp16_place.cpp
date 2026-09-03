// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// sfpu_bp16_place.cpp — MILESTONE M3 of the SFPU-vectorized exact-quire b-posit matmul:
// lane-parallel WINDOWED PLACEMENT of a bp16xbp16 product into the 256-bit quire =
// vectorize the scalar bp16_prod_to_q256 (kernel/bp16_quire.h). Per lane, from
// two codes (a,b): decode (M1) -> product (M2) -> place M_p<<(E2_p+QFRAC) into the 8-limb
// (8x u32 LE) quire contribution. Now M3b (full SIGNED): the positive placement (M3a, the
// hard variable cross-limb shift) PLUS the conditional two's-complement when sign_p — done
// as a SEPARATE 1-LIMB-per-call neg op (4 small faces/lease) so each face stays under the
// SFPU reload budget (the 4-limb combined/separate variants ICE'd). Matches the FULL scalar
// bp16_prod_to_q256 (incl. its sign negate), 32/32 bit-exact across same/opposite signs.
//
// Scalar spec matched (positive part):
//   M=Ma*Mb (<2^22); shift=Ea+Eb+QFRAC(=96); if M==0||shift<0||(shift>>5)>=8 -> all-zero
//   wi=shift>>5, bit=shift&31; lo=M<<bit; hi = bit? M>>(32-bit) : 0
//   out[wi]=lo, out[wi+1]=hi (wi+1<8), other limbs 0
//
// dst budget (8, SyncFull/accum): 8 output limbs + inputs can't co-reside, so produce the
// 8 limbs in TWO leases of 4 (limbs 0..3 then 4..7). Instead of a handoff CB we RECOMPUTE
// decode+product in each lease (cheap, codes stay in cb_a/cb_b) -> peak live dst = sign_p,
// M_p, E2_p (0,1,2) + 4 output limbs (4,5,6,7) = 7 <= 8.
//
// SFPU lessons applied (q256_add + M1 + M2): no magnitude compare / no large immediate ->
// bit algebra; split nested exprs (tiny LREG file -> reload ICE); void kernel_main();
// assign dst_reg[] to vUInt before bit ops; INT32 -> host fp32_dest_acc_en + dst_full_sync.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

#define SLOT_SP   0   // sign_p
#define SLOT_MP   1   // M_p
#define SLOT_EP   2   // E2_p
#define SLOT_SB   3
#define SLOT_MB   4
#define SLOT_EB   5
#define SLOT_CODE 6
#define SLOT_O0   4   // 4 output limbs reuse slots 4,5,6,7 (SB/MB/EB free after product)

#define BP16_QFRAC 96

#ifdef TRISC_MATH
using namespace sfpi;

// --- M1 decode face (verbatim) ----------------------------------------------
inline void bp16_decode_face(
    const uint32_t dst_code, const uint32_t dst_sign, const uint32_t dst_M, const uint32_t dst_E2) {
    constexpr uint32_t NV = 32;
    const uint32_t cb = dst_code * NV, sb = dst_sign * NV, mb = dst_M * NV, eb = dst_E2 * NV;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt p = dst_reg[cb + i];
        p = p & 0xFFFF;
        vUInt sign = p >> 15;
        vUInt rest = p & 0x7FFF;
        vUInt t;
        v_if (sign != 0) { t = ~rest; t = t + 1; rest = t & 0x7FFF; } v_endif;
        vUInt lead = rest >> 14;
        vUInt norm = rest;
        v_if (lead != 0) { t = ~rest; norm = t & 0x7FFF; } v_endif;
        vUInt v = norm;
        vUInt b = 0;
        t = v >> 8;  v_if (t != 0) { b = b + 8; v = v >> 8; } v_endif;
        t = v >> 4;  v_if (t != 0) { b = b + 4; v = v >> 4; } v_endif;
        t = v >> 2;  v_if (t != 0) { b = b + 2; v = v >> 2; } v_endif;
        t = v >> 1;  v_if (t != 0) { b = b + 1;             } v_endif;
        vUInt rs = vUInt(14) - b;
        v_if (norm == 0) { rs = 15; } v_endif;
        vUInt k = rs - 1;
        v_if (lead == 0) { k = vUInt(0) - rs; } v_endif;
        vUInt remaining = vUInt(14) - rs;
        v_if (rs == 15) { remaining = 0; } v_endif;
        t = vUInt(1) << remaining; t = t - 1;
        vUInt rest2 = rest & t;
        vUInt e_width = remaining;
        t = remaining >> 2; v_if (t != 0) { e_width = 3; } v_endif;
        vUInt fw = remaining - e_width;
        vUInt e = rest2 >> fw;
        t = vUInt(1) << e_width; t = t - 1;
        e = e & t;
        t = vUInt(3) - e_width;
        e = e << t;
        t = vUInt(1) << fw; t = t - 1;
        vUInt M = rest2 & t;
        t = vUInt(1) << fw;
        M = M + t;
        vUInt E2 = k << 3;
        E2 = E2 + e;
        E2 = E2 - fw;
        v_if (rest == 0) { sign = 0; M = 0; E2 = 0; } v_endif;
        dst_reg[sb + i] = sign;
        dst_reg[mb + i] = M;
        dst_reg[eb + i] = E2;
    }
}

// --- M2 product face (verbatim) ---------------------------------------------
inline void bp16_product_face(
    const uint32_t dst_sa, const uint32_t dst_Ma, const uint32_t dst_Ea,
    const uint32_t dst_sb, const uint32_t dst_Mb, const uint32_t dst_Eb) {
    constexpr uint32_t NV = 32;
    const uint32_t sa = dst_sa * NV, ma = dst_Ma * NV, ea = dst_Ea * NV;
    const uint32_t sb = dst_sb * NV, mb = dst_Mb * NV, eb = dst_Eb * NV;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt sgn = dst_reg[sa + i];
        vUInt tb = dst_reg[sb + i];
        sgn = sgn ^ tb;
        vUInt e2 = dst_reg[ea + i];
        tb = dst_reg[eb + i];
        e2 = e2 + tb;
        vUInt prod = 0;
        vUInt a = dst_reg[ma + i];
        vUInt m = dst_reg[mb + i];
        vUInt t;
        for (uint32_t s = 0; s < 11; s++) {
            t = m & 1;
            v_if (t != 0) { prod = prod + a; } v_endif;
            a = a << 1;
            m = m >> 1;
        }
        dst_reg[sa + i] = sgn;
        dst_reg[ma + i] = prod;
        dst_reg[ea + i] = e2;
    }
}

// --- M3 place face: scatter the product (M_p,E2_p) into 4 limbs [base..base+3] ----
// Writes 4 output tiles at dst_o0..dst_o0+3 (= quire limbs base..base+3). POSITIVE only.
inline void bp16_place_face(
    const uint32_t dst_mp, const uint32_t dst_ep, const uint32_t dst_o0, const uint32_t base) {
    constexpr uint32_t NV = 32;
    const uint32_t mpb = dst_mp * NV, epb = dst_ep * NV, ob = dst_o0 * NV;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt Mp = dst_reg[mpb + i];
        vUInt E2 = dst_reg[epb + i];
        vUInt shift = E2 + BP16_QFRAC;   // E2 two's-complement; shift may be "negative" (bit31 set)

        // valid = (Mp != 0) && (shift >= 0) && (shift < 256)
        vUInt valid = 1;
        vUInt t = shift >> 31;  v_if (t != 0) { valid = 0; } v_endif;   // bit31 set -> negative
        t = shift >> 8;         v_if (t != 0) { valid = 0; } v_endif;   // >= 256 -> wi >= 8
        v_if (Mp == 0)          { valid = 0; } v_endif;

        vUInt wi = shift >> 5;
        vUInt bit = shift & 31;
        vUInt lo = Mp << bit;
        vUInt hi = 0;
        v_if (bit != 0) { t = vUInt(32) - bit; hi = Mp >> t; } v_endif;  // guard bit==0 (M>>32 is UB)
        v_if (valid == 0) { lo = 0; hi = 0; } v_endif;                   // invalid -> contributes nothing

        // scatter: out[base+k] = (wi==base+k ? lo:0) | (wi+1==base+k ? hi:0)
        for (uint32_t k = 0; k < 4; k++) {
            const uint32_t j = base + k;
            vUInt outj = 0;
            v_if (wi == j) { outj = lo; } v_endif;
            if (j >= 1) { v_if (wi == (j - 1)) { outj = outj | hi; } v_endif; }  // wi+1==j
            dst_reg[(dst_o0 + k) * NV + i] = outj;
        }
    }
}
// --- M3b negate, ONE LIMB per call (small face -> stays under the SFPU reload budget) ----
// When sign_p, overwrites the positive value in dst_slot (= quire limb j) with the
// two's-complement closed form. Called 4x per lease (j = base..base+3). NO cross-limb carry
// (value nonzero only at wi/wi+1): j<wi->0, j==wi->(0-lo), j==wi+1->(0-hi-(lo!=0)), j>wi+1->-1.
inline void bp16_neg1_face(
    const uint32_t dst_mp, const uint32_t dst_ep, const uint32_t dst_sp, const uint32_t dst_slot,
    const uint32_t j) {
    constexpr uint32_t NV = 32;
    const uint32_t mpb = dst_mp * NV, epb = dst_ep * NV, spb = dst_sp * NV, ob = dst_slot * NV;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt Mp = dst_reg[mpb + i];
        vUInt E2 = dst_reg[epb + i];
        vUInt sp = dst_reg[spb + i];
        vUInt shift = E2 + BP16_QFRAC;
        vUInt valid = 1;
        vUInt t = shift >> 31;  v_if (t != 0) { valid = 0; } v_endif;
        t = shift >> 8;         v_if (t != 0) { valid = 0; } v_endif;
        v_if (Mp == 0)          { valid = 0; } v_endif;
        vUInt wi = shift >> 5;
        vUInt bit = shift & 31;
        vUInt lo = Mp << bit;
        vUInt hi = 0;
        v_if (bit != 0) { t = vUInt(32) - bit; hi = Mp >> t; } v_endif;
        v_if (valid == 0) { lo = 0; hi = 0; } v_endif;

        vUInt cur = dst_reg[ob + i];              // positive value from the place op
        vUInt allones = vUInt(0) - 1;
        vUInt d = wi - vUInt(j);
        vUInt sj = 0;
        vUInt isneg = d >> 31;
        v_if (isneg != 0) { sj = allones; } v_endif;       // wi < j
        v_if (d == allones) { vUInt nhi = vUInt(0) - hi; v_if (lo != 0) { nhi = nhi - 1; } v_endif; sj = nhi; } v_endif;  // wi==j-1
        v_if (d == 0) { sj = vUInt(0) - lo; } v_endif;     // wi == j

        vUInt outj = cur;
        v_if (sp != 0) { outj = sj; } v_endif;
        v_if (valid == 0) { outj = 0; } v_endif;
        dst_reg[ob + i] = outj;
    }
}
#endif  // TRISC_MATH

inline void bp16_decode_tile(uint32_t dc, uint32_t ds, uint32_t dm, uint32_t de) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(bp16_decode_face, dc, ds, dm, VectorMode::RC, de));
}
inline void bp16_product_tile(uint32_t sa, uint32_t ma, uint32_t ea, uint32_t sb, uint32_t mb, uint32_t eb) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(bp16_product_face, sa, ma, ea, VectorMode::RC, sb, mb, eb));
}
inline void bp16_place_tile(uint32_t dmp, uint32_t dep, uint32_t do0, uint32_t base) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(bp16_place_face, dmp, dep, do0, VectorMode::RC, base));
}
inline void bp16_neg1_tile(uint32_t dmp, uint32_t dep, uint32_t dsp, uint32_t dslot, uint32_t j) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(bp16_neg1_face, dmp, dep, dsp, VectorMode::RC, dslot, j));
}

// decode A + decode B + product -> sign_p(SLOT_SP), M_p(SLOT_MP), E2_p(SLOT_EP).
inline void decode_and_product(uint32_t cb_a, uint32_t cb_b) {
    copy_tile_to_dst_init_short(cb_a);
    copy_tile(cb_a, 0, SLOT_CODE);
    bp16_decode_tile(SLOT_CODE, SLOT_SP, SLOT_MP, SLOT_EP);
    copy_tile_to_dst_init_short(cb_b);
    copy_tile(cb_b, 0, SLOT_CODE);
    bp16_decode_tile(SLOT_CODE, SLOT_SB, SLOT_MB, SLOT_EB);
    bp16_product_tile(SLOT_SP, SLOT_MP, SLOT_EP, SLOT_SB, SLOT_MB, SLOT_EB);
}

void kernel_main() {
    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_a, cb_out);
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);

    // Two leases of 4 limbs each (recompute decode+product per lease; codes stay resident).
    for (uint32_t g = 0; g < 2; ++g) {
        const uint32_t base = g * 4;
        tile_regs_acquire();
        decode_and_product(cb_a, cb_b);
        bp16_place_tile(SLOT_MP, SLOT_EP, SLOT_O0, base);   // positive limbs -> slots SLOT_O0..+3
        // M3b: negate one limb at a time (small face) when sign_p
        bp16_neg1_tile(SLOT_MP, SLOT_EP, SLOT_SP, SLOT_O0 + 0, base + 0);
        bp16_neg1_tile(SLOT_MP, SLOT_EP, SLOT_SP, SLOT_O0 + 1, base + 1);
        bp16_neg1_tile(SLOT_MP, SLOT_EP, SLOT_SP, SLOT_O0 + 2, base + 2);
        bp16_neg1_tile(SLOT_MP, SLOT_EP, SLOT_SP, SLOT_O0 + 3, base + 3);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 4);
        pack_tile(SLOT_O0 + 0, cb_out, 0);
        pack_tile(SLOT_O0 + 1, cb_out, 1);
        pack_tile(SLOT_O0 + 2, cb_out, 2);
        pack_tile(SLOT_O0 + 3, cb_out, 3);
        cb_push_back(cb_out, 4);
        tile_regs_release();
    }

    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}
