// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// sfpu_bp16_dot.cpp — MILESTONE M3.5 of the SFPU-vectorized exact-quire b-posit matmul:
// one accumulation step of an exact bp16 DOT PRODUCT. Per dispatch (pair k), per lane:
//   C_k[8] = signed 256-bit quire contribution of a_k*b_k  (M3b: decode+product+placement)
//   q[8]   = q[8] + C_k[8]                                  (the SOLVED q256_add carry)
// The host loops K dispatches with q persisted in DRAM (zero-init) -> q accumulates the
// exact dot product sum_k a_k*b_k, bit-identical to the scalar bp16_madd_q256 chain.
//
// Two phases per dispatch, each built from PROVEN small faces (keep faces under the SFPU
// reload budget): phase 1 reuses M1/M2/M3a/M3b faces verbatim -> packs C to cb_C; phase 2
// reuses the sfpu_q256_add carry (compare-free, 4+4 leases + carry-mid CB) -> packs q to cb_qout.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

// phase-1 (place+neg) dst slots
#define SLOT_SP   0
#define SLOT_MP   1
#define SLOT_EP   2
#define SLOT_SB   3
#define SLOT_MB   4
#define SLOT_EB   5
#define SLOT_CODE 6
#define SLOT_O0   4
// phase-2 (q256_add) dst slots
#define QSLOT_Q     0
#define QSLOT_X     1
#define QSLOT_CARRY 2
#define QSLOT_OUT0  3   // OUT 3,4,5,6
#define QADD_BATCH  4

#define BP16_QFRAC 96

#ifdef TRISC_MATH
using namespace sfpi;

// ===== M1 decode + M2 product + M3a place + M3b neg1 faces (verbatim from bposit_place) =====
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
inline void bp16_place_face(
    const uint32_t dst_mp, const uint32_t dst_ep, const uint32_t dst_o0, const uint32_t base) {
    constexpr uint32_t NV = 32;
    const uint32_t mpb = dst_mp * NV, epb = dst_ep * NV;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt Mp = dst_reg[mpb + i];
        vUInt E2 = dst_reg[epb + i];
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
        for (uint32_t k = 0; k < 4; k++) {
            const uint32_t j = base + k;
            vUInt outj = 0;
            v_if (wi == j) { outj = lo; } v_endif;
            if (j >= 1) { v_if (wi == (j - 1)) { outj = outj | hi; } v_endif; }
            dst_reg[(dst_o0 + k) * NV + i] = outj;
        }
    }
}
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
        vUInt cur = dst_reg[ob + i];
        vUInt allones = vUInt(0) - 1;
        vUInt d = wi - vUInt(j);
        vUInt sj = 0;
        vUInt isneg = d >> 31;
        v_if (isneg != 0) { sj = allones; } v_endif;
        v_if (d == allones) { vUInt nhi = vUInt(0) - hi; v_if (lo != 0) { nhi = nhi - 1; } v_endif; sj = nhi; } v_endif;
        v_if (d == 0) { sj = vUInt(0) - lo; } v_endif;
        vUInt outj = cur;
        v_if (sp != 0) { outj = sj; } v_endif;
        v_if (valid == 0) { outj = 0; } v_endif;
        dst_reg[ob + i] = outj;
    }
}

// ===== q256_add carry face (verbatim from sfpu_q256_add.cpp: compare-free MSB carry) =====
inline void q256_add_limb_face(
    const uint32_t dst_q, const uint32_t dst_x, const uint32_t dst_out, const uint32_t dst_carry) {
    constexpr uint32_t NV = 32;
    const uint32_t qb = dst_q * NV, xb = dst_x * NV, ob = dst_out * NV, cb = dst_carry * NV;
    for (uint32_t i = 0; i < 8; i++) {
        vUInt q = dst_reg[qb + i];
        vUInt x = dst_reg[xb + i];
        vUInt carry_in = dst_reg[cb + i];
        vUInt s0 = q + x;
        vUInt cout = ((q & x) | ((q | x) & ~s0)) >> 31u;   // compare-free carry
        vUInt s = s0 + carry_in;
        v_if (carry_in != 0) { v_if (s == 0) { cout = 1; } v_endif; } v_endif;
        dst_reg[ob + i] = s;
        dst_reg[cb + i] = cout;
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
inline void q256_add_limb_tile(uint32_t dq, uint32_t dx, uint32_t dout, uint32_t dcarry) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(q256_add_limb_face, dq, dx, dout, VectorMode::RC, dcarry));
}

inline void decode_and_product(uint32_t cb_a, uint32_t cb_b, uint32_t a_tile, uint32_t b_tile) {
    copy_tile_to_dst_init_short(cb_a);
    copy_tile(cb_a, a_tile, SLOT_CODE);
    bp16_decode_tile(SLOT_CODE, SLOT_SP, SLOT_MP, SLOT_EP);
    copy_tile_to_dst_init_short(cb_b);
    copy_tile(cb_b, b_tile, SLOT_CODE);
    bp16_decode_tile(SLOT_CODE, SLOT_SB, SLOT_MB, SLOT_EB);
    bp16_product_tile(SLOT_SP, SLOT_MP, SLOT_EP, SLOT_SB, SLOT_MB, SLOT_EB);
}

// ---- C_k: signed contribution of pair k (place + 1-limb negate) -> cb_C (8 tiles) ----
inline void compute_C(uint32_t cb_a, uint32_t cb_b, uint32_t cb_C, uint32_t k) {
    for (uint32_t g = 0; g < 2; ++g) {
        const uint32_t base = g * 4;
        tile_regs_acquire();
        decode_and_product(cb_a, cb_b, k, k);
        bp16_place_tile(SLOT_MP, SLOT_EP, SLOT_O0, base);
        bp16_neg1_tile(SLOT_MP, SLOT_EP, SLOT_SP, SLOT_O0 + 0, base + 0);
        bp16_neg1_tile(SLOT_MP, SLOT_EP, SLOT_SP, SLOT_O0 + 1, base + 1);
        bp16_neg1_tile(SLOT_MP, SLOT_EP, SLOT_SP, SLOT_O0 + 2, base + 2);
        bp16_neg1_tile(SLOT_MP, SLOT_EP, SLOT_SP, SLOT_O0 + 3, base + 3);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_C, 4);
        pack_tile(SLOT_O0 + 0, cb_C, 0);
        pack_tile(SLOT_O0 + 1, cb_C, 1);
        pack_tile(SLOT_O0 + 2, cb_C, 2);
        pack_tile(SLOT_O0 + 3, cb_C, 3);
        cb_push_back(cb_C, 4);
        tile_regs_release();
    }
}

// ---- q_out = q_in + C  (q256_add, 4+4 leases + carry-mid CB) ----
inline void quire_add(uint32_t cb_qin, uint32_t cb_C, uint32_t cb_qout, uint32_t cb_zero,
                      uint32_t cb_carry_mid) {
    for (uint32_t batch = 0; batch < 2; ++batch) {
        const uint32_t limb0 = batch * QADD_BATCH;
        const auto cb_cin = (batch == 0) ? cb_zero : cb_carry_mid;
        if (batch != 0) cb_wait_front(cb_carry_mid, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_cin);
        copy_tile(cb_cin, 0, QSLOT_CARRY);
        for (uint32_t k = 0; k < QADD_BATCH; ++k) {
            const uint32_t limb = limb0 + k;
            copy_tile_to_dst_init_short(cb_qin);
            copy_tile(cb_qin, limb, QSLOT_Q);
            copy_tile_to_dst_init_short(cb_C);
            copy_tile(cb_C, limb, QSLOT_X);
            q256_add_limb_tile(QSLOT_Q, QSLOT_X, QSLOT_OUT0 + k, QSLOT_CARRY);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_qout, QADD_BATCH);
        for (uint32_t k = 0; k < QADD_BATCH; ++k) pack_tile(QSLOT_OUT0 + k, cb_qout, k);
        cb_push_back(cb_qout, QADD_BATCH);
        if (batch == 0) {
            cb_reserve_back(cb_carry_mid, 1);
            pack_tile(QSLOT_CARRY, cb_carry_mid, 0);
            cb_push_back(cb_carry_mid, 1);
        }
        tile_regs_release();
        if (batch != 0) cb_pop_front(cb_carry_mid, 1);
    }
}

// ---- copy q (8 tiles) from src CB to dst CB (the final result -> cb_out for the writer) ----
inline void copy_q(uint32_t cb_src, uint32_t cb_dst) {
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_src);
    for (uint32_t l = 0; l < 8; ++l) copy_tile(cb_src, l, l);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_dst, 8);
    for (uint32_t l = 0; l < 8; ++l) pack_tile(l, cb_dst, l);
    cb_push_back(cb_dst, 8);
    tile_regs_release();
}


// Seed the quire ping buffer cb_qA = 0 for a row (compute-side, from the single cb_zero tile).
// Keeping the seed compute-side means cb_qA is used ONLY by the K-loop ping-pong (no reader
// interleave -> no FIFO race on cb_qA mid-row).
inline void seed_qzero(uint32_t cb_zero, uint32_t cb_qA) {
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_zero);
    for (uint32_t l = 0; l < 8; ++l) copy_tile(cb_zero, 0, l);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_qA, 8);
    for (uint32_t l = 0; l < 8; ++l) pack_tile(l, cb_qA, l);
    cb_push_back(cb_qA, 8);
    tile_regs_release();
}

// MULTI-ROW per core: loop over this core's assigned output rows. W (cb_b) is loaded ONCE and
// reused across all rows; per row the compute seeds the quire to 0, the reader feeds the row's
// K a-tiles, the in-kernel K-loop accumulates (ping-pong), and the row's quire is written.

// FULL arbitrary-(M,N,K): multi-row per core AND streaming-K. Per row: seed quire=0; for each k
// consume the STREAMED pair (a_k,b_k) just-in-time (cb_a/cb_b shallow ring buffers, tile 0), pop,
// accumulate via the ping-pong quire; write the row's quire. Both A and W stream per-k (K unbounded
// by L1). Reuses every face + compute_C/quire_add/copy_q/seed_qzero verbatim.
void kernel_main() {
    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;
    constexpr auto cb_zero = tt::CBIndex::c_2;
    constexpr auto cb_carry_mid = tt::CBIndex::c_3;
    constexpr auto cb_C = tt::CBIndex::c_5;
    constexpr auto cb_qA = tt::CBIndex::c_6;
    constexpr auto cb_qB = tt::CBIndex::c_7;
    constexpr auto cb_out = tt::CBIndex::c_16;

    const uint32_t n_rows = get_arg_val<uint32_t>(0);
    const uint32_t k_len = get_arg_val<uint32_t>(1);

    init_sfpu(cb_a, cb_C);
    cb_wait_front(cb_zero, 1);

    for (uint32_t r = 0; r < n_rows; ++r) {
        seed_qzero(cb_zero, cb_qA);   // q = 0 for this row
        for (uint32_t k = 0; k < k_len; ++k) {
            const auto cb_qin = (k & 1u) ? cb_qB : cb_qA;
            const auto cb_qout = (k & 1u) ? cb_qA : cb_qB;
            cb_wait_front(cb_a, 1);       // streamed pair k of this row
            cb_wait_front(cb_b, 1);
            cb_wait_front(cb_qin, 8);
            compute_C(cb_a, cb_b, cb_C, /*tile=*/0);
            cb_pop_front(cb_a, 1);
            cb_pop_front(cb_b, 1);
            cb_wait_front(cb_C, 8);
            quire_add(cb_qin, cb_C, cb_qout, cb_zero, cb_carry_mid);
            cb_pop_front(cb_C, 8);
            cb_pop_front(cb_qin, 8);
        }
        const auto cb_final = (k_len & 1u) ? cb_qB : cb_qA;
        copy_q(cb_final, cb_out);
        cb_pop_front(cb_final, 8);
    }
    cb_pop_front(cb_zero, 1);
}
