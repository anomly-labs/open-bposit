// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// sfpu_bp16_product.cpp — MILESTONE M2 of the SFPU-vectorized exact-quire b-posit
// matmul: a LANE-PARALLEL bp16 PRODUCT. Per lane, two bposit-16 codes (a, b) are each
// decoded (M1's bp16_decode) to (sign, M, E2), then combined into the dyadic product:
//     sign_p = sa ^ sb ;  M_p = Ma * Mb ;  E2_p = Ea + Eb
// bit-identical per lane to the scalar oracle (decode both, then xor/mul/add). This is
// the product half of the scalar bp16_madd_q256 (kernel/bp16_quire.h:77), before
// the windowed placement+carry (M3).
//
// One acquire lease, three SFPU ops (q256_add proved multi-op-per-lease works):
//   op1 decode A -> dst SLOT_SA/MA/EA          (reads SLOT_CODE)
//   op2 decode B -> dst SLOT_SB/MB/EB          (reads SLOT_CODE, reloaded)
//   op3 product  -> writes sign_p/M_p/E2_p IN PLACE over SLOT_SA/MA/EA (reads all 6)
// Live INT32 dst tiles: SA,MA,EA,SB,MB,EB + one CODE scratch = 7 <= 8 (SyncFull/accum).
//
// Mantissa multiply = SHIFT-ADD (Ma,Mb < 2^11): provably exact and fully controlled,
// chosen over sfpmul24/fractional_mul (a fixed-point "fractional" multiply with
// half-selection scaling — ambiguous for an exact integer product). 11 conditional
// shifted adds; only prod/a/m/t are live in the multiply (register-light).
//
// SFPU LESSONS (carried from q256_add + decode M1): no magnitude compare, no large
// immediate; split nested exprs to single ops (tiny LREG file -> reload ICE);
// kernel_main entry; assign dst_reg[] to vUInt before bit ops; INT32 => host
// fp32_dest_acc_en=true + dst_full_sync_en=true.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

// dst tiles
#define SLOT_SA   0   // sign_a  -> overwritten with sign_p
#define SLOT_MA   1   // M_a     -> overwritten with M_p
#define SLOT_EA   2   // E2_a    -> overwritten with E2_p
#define SLOT_SB   3
#define SLOT_MB   4
#define SLOT_EB   5
#define SLOT_CODE 6   // scratch for the code being decoded

#ifdef TRISC_MATH
using namespace sfpi;

// --- M1 decode face (verbatim from bposit_decode_sfpu) -----------------------
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

// --- M2 product face: (sa,Ma,Ea),(sb,Mb,Eb) -> sign_p,M_p,E2_p in place ------
// Writes sign_p over dst_sa, M_p over dst_Ma, E2_p over dst_Ea.
inline void bp16_product_face(
    const uint32_t dst_sa, const uint32_t dst_Ma, const uint32_t dst_Ea,
    const uint32_t dst_sb, const uint32_t dst_Mb, const uint32_t dst_Eb) {
    constexpr uint32_t NV = 32;
    const uint32_t sa = dst_sa * NV, ma = dst_Ma * NV, ea = dst_Ea * NV;
    const uint32_t sb = dst_sb * NV, mb = dst_Mb * NV, eb = dst_Eb * NV;
    for (uint32_t i = 0; i < 8; i++) {
        // sign_p = sa ^ sb ; E2_p = Ea + Eb (two's-complement) — simple.
        vUInt sgn = dst_reg[sa + i];
        vUInt tb = dst_reg[sb + i];
        sgn = sgn ^ tb;

        vUInt e2 = dst_reg[ea + i];
        tb = dst_reg[eb + i];
        e2 = e2 + tb;

        // M_p = Ma * Mb via shift-add (Ma,Mb < 2^11). Only prod/a/m/t live here.
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

        dst_reg[sa + i] = sgn;   // sign_p
        dst_reg[ma + i] = prod;  // M_p
        dst_reg[ea + i] = e2;    // E2_p
    }
}
#endif  // TRISC_MATH

inline void bp16_decode_tile(uint32_t dst_code, uint32_t dst_sign, uint32_t dst_M, uint32_t dst_E2) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(
        bp16_decode_face, dst_code, dst_sign, dst_M, VectorMode::RC, dst_E2));
}
inline void bp16_product_tile(uint32_t dst_sa, uint32_t dst_Ma, uint32_t dst_Ea,
                              uint32_t dst_sb, uint32_t dst_Mb, uint32_t dst_Eb) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(
        bp16_product_face, dst_sa, dst_Ma, dst_Ea, VectorMode::RC, dst_sb, dst_Mb, dst_Eb));
}

void kernel_main() {
    // CB layout (host): c_0 = CODE_A (1 tile), c_1 = CODE_B (1 tile);
    //                   c_16 = OUT (3 tiles: sign_p, M_p, E2_p).
    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_a, cb_out);
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);

    tile_regs_acquire();

    // op1: decode A
    copy_tile_to_dst_init_short(cb_a);
    copy_tile(cb_a, 0, SLOT_CODE);
    bp16_decode_tile(SLOT_CODE, SLOT_SA, SLOT_MA, SLOT_EA);

    // op2: decode B (restore datacopy MATH mode before the copy — the SFPU op above
    // reconfigured MATH, same interleave fix as q256_add's per-limb copies).
    copy_tile_to_dst_init_short(cb_b);
    copy_tile(cb_b, 0, SLOT_CODE);
    bp16_decode_tile(SLOT_CODE, SLOT_SB, SLOT_MB, SLOT_EB);

    // op3: product in place over A's slots
    bp16_product_tile(SLOT_SA, SLOT_MA, SLOT_EA, SLOT_SB, SLOT_MB, SLOT_EB);

    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 3);
    pack_tile(SLOT_SA, cb_out, 0);   // sign_p
    pack_tile(SLOT_MA, cb_out, 1);   // M_p
    pack_tile(SLOT_EA, cb_out, 2);   // E2_p
    cb_push_back(cb_out, 3);
    tile_regs_release();

    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}
