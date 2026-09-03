// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// sfpu_bp16_decode.cpp — MILESTONE M1 of the SFPU-vectorized exact-quire b-posit
// matmul: a LANE-PARALLEL bp16_decode. 32 bposit-16 codes (one per SFPU lane) are
// decoded simultaneously to their dyadic (sign, M, E2) form, BIT-IDENTICAL per lane
// to the scalar oracle bp16_decode (kernel/bp16_decode.h, ES=3, useed=256).
//
// This isolates the decode the same way bposit_quire_sfpu_add isolated the carry
// vector. Unlike the quire add, decode is PER-LANE INDEPENDENT (no inter-limb carry),
// so there is no lease/carry-handoff machinery: copy the 32-code tile once, run ONE
// SFPU op that writes three output tiles (sign, M, E2), pack. Same INT32-on-SFPU
// constraints as the add: fp32_dest_acc_en=true (host), VectorMode::RC sweeps faces.
//
// THE SIMD DECODE (vs the scalar bp16_decode.h):
//   p     = code & 0xFFFF
//   sign  = (p>>15)&1 ;  rest = p & 0x7FFF ;  if(sign) rest = (~rest+1)&0x7FFF
//   lead  = (rest>>14)&1                          // regime's leading bit
//   norm  = lead ? (~rest & 0x7FFF) : rest        // make the leading run be leading-ZEROS
//   rs    = (norm==0) ? 15 : 14 - msb(norm)       // regime run-length; msb via 4-step search
//   k     = lead ? (rs-1) : (-rs)                 // uniform: rs-1 gives 14 at rs=15; -rs gives -15
//   remaining = (rs==15) ? 0 : 14 - rs            // clamp saturated regime to the no-field case
//   rest2 = rest & ((1<<remaining)-1)
//   e_width = (remaining>=4) ? 3 : remaining      // == min(ES=3, remaining) since remaining<=3 is itself
//   e     = ((rest2 >> (remaining-e_width)) & ((1<<e_width)-1)) << (3 - e_width)
//   f_width = remaining - e_width
//   f_bits  = rest2 & ((1<<f_width)-1)            // 0 when f_width==0
//   M     = (1<<f_width) + f_bits                 // == 1 when f_width==0 (uniform; no branch)
//   E2    = (k<<3) + e - f_width                  // 8*k + e - f_width (two's-complement)
//   if (rest==0) { sign=0; M=0; E2=0 }            // ZERO or NaR -> special (both give rest==0)
// Every step is a native vUInt op: & | ~ + - and variable << >> (sfpi_funcs.h:511-534),
// with v_if predication for the 5 data-dependent branches. NO magnitude compare and NO
// large immediate (the two SFPU traps that stalled the carry — see sfpu_q256_add.cpp).
//
// SFPU OPS RELIED ON (file:line in runtime/sfpi/include/sfpi_funcs.h):
//   + - on vUInt ............ 512,513,515,517      & | ~ ... 522,523,529 (+ uint32 overloads)
//   << >> by vUInt amount ... 519,521              << >> by const ... 518,520
//   != == vs uint32_t ....... 551,550              v_if / v_endif ... sfpi.h
//   dst_reg[i] as vUInt ..... sfpi_classes.h        face-fn / _params_ ... as in sfpu_q256_add.cpp

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

// dst tiles: CODE in, SIGN/M/E2 out. 4 INT32 tiles <= 8 (SyncFull/accum=true budget).
#define SLOT_CODE 0
#define SLOT_SIGN 1
#define SLOT_M    2
#define SLOT_E2   3

#ifdef TRISC_MATH
using namespace sfpi;

// Decode one tile's worth (VectorMode::RC sweeps all 4 faces; i<8 vectors per face,
// exactly the sfpu_q256_add.cpp face shape). Reads SLOT_CODE, writes SLOT_SIGN / SLOT_M
// / SLOT_E2 — all per lane, bit-identical to scalar bp16_decode.
inline void bp16_decode_face(
    const uint32_t dst_code, const uint32_t dst_sign, const uint32_t dst_M, const uint32_t dst_E2) {
    constexpr uint32_t NV = 32;
    const uint32_t cb = dst_code * NV, sb = dst_sign * NV, mb = dst_M * NV, eb = dst_E2 * NV;

    // NOTE: every compound expression is broken into single binary ops with a shared scratch
    // `t` — the SFPU LREG file is tiny, and deeply-nested expressions overflow the register
    // allocator ("maximum reload insns per insn" ICE). Keep statements simple, reuse `t`.
    for (uint32_t i = 0; i < 8; i++) {
        vUInt p = dst_reg[cb + i];
        p = p & 0xFFFF;

        vUInt sign = p >> 15;                 // p is 16-bit -> 0/1
        vUInt rest = p & 0x7FFF;
        vUInt t;
        v_if (sign != 0) { t = ~rest; t = t + 1; rest = t & 0x7FFF; } v_endif;  // 2's-comp of 15-bit field

        // regime: normalize the leading run to leading-zeros (norm), then rs = 14 - msb(norm).
        vUInt lead = rest >> 14;              // rest is 15-bit -> 0/1
        vUInt norm = rest;
        v_if (lead != 0) { t = ~rest; norm = t & 0x7FFF; } v_endif;

        vUInt v = norm;                       // search on a COPY so norm survives for the ==0 test
        vUInt b = 0;                          // msb(norm) in [0,13] (norm bit14 always 0); 4-step search
        t = v >> 8;  v_if (t != 0) { b = b + 8; v = v >> 8; } v_endif;
        t = v >> 4;  v_if (t != 0) { b = b + 4; v = v >> 4; } v_endif;
        t = v >> 2;  v_if (t != 0) { b = b + 2; v = v >> 2; } v_endif;
        t = v >> 1;  v_if (t != 0) { b = b + 1;             } v_endif;

        vUInt rs = vUInt(14) - b;
        v_if (norm == 0) { rs = 15; } v_endif;   // norm==0 (all-equal regime field, incl. 0x7FFF) -> saturated

        // k = lead ? (rs-1) : (-rs)
        vUInt k = rs - 1;
        v_if (lead == 0) { k = vUInt(0) - rs; } v_endif;

        // remaining = (rs==15) ? 0 : 14 - rs  (clamp; collapses the saturated regime to M=1,e=0,E2=8k)
        vUInt remaining = vUInt(14) - rs;
        v_if (rs == 15) { remaining = 0; } v_endif;

        // rest2 = rest & ((1<<remaining)-1)
        t = vUInt(1) << remaining; t = t - 1;
        vUInt rest2 = rest & t;

        // e_width = min(3, remaining)
        vUInt e_width = remaining;
        t = remaining >> 2; v_if (t != 0) { e_width = 3; } v_endif;   // remaining>=4 -> 3

        vUInt fw = remaining - e_width;          // f_width
        // e = ((rest2 >> fw) & ((1<<e_width)-1)) << (3 - e_width)
        vUInt e = rest2 >> fw;
        t = vUInt(1) << e_width; t = t - 1;
        e = e & t;
        t = vUInt(3) - e_width;
        e = e << t;

        // f_bits = rest2 & ((1<<fw)-1) ;  M = (1<<fw) + f_bits  (== 1 when fw==0)
        t = vUInt(1) << fw; t = t - 1;
        vUInt M = rest2 & t;                      // reuse: M temporarily holds f_bits
        t = vUInt(1) << fw;
        M = M + t;

        // E2 = (k<<3) + e - fw   (8*k + e - f_width, two's-complement)
        vUInt E2 = k << 3;
        E2 = E2 + e;
        E2 = E2 - fw;

        // ZERO (0x0000) and NaR (0x8000) both reduce to rest==0 -> special (sign=M=E2=0).
        v_if (rest == 0) { sign = 0; M = 0; E2 = 0; } v_endif;

        dst_reg[sb + i] = sign;
        dst_reg[mb + i] = M;
        dst_reg[eb + i] = E2;
    }
}
#endif  // TRISC_MATH

// Forward the 4 dst indices through the variadic _params_ (VectorMode::RC explicit),
// exactly the binding sfpu_q256_add.cpp uses — the "binary" name is just the dst-index
// forwarding harness; we use one input (CODE) and three outputs.
inline void bp16_decode_tile(uint32_t dst_code, uint32_t dst_sign, uint32_t dst_M, uint32_t dst_E2) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(
        bp16_decode_face, dst_code, dst_sign, dst_M, VectorMode::RC, dst_E2));
}

void kernel_main() {
    // CB layout (host): c_0 = CODES (1 Int32 tile, 32 lane codes replicated across rows);
    //                   c_16 = OUT (3 Int32 tiles: sign, M, E2).
    constexpr auto cb_codes = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_codes, cb_out);
    cb_wait_front(cb_codes, 1);

    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_codes);
    copy_tile(cb_codes, /*tile=*/0, /*dst=*/SLOT_CODE);
    bp16_decode_tile(SLOT_CODE, SLOT_SIGN, SLOT_M, SLOT_E2);
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_out, 3);
#if defined(ECHO_INPUT) && (ECHO_INPUT == 1)
    pack_tile(SLOT_CODE, cb_out, 0);   // echo the loaded code back (verify the SFPU saw the right input)
    pack_tile(SLOT_CODE, cb_out, 1);
    pack_tile(SLOT_CODE, cb_out, 2);
#else
    pack_tile(SLOT_SIGN, cb_out, 0);
    pack_tile(SLOT_M, cb_out, 1);
    pack_tile(SLOT_E2, cb_out, 2);
#endif
    cb_push_back(cb_out, 3);
    tile_regs_release();
    cb_pop_front(cb_codes, 1);
}
