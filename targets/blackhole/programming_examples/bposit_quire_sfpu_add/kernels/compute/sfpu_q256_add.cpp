// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// sfpu_q256_add.cpp — the FIRST SFPU-vectorized milestone of Anomly's EXACT
// 256-bit b-posit16 Kulisch quire on Blackhole: a LANE-PARALLEL exact q256_add.
//
// 32 INDEPENDENT 256-bit accumulators run across the SFPU's 32 lanes. Each lane
// holds an 8-limb (8x u32, little-endian, two's-complement) quire and adds a
// second 8-limb operand, with correct inter-limb RIPPLE CARRY propagated limb-to
// -limb across all 32 lanes simultaneously. Must be BIT-IDENTICAL per lane to the
// scalar reference q256_add (kernel/bp16_quire.h:62-65):
//
//     unsigned long long carry = 0;
//     for (int i = 0; i < 8; i++) {
//         unsigned long long t = (u64)q[i] + x[i] + carry;
//         q[i] = (u32)t;
//         carry = t >> 32;
//     }
//
// i.e. plain 256-bit two's-complement (mod 2^256) addition. This milestone
// ISOLATES and proves the vectorized carry vector — NOT the bp16 decode/product
// (a later milestone). The carry is the hardest SFPU primitive: the SFPU has no
// native cross-lane carry-out flag, so carry-out is synthesized PER LANE by pure bit
// algebra — the closed-form MSB carry ((a&b)|((a|b)&~s0))>>31 — NOT an unsigned-< compare.
// (SFPI operator< does not give unsigned LT on this silicon and the 0x80000000 sign-flip
// immediate does not encode; both no-op'd at 14/32 — the bit-algebra form took it to 32/32,
// 2026-06-28.) See docs/blackhole/quire-sfpu-kernel-design.md §4.3 step 4.
//
// ============================================================================
// FINAL DESIGN (2026-06-24) — the 4+4 TWO-BATCH single-in-place-carry ripple.
//
// Two confirmed hard constraints shape this kernel:
//
//   (1) INT32/32-bit data REQUIRES fp32_dest_acc_en = TRUE (host). DEST is a 16-bit
//       register file; loading 32-bit tiles into a 16-bit-configured DEST produces
//       TILE-ALIGNED CORRUPTION — stated verbatim at binary_ng_program_factory.cpp:
//       714-721, and every shipped INT32 SFPU op sets it true (ternary_program_
//       factory.cpp:1253-1254, used by ternary_addcmul_int_sfpu.cpp). This was the
//       real bug behind FIX 1–5: with it false, limbs 0–2 aliased OK, limbs 3–7 read
//       0, and the carry showed a period-4 artifact (a 16-vs-32-bit stride mismatch,
//       NOT the carry algebra). Setting it true took the gate 8/32 → 14/32.
//
//   (2) With fp32_dest_acc_en=true the INT32 DEST budget HALVES to 8 tiles in SyncFull
//       (get_dest_max_tiles<SyncFull,accum=true,Tile32x32> = (1024>>1)>>6 = 8;
//       ckernel.h:835-842, tensix_types.h:191-194; cf. add_int_sfpu.h:21-22). AND
//       pack_tile REQUIRES a preceding tile_regs_commit (_llk_math_dest_section_done_,
//       llk_math_common.h:86-95, which posts the MATH_PACK sem that the packer waits
//       on, llk_pack_common.h:19-22) — and after the pack, tile_regs_release ZEROes
//       the WHOLE DEST in SyncFull (TTI_ZEROACC CLR_ALL, llk_pack_common.h:39-42). So
//       you CANNOT continue math (the next limb) after a pack in the same lease, and a
//       dst slot (e.g. a persistent carry) does NOT survive a release. Every shipped
//       compute kernel is (acquire; ALL math; commit; wait; ALL packs; release) — no
//       math after any pack in a lease (eltwise_binary_sfpu.cpp:66-91,
//       sort_single_row_single_core.cpp:139-189, combine_welford.h:66-153).
//
// Consequence: a SINGLE lease cannot hold all 8 OUT (8) + carry (1) + Q/X scratch (2)
// = 11 > 8, and cannot pack-then-keep-rippling. The clean structure that satisfies
// both constraints is TWO leases of FOUR limbs each:
//
//   Lease A (limbs 0..3): carry lives in ONE persistent dst slot, seeded 0, updated
//     IN PLACE across the 4 limbs (read carry_in, write carry_out to the SAME slot —
//     safe per-lane because carry_in is read to a register before carry_out is written;
//     this is the proven Welford in-place-accumulate pattern, combine_welford.h:101-
//     103,114-115,126-127). Q,X are reused scratch each limb. OUT0..3 each in their own
//     dst slot. Live INT32 tiles = OUT0..3(4) + CARRY(1) + Q(1) + X(1) = 7 ≤ 8. After
//     the 4 SFPU adds: ONE commit, then pack OUT0..3 to cb_out AND the limb-3 carry-out
//     to a SINGLE carry CB (the handoff). All packs after the single commit; no math
//     after any pack.
//
//   Lease B (limbs 4..7): copy the handed-off carry from the single carry CB into the
//     dst carry slot (carry_in for limb 4), then ripple limbs 4..7 in place exactly as
//     Lease A. Pack OUT4..7. The carry CB tile is WRITTEN in Lease A and READ in Lease
//     B — different leases, so there is NO self-alias hazard (the single-CB round-trip
//     that broke the earlier per-limb ping-pong attempts only self-aliased when read
//     and written within ONE lease). One carry handoff, one CB tile, one direction.
//
// SyncFull (host dst_full_sync_en=true) PINS the dst base at 0 (no per-acquire half-
// flip; set_dest_section_base<StartZero>, llk_math_common.h:104-108), so both leases
// share one stable slot map. The per-lane carry arithmetic is reused VERBATIM (it was
// always bit-exact vs the scalar oracle); only the carry's residence changed.
// ============================================================================
//
// SFPU OPS RELIED ON — every one grounded in real tt-metal (file:line):
//   - vUInt / vInt = 32-bit x 32 lanes ......... runtime/sfpi/include/sfpi.h, sfpi_classes.h
//   - vUInt operator+ (vUInt,vUInt) ............ sfpi_funcs.h:512 (a.int_add(b))
//   - vUInt & | ~ >> (closed-form carry) ....... sfpi_funcs.h:523,529,522,520 (NOT operator<:
//        it isn't unsigned-LT here, so carry is bit-algebra ((a&b)|((a|b)&~s0))>>31)
//   - operator!= (vUInt,uint32_t) .............. sfpi_funcs.h:551 (carry_in != 0)
//   - operator== (vUInt,uint32_t) .............. sfpi_funcs.h:550 (s == 0)
//   - v_if / v_endif predication ............... sfpi.h (v_if/v_endif macros)
//   - dst_reg[i] read/write as vUInt ........... sfpi_classes.h:298-306,366-368
//   - face fn shape (dst_reg[base+i], i<8) ..... custom_sfpi_add/.../tiles_add.cpp:62-77
//   - _llk_math_eltwise_binary_sfpu_params_ .... tt_llk_blackhole/llk_lib/
//        llk_math_eltwise_binary_sfpu_params.h:12-27 (variadic; VectorMode::RC sweeps
//        all 4 faces, llk_math_eltwise_sfpu_common.h:80-89).
//   - init_sfpu/copy_tile/tile_regs_*/pack_tile  custom_sfpi_add/.../tiles_add.cpp:122-150
//   - in-place accumulate across a loop in ONE lease (the carry model) ... Welford
//        ttnn/.../normalization/kernel_util/compute/combine_welford.h:66-153

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

#define QSFPU_LIMBS 8
#define QSFPU_BATCH 4   // limbs per lease (4 OUT + CARRY + Q + X = 7 <= 8 INT32 budget)

// --- Dst-register tile slots (one INT32 tile = 32 lanes of one limb) ---------
// Within EITHER lease, only these 7 INT32 dst tiles are live (<= 8 SyncFull/accum=true):
//   Q(0), X(1) : reused scratch each limb of the batch
//   CARRY(2)   : persistent across the 4 limbs of the batch; in-place carry_in==carry_out
//   OUT0..3    : slots 3..6 — one per limb of the batch, packed once after the batch
#define SLOT_Q 0
#define SLOT_X 1
#define SLOT_CARRY 2
#define SLOT_OUT0 3   // batch OUT tiles occupy slots 3,4,5,6

// The SFPU is only on the MATH (TRISC) core; TRISC_MATH gates the sfpi body
// (same guard as custom_sfpi_add/kernels/compute/tiles_add.cpp:14).
#ifdef TRISC_MATH

using namespace sfpi;

// One LIMB of the lane-parallel add, for one tile FACE. The _params_ helper calls
// this once per face and auto-advances the dst face base between calls; within a
// face we sweep the 8 SIMD vectors (32 lanes each) just like tiles_add.cpp:62-77.
//
// Per lane, exactly mirroring the scalar u64 ripple of q256_add:
//   s0     = q + x                                 (mod 2^32)
//   cout_a = ((q&x)|((q|x)&~s0)) >> 31  in {0,1}    // q+x overflowed 2^32 (closed-form MSB carry,
//                                                   //   NOT s0<x — operator< isn't unsigned here)
//   s      = s0 + carry_in                          (carry_in in {0,1})
//   cout_b = (carry_in==1 && s==0)     ? 1 : 0     // adding the carry overflowed
//   OUT    = s ;  COUT = cout_a | cout_b
//
// cout_a and cout_b are MUTUALLY EXCLUSIVE for a given limb: cout_b needs
// s0==0xFFFFFFFF (so q+x did NOT wrap, cout_a=0); cout_a needs s0!=0xFFFFFFFF after a
// wrap (so +1 cannot wrap again, cout_b=0). Hence an OR == the u64 carry.
//
// The CARRY slot is read (carry_in) and written (carry_out) IN PLACE: dst_carry is the
// SAME slot for both. carry_in is loaded into a register BEFORE cout is written back, so
// the in-place update is safe per lane. The carry persists across the 4 limbs of the
// batch because the caller passes the SAME SLOT_CARRY each limb within ONE lease.
inline void q256_add_limb_face(
    const uint32_t dst_q, const uint32_t dst_x, const uint32_t dst_out, const uint32_t dst_carry) {
    constexpr uint32_t n_vector_in_tile = 32;  // 32 SIMD lanes per tile (BH/WH)
    const uint32_t q_base = dst_q * n_vector_in_tile;
    const uint32_t x_base = dst_x * n_vector_in_tile;
    const uint32_t out_base = dst_out * n_vector_in_tile;
    const uint32_t carry_base = dst_carry * n_vector_in_tile;

    for (uint32_t i = 0; i < 8; i++) {
        vUInt q = dst_reg[q_base + i];
        vUInt x = dst_reg[x_base + i];
        vUInt carry_in = dst_reg[carry_base + i];  // 0 or 1 per lane (prev limb's carry)

        vUInt s0 = q + x;                       // sfpi_funcs.h:512
        // Carry-out of q+x, COMPARE-FREE. The SFPI unsigned-< wrap test (s0 < x) and its
        // sign-flip variant both no-op on silicon — operator< doesn't give unsigned LT and the
        // 0x80000000 immediate doesn't encode. So derive the carry by pure bit algebra (the
        // closed form for the MSB carry-out of a+b): cout = ((a&b) | ((a|b) & ~s0)) >> 31.
        // Hand-verified bit-exact vs the scalar oracle on lanes 7/12/16. No immediate, no compare;
        // & | ~ >> are all native vUInt ops (sfpi_funcs.h:520,522,523,529). >> binds tighter than
        // |, so the outer parens are load-bearing.
        vUInt cout = ((q & x) | ((q | x) & ~s0)) >> 31u;

        vUInt s = s0 + carry_in;
        v_if(carry_in != 0) {                   // carry_in == 1 (sfpi_funcs.h:551)
            v_if(s == 0) {                      // s0 was all-ones, +1 -> 0 (sfpi_funcs.h:550)
                cout = 1;                       // mutually exclusive with the above
            }
            v_endif;
        }
        v_endif;

        dst_reg[out_base + i] = s;
        dst_reg[carry_base + i] = cout;         // in-place: carry_in of the NEXT limb (residence proven irrelevant)
    }
}

#endif  // TRISC_MATH

// Run ONE limb's lane-parallel add over the whole tile. Same wrapping idiom as
// custom_sfpi_add/kernels/compute/tiles_add.cpp:107-109, but our face fn takes one
// EXTRA dst index (the persistent in-place carry slot); the variadic _params_ forwards
// it AFTER VectorMode (llk_math_eltwise_binary_sfpu_params.h:12-27), so VectorMode::RC
// is explicit and sweeps all 4 faces.
inline void q256_add_limb_tile(uint32_t dst_q, uint32_t dst_x, uint32_t dst_out, uint32_t dst_carry) {
    MATH(_llk_math_eltwise_binary_sfpu_params_(
        q256_add_limb_face, dst_q, dst_x, dst_out, VectorMode::RC, dst_carry));
}

void kernel_main() {
    // CB layout (set by the host program):
    //   c_0  : Q limbs    — 8 Int32 tiles (limb 0..7)
    //   c_1  : X limbs    — 8 Int32 tiles (limb 0..7)
    //   c_2  : ZERO       — 1 Int32 all-zero tile (seeds CARRY = 0 in Lease A)
    //   c_3  : CARRY-MID  — 1 Int32 tile: the limb-3 carry-out handed from Lease A to B
    //   c_16 : OUT        — 8 Int32 tiles (limb 0..7), produced low->high
    constexpr auto cb_q = tt::CBIndex::c_0;
    constexpr auto cb_x = tt::CBIndex::c_1;
    constexpr auto cb_zero = tt::CBIndex::c_2;
    constexpr auto cb_carry_mid = tt::CBIndex::c_3;
    constexpr auto cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_q, cb_out);

    cb_wait_front(cb_q, QSFPU_LIMBS);
    cb_wait_front(cb_x, QSFPU_LIMBS);
    cb_wait_front(cb_zero, 1);

    // Two batches of 4 limbs. Batch b covers limbs [b*4 .. b*4+3]. The carry crosses
    // ONCE, from batch 0 to batch 1, via the single carry-mid CB tile.
    for (uint32_t batch = 0; batch < (QSFPU_LIMBS / QSFPU_BATCH); ++batch) {
        const uint32_t limb0 = batch * QSFPU_BATCH;

        // Carry-in source for this batch: batch 0 seeds 0 (cb_zero); batch 1 reads the
        // limb-3 carry-out from cb_carry_mid (written by batch 0 below).
        const auto cb_cin = (batch == 0) ? cb_zero : cb_carry_mid;
        if (batch != 0) {
            cb_wait_front(cb_carry_mid, 1);
        }

        tile_regs_acquire();

        // Seed the persistent CARRY slot for this batch (in dst SLOT_CARRY). It then
        // persists and updates IN PLACE across the 4 limbs of the batch.
        copy_tile_to_dst_init_short(cb_cin);
        copy_tile(cb_cin, /*tile=*/0, /*dst=*/SLOT_CARRY);

        for (uint32_t k = 0; k < QSFPU_BATCH; ++k) {
            const uint32_t limb = limb0 + k;
            // Restore datacopy MATH mode before each copy group: the previous limb's SFPU
            // BINARY op (_llk_math_eltwise_binary_sfpu_params_ -> _llk_math_eltwise_sfpu_
            // start_) reconfigured MATH for the SFPU op, so the datacopy init must be
            // restored each limb before copy_tile. Int32 format is uniform so the SHORT
            // init suffices — exactly the interleaved-copy+SFPU pattern proven by Welford
            // (combine_welford.h:86,118 interleaves copy_tile_to_dst_init_short with SFPU
            // ops in one lease).
            copy_tile_to_dst_init_short(cb_q);
            copy_tile(cb_q, /*tile=*/limb, /*dst=*/SLOT_Q);
            copy_tile_to_dst_init_short(cb_x);
            copy_tile(cb_x, /*tile=*/limb, /*dst=*/SLOT_X);

            // Reads SLOT_Q, SLOT_X, SLOT_CARRY; writes the limb sum to OUT slot k of this
            // batch and the new carry back into SLOT_CARRY in place (carry_in of limb+1).
            q256_add_limb_tile(SLOT_Q, SLOT_X, SLOT_OUT0 + k, SLOT_CARRY);
        }

        tile_regs_commit();
        tile_regs_wait();

        // Pack the 4 OUT limbs of this batch, and (only for batch 0) the limb-3 carry-out
        // to the carry-mid CB for the handoff. All packs are AFTER the single commit; no
        // math runs after any pack in this lease.
        cb_reserve_back(cb_out, QSFPU_BATCH);
        for (uint32_t k = 0; k < QSFPU_BATCH; ++k) {
#if defined(ECHO_INPUT) && (ECHO_INPUT == 1)
            pack_tile(SLOT_Q, cb_out, /*out_tile=*/k);            // echo loaded Q (limb limb0+3 only)
#elif defined(ECHO_INPUT) && (ECHO_INPUT == 2)
            pack_tile(SLOT_X, cb_out, /*out_tile=*/k);            // echo loaded X (limb limb0+3 only)
#elif defined(ECHO_INPUT) && (ECHO_INPUT == 3)
            pack_tile(SLOT_CARRY, cb_out, /*out_tile=*/k);        // echo this batch's final carry
#elif defined(ECHO_TRACE) && (ECHO_TRACE == 1)
            // TRACE: pack the batch's FINAL carry (SLOT_CARRY = carry-out of limb limb0+3)
            // into every OUT tile of the batch — so the host sees, per batch, where the
            // carry chain ended. (Coarser than the old per-limb trace, but the in-place
            // carry no longer keeps per-limb history; the batch-boundary carry is what
            // localizes a cross-batch handoff failure vs. an in-batch ripple failure.)
            pack_tile(SLOT_CARRY, cb_out, /*out_tile=*/k);
#else
            pack_tile(SLOT_OUT0 + k, cb_out, /*out_tile=*/k);     // OUT[limb0+k] -> CB tile k
#endif
        }
        cb_push_back(cb_out, QSFPU_BATCH);

        // Hand the limb-3 carry-out (SLOT_CARRY, which now holds the carry-out of the
        // batch's last limb) to the next batch via the single carry-mid CB. Only batch 0
        // produces it; batch 1's carry-out is the final 256-bit carry and is discarded
        // (mod-2^256 wrap), matching the scalar reference.
        if (batch == 0) {
            cb_reserve_back(cb_carry_mid, 1);
            pack_tile(SLOT_CARRY, cb_carry_mid, /*out_tile=*/0);
            cb_push_back(cb_carry_mid, 1);
        }

        tile_regs_release();

        if (batch != 0) {
            cb_pop_front(cb_carry_mid, 1);
        }
    }

    cb_pop_front(cb_q, QSFPU_LIMBS);
    cb_pop_front(cb_x, QSFPU_LIMBS);
    cb_pop_front(cb_zero, 1);
}
