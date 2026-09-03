// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: Apache-2.0
//
// quire_attention_baby.cpp — ATTENTION SCORE under MASSIVE-ACTIVATION
// OUTLIERS, on ONE Blackhole baby (RISC-V data-movement) core. For one query q
// and one key k (head_dim D), compute the attention score s = q.k TWO ways and
// write both back:
//
//   (1) EXACT 256-bit Kulisch quire DOT:  q256 = Sum_i bp16_prod_to_q256(q_i,k_i),
//       NO per-product rounding (the EXACT bp16xbp16 FMA-into-quire primitive,
//       kernel/bp16_quire.h:43-60), exact 256-bit ripple-carry accumulate, single
//       truncating bp16 readout (bp16_encode.h). This is the SAME proven path as
//       quire_dot_baby.cpp (bposit_quire_reduce), reused verbatim.
//
//   (2) GENUINE IEEE-754 fp32 NAIVE dot: acc = fp32_add(acc, fp32(q_i)*fp32(k_i)),
//       products formed exactly (bp16*bp16 fits the fp32 24-bit significand) then
//       summed sequentially in DIM ORDER, round-to-nearest-ties-to-even, in PURE
//       INTEGER arithmetic (kernels/ieee_softfp32.h) so it needs no FPU / libgcc
//       and is host-verified bit-exact vs native C `float`. This is exactly what
//       a real fp32 attention kernel does — and where massive-activation outliers
//       (q_i*k_i = +/-2^33 in this construction) catastrophically cancel and
//       SWALLOW the small-but-meaningful signal, distorting the would-be softmax.
//
// The host (bposit_quire_attention.cpp) runs this once per key, gates the quire
// result == exact ground truth (quire error 0) and the fp32 result == the genuine
// fp32 baseline, then softmaxes both score vectors and shows the fp32 argmax FLIP.
// Marshalling is the proven single-page / bulk-NoC discipline (one page per
// buffer, one bulk read/write) from quire_dot_baby.cpp.

#include <cstdint>

// === REUSED golden-exact b-posit numerics (absolute path; do NOT reimplement) =
// Freestanding RV32IM-clean headers, proven byte-identical x86 == qemu-rv32 ==
// ttsim-BRISC, already bit-exact on this Blackhole in bposit_quire_reduce /
// bposit_quire_vs_float. bp16_prod_to_q256 is the EXACT bp16xbp16 product into the
// quire (bp16_quire.h:43-60) — the GEMM/dot primitive, reused here for q.k.
#include "../../../kernel/bp16_quire.h"   // bp16_prod_to_q256, q256_add
#include "../../../kernel/bp16_encode.h"  // bp16_encode_quire256 (QFRAC=96, truncation)
#include "../../../kernel/bp16_decode.h"  // bp16_decode (sign, M, E2)

// Pure-integer IEEE-754 fp32 add (ships beside this kernel, reused verbatim from
// bposit_quire_vs_float). Absolute path, like the bp16 headers above, so it
// resolves regardless of the kernel-compile sandbox include search path.
#include "$TT_METAL_HOME/tt_metal/programming_examples/bposit_quire_attention/kernels/ieee_softfp32.h"

#define QLIMBS 8  // 8x uint32 = 256-bit quire, fraction point at bit 96

// EXACT product q_i*k_i of two bp16 codes, returned as an IEEE-754 binary32 bit
// pattern — built DIRECTLY from the dyadic decode, no rounding, no fp32 multiply.
// The product of two bp16 values is sign_q^sign_k * (M_q*M_k) * 2^(E2_q+E2_k)
// (the SAME exact form bp16_prod_to_q256 uses, bp16_quire.h:43-60), with
// M_q*M_k < 2^22 — so it fits the fp32 24-bit significand EXACTLY and the fp32
// encode is lossless. This is why the comparison is honest: the per-dim product
// the fp32 accumulator sees is the EXACT product (no multiply error); ALL of the
// fp32 error comes from the running ADD (fp32_add), exactly as on a real fp32
// pipeline. It also keeps the kernel 32-bit-integer-only (M_q*M_k is a 11b x 11b
// -> 22b multiply, fits a single uint32 — no 64-bit __muldi3 on RV32), matching
// the discipline of the reused bp16_*.h headers.
static uint32_t bp16_prod_to_fp32_bits(int acode, int bcode) {
    int sa, sb, Ea, Eb;
    unsigned Ma, Mb;
    bp16_decode(acode, &sa, &Ma, &Ea);
    bp16_decode(bcode, &sb, &Mb, &Eb);
    int sign = sa ^ sb;
    if (Ma == 0u || Mb == 0u) return (uint32_t)sign << 31;  // zero / NaR -> +/-0
    unsigned M = Ma * Mb;            // < 2^22, exact, fits uint32 (32x32->low32)
    int E2 = Ea + Eb;                // exact product exponent
    int msb = 31;
    while (msb >= 0 && !((M >> msb) & 1u)) msb--;            // msb <= 21
    // significand: M shifted so its MSB becomes the fp32 implicit leading 1 (bit 23).
    uint32_t sig = (msb <= 23) ? (M << (23 - msb)) : (M >> (msb - 23));
    int biased = E2 + msb + 127;     // unbiased exp of the value = E2 + msb
    if (biased >= 0xFF) return ((uint32_t)sign << 31) | 0x7F800000u;  // (won't happen for bp16)
    if (biased <= 0) return (uint32_t)sign << 31;                      // (won't happen for bp16)
    return ((uint32_t)sign << 31) | ((uint32_t)biased << 23) | (sig & 0x007FFFFFu);
}

void kernel_main() {
    // ---- runtime args (set by the host SetRuntimeArgs, same order) ----------
    uint32_t q_dram = get_arg_val<uint32_t>(0);    // DRAM base: D query codes, 1 uint32 each
    uint32_t k_dram = get_arg_val<uint32_t>(1);    // DRAM base: D key   codes, 1 uint32 each
    uint32_t out_dram = get_arg_val<uint32_t>(2);  // DRAM base: 8 quire limbs + readout + fp32 (uint32 each)
    uint32_t q_l1 = get_arg_val<uint32_t>(3);      // L1 scratch for the query codes
    uint32_t k_l1 = get_arg_val<uint32_t>(4);      // L1 scratch for the key codes
    uint32_t out_l1 = get_arg_val<uint32_t>(5);    // L1 scratch for the output words
    uint32_t d_dim = get_arg_val<uint32_t>(6);     // head dimension D (number of (q_i,k_i) pairs)
    uint32_t dbg_dram = get_arg_val<uint32_t>(7);  // [DEBUG] echo 2*D codes: q[] then k[]

    constexpr uint32_t WORD = sizeof(uint32_t);

    // ---- 1. NoC-read BOTH vectors DRAM -> L1 (one bulk read each) -----------
    // Single-page marshalling discipline (proven in quire_dot_baby.cpp): each
    // input buffer is ONE page (page_size == d_dim*WORD), bank-base-aligned, the
    // codes one contiguous blob — one bulk read, no per-page stride to get wrong.
    InterleavedAddrGen<true> q_gen = {.bank_base_address = q_dram, .page_size = d_dim * WORD};
    InterleavedAddrGen<true> k_gen = {.bank_base_address = k_dram, .page_size = d_dim * WORD};
    noc_async_read(q_gen.get_noc_addr(0), q_l1, d_dim * WORD);
    noc_async_read(k_gen.get_noc_addr(0), k_l1, d_dim * WORD);
    noc_async_read_barrier();  // both vectors resident in L1

    // ---- 1b. DEBUG (removable): echo q[] then k[] back so host can confirm ---
    InterleavedAddrGen<true> dbg_gen = {.bank_base_address = dbg_dram, .page_size = 2u * d_dim * WORD};
    noc_async_write(q_l1, dbg_gen.get_noc_addr(0), d_dim * WORD);                              // q[] -> [0..D)
    noc_async_write(k_l1, dbg_gen.get_noc_addr(0) + (uint64_t)(d_dim * WORD), d_dim * WORD);   // k[] -> [D..2D)
    noc_async_write_barrier();

    volatile uint32_t* qc = (volatile uint32_t*)q_l1;
    volatile uint32_t* kc = (volatile uint32_t*)k_l1;

    // ---- 2a. EXACT 256-bit quire DOT (the crown jewel) ----------------------
    //   q256 = 0; for each dim i:  q256 += bp16_prod_to_q256(q_i, k_i)  [exact]
    // bp16_prod_to_q256 forms sign_q^sign_k * (M_q*M_k) * 2^(E2_q+E2_k) and places
    // it into the 256-bit quire with NO per-product rounding (bp16_quire.h:43-60),
    // so the whole contraction is exact; q256_add is exact ripple-carry. The
    // massive-activation outlier products (+/-2^33 here) are represented EXACTLY
    // and cancel EXACTLY — the signal survives. Single readout rounds once.
    unsigned q256[QLIMBS] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};  // quire256_zero()
    for (uint32_t i = 0; i < d_dim; ++i) {
        unsigned contrib[QLIMBS];
        int a = (int)(qc[i] & 0xFFFFu);
        int b = (int)(kc[i] & 0xFFFFu);
        bp16_prod_to_q256(a, b, contrib);  // EXACT q_i*k_i into 8 limbs
        q256_add(q256, contrib);            // exact 256-bit ripple-carry add
    }
    int readout = bp16_encode_quire256(q256);  // the SINGLE rounding (truncation)

    // ---- 2b. GENUINE IEEE-754 fp32 NAIVE dot (the honest baseline) ----------
    //   acc = 0.0f; for each dim i:  acc = fp32_add(acc, fp32(q_i)*fp32(k_i))
    // Same SEQUENTIAL naive dot a float attention kernel does, in DIM ORDER. The
    // per-dim product is exact; the LOSS is in the running sum — once acc holds a
    // +2^33 outlier product, the O(1) signal terms fall below its ULP (~2^10) and
    // are swallowed, then the -2^33 outlier returns acc to ~0 having recorded none
    // of the signal: catastrophic cancellation of massive activations.
    uint32_t facc = 0x00000000u;  // +0.0f
    for (uint32_t i = 0; i < d_dim; ++i) {
        int a = (int)(qc[i] & 0xFFFFu);
        int b = (int)(kc[i] & 0xFFFFu);
        uint32_t prod = bp16_prod_to_fp32_bits(a, b);  // EXACT product q_i*k_i as fp32 bits
        facc = fp32_add(facc, prod);                    // genuine fp32 add (RTNE) -- the only error
    }

    // ---- 3. write 8 quire limbs + bp16 readout + fp32 bits back: L1 -> DRAM --
    // out_dram layout (uint32 each, LE): [q0..q7, readout, fp32_bits].
    volatile uint32_t* out = (volatile uint32_t*)out_l1;
    for (int i = 0; i < QLIMBS; ++i) out[i] = q256[i];
    out[QLIMBS] = (uint32_t)(readout & 0xFFFF);
    out[QLIMBS + 1] = facc;

    constexpr uint32_t N_OUT = QLIMBS + 2;  // 10 words
    InterleavedAddrGen<true> out_gen = {.bank_base_address = out_dram, .page_size = N_OUT * WORD};
    noc_async_write(out_l1, out_gen.get_noc_addr(0), N_OUT * WORD);
    noc_async_write_barrier();
}
