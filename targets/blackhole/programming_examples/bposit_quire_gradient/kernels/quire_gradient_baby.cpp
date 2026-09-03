// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: Apache-2.0
//
// quire_gradient_baby.cpp — the -for-TRAINING kernel: on ONE Blackhole baby
// (RISC-V data-movement) core, accumulate a per-parameter GRADIENT g = Sum_t d_t
// (a sum of many tiny per-step contributions) TWO ways and write both back:
//
//   (1) EXACT 256-bit Kulisch quire: q = Sum_t bp16_to_q256(code_t), no rounding in
//       the accumulate, single truncating readout (the proven reduce path, reusing
//       kernel/bp16_quire.h + bp16_encode.h verbatim — the same code already
//       bit-exact vs golden on this silicon in bposit_quire_reduce / _vs_float).
//
//   (2) GENUINE IEEE-754 fp32 running sum: acc = fp32_add(acc, fp32(value_t)),
//       round-to-nearest-ties-to-even, in PURE INTEGER arithmetic
//       (kernels/ieee_softfp32.h, reused verbatim from bposit_quire_vs_float, host-
//       verified bit-exact vs native C `float`) so it needs no FPU / libgcc soft-
//       float on the baby core.
//
// The training point: a large early contribution inflates the running accumulator to
// magnitude M, so each subsequent tiny per-step gradient d_t falls below ULP(M) and
// is SWALLOWED by the fp32 running sum (the swamping / gradient-underflow problem in
// large-batch training). The exact quire keeps every d_t, so it recovers the tiny-
// tail gradient mass fp32 drops. The host (bposit_quire_gradient.cpp) gates quire ==
// exact gradient (error 0) while fp32 loses a measured fraction of g — on real
// silicon. Marshalling is the proven single-page / bulk-NoC discipline from
// quire_reduce_baby.cpp (one page per buffer, one bulk read/write).

#include <cstdint>

// === REUSED golden-exact b-posit numerics (absolute path; do NOT reimplement) =
// Freestanding RV32IM-clean headers, proven byte-identical x86 == qemu-rv32 ==
// ttsim-BRISC, and already bit-exact on this Blackhole in bposit_quire_reduce.
#include "../../../kernel/bp16_quire.h"   // bp16_to_q256, q256_add
#include "../../../kernel/bp16_encode.h"  // bp16_encode_quire256
#include "../../../kernel/bp16_decode.h"  // bp16_decode (sign, M, E2)

// Pure-integer IEEE-754 fp32 add, reused verbatim from bposit_quire_vs_float. Absolute
// path, like the bp16 headers above, so it resolves regardless of whether the kernel-
// compile sandbox puts this kernels/ dir on the include search path.
#include "$TT_METAL_HOME/tt_metal/programming_examples/bposit_quire_vs_float/kernels/ieee_softfp32.h"

#define QLIMBS 8  // 8x uint32 = 256-bit quire, fraction point at bit 96

// Convert a b-posit16 code to its IEEE-754 binary32 bit pattern. The bp16 value is
// the dyadic sign * M * 2^E2 (bp16_decode); since |M| < 2^11 it fits the fp32 24-bit
// significand exactly, so this conversion is EXACT (no rounding) for every
// representable bp16 value — the fp32 baseline only loses signal in the *running sum*
// (tiny gradients below ULP(M)), never in the input conversion. That keeps the
// comparison honest: fp32's error is purely from accumulation, exactly as on a real
// fp32 training pipeline. (Identical to the host helper, kept in lockstep.)
static uint32_t bp16_code_to_fp32_bits(int code) {
    int sign;
    unsigned M;
    int E2;
    bp16_decode(code, &sign, &M, &E2);
    if (M == 0u) {
        return sign ? 0x80000000u : 0x00000000u;  // zero / NaR -> +/-0
    }
    int msb = 31;
    while (msb >= 0 && !((M >> msb) & 1u)) msb--;
    uint32_t sig;
    if (msb <= 23) {
        sig = M << (23 - msb);
    } else {
        sig = M >> (msb - 23);  // never happens for M < 2^11, but keep it total
    }
    int unbiased = E2 + msb;
    int biased = unbiased + 127;
    if (biased >= 0xFF) {
        return ((uint32_t)sign << 31) | 0x7F800000u;  // overflow -> Inf (won't happen for bp16)
    }
    if (biased <= 0) {
        return (uint32_t)sign << 31;  // far below bp16 range -> signed zero
    }
    uint32_t mant = sig & 0x007FFFFFu;  // drop the hidden bit
    return ((uint32_t)sign << 31) | ((uint32_t)biased << 23) | mant;
}

void kernel_main() {
    // ---- runtime args (set by the host SetRuntimeArgs, same order) ----------
    uint32_t in_dram = get_arg_val<uint32_t>(0);   // DRAM base: N codes, 1 uint32 each
    uint32_t out_dram = get_arg_val<uint32_t>(1);  // DRAM base: 8 quire limbs + readout + fp32 (uint32 each)
    uint32_t in_l1 = get_arg_val<uint32_t>(2);     // L1 scratch for the input codes
    uint32_t out_l1 = get_arg_val<uint32_t>(3);    // L1 scratch for the output words
    uint32_t n_codes = get_arg_val<uint32_t>(4);   // number of gradient contributions
    uint32_t dbg_dram = get_arg_val<uint32_t>(5);  // [DEBUG] echo the codes we read

    constexpr uint32_t WORD = sizeof(uint32_t);

    // ---- 1. NoC-read all input codes DRAM -> L1 (one bulk read) -------------
    // Single-page marshalling discipline (proven in quire_reduce_baby.cpp): the host
    // allocates in_dram as ONE page (page_size == in_bytes), so its base is bank-base-
    // aligned and the N codes are one contiguous blob — one bulk read, no per-page
    // InterleavedAddrGen stride to get wrong.
    InterleavedAddrGen<true> in_gen = {.bank_base_address = in_dram, .page_size = n_codes * WORD};
    noc_async_read(in_gen.get_noc_addr(0), in_l1, n_codes * WORD);
    noc_async_read_barrier();  // codes resident in L1

    // ---- 1b. DEBUG (removable): echo codes back so the host can confirm them --
    InterleavedAddrGen<true> dbg_gen = {.bank_base_address = dbg_dram, .page_size = n_codes * WORD};
    noc_async_write(in_l1, dbg_gen.get_noc_addr(0), n_codes * WORD);
    noc_async_write_barrier();

    volatile uint32_t* codes = (volatile uint32_t*)in_l1;

    // ---- 2a. EXACT 256-bit quire accumulate (the crown jewel) ---------------
    //   q = 0; for each gradient contribution:  q += bp16_to_q256(code)  [exact]
    // Every tiny per-step gradient is placed exactly into the 256-bit Kulisch field;
    // no contribution is ever below the accumulator's ULP, because the quire has no
    // ULP-relative-to-a-running-sum — it is a fixed-point register wide enough for
    // the full b-posit dynamic range. Nothing is swamped.
    unsigned q[QLIMBS] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};  // quire256_zero()
    for (uint32_t i = 0; i < n_codes; ++i) {
        unsigned contrib[QLIMBS];
        int code = (int)(codes[i] & 0xFFFFu);
        bp16_to_q256(code, contrib);  // exact placement into 256 bits
        q256_add(q, contrib);          // exact 256-bit ripple-carry add
    }
    int readout = bp16_encode_quire256(q);  // the SINGLE rounding (truncation)

    // ---- 2b. GENUINE IEEE-754 fp32 running sum (the honest baseline) --------
    //   acc = 0.0f; for each gradient contribution:  acc = fp32_add(acc, fp32(d_t))
    // Same SEQUENTIAL running-sum a float training accumulator does; once acc reaches
    // the running-sum magnitude M, each tiny d_t < ULP(M) adds nothing (swamped) — the
    // fp32 result freezes below the true gradient.
    uint32_t facc = 0x00000000u;  // +0.0f
    for (uint32_t i = 0; i < n_codes; ++i) {
        int code = (int)(codes[i] & 0xFFFFu);
        uint32_t fv = bp16_code_to_fp32_bits(code);
        facc = fp32_add(facc, fv);
    }

    // ---- 3. write 8 quire limbs + bp16 readout + fp32 bits back: L1 -> DRAM --
    // out_dram layout (uint32 each, LE): [q0..q7, readout, fp32_bits].
    volatile uint32_t* out = (volatile uint32_t*)out_l1;
    for (int i = 0; i < QLIMBS; ++i) out[i] = q[i];
    out[QLIMBS] = (uint32_t)(readout & 0xFFFF);
    out[QLIMBS + 1] = facc;

    constexpr uint32_t N_OUT = QLIMBS + 2;  // 10 words
    InterleavedAddrGen<true> out_gen = {.bank_base_address = out_dram, .page_size = N_OUT * WORD};
    noc_async_write(out_l1, out_gen.get_noc_addr(0), N_OUT * WORD);
    noc_async_write_barrier();
}
