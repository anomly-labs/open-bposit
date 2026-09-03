// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: Apache-2.0
//
// cross_entropy_baby.cpp — cross-entropy H(p,q) = -Sum_i p_i*log2(q_i),
// the standard classification training loss, run scalar on ONE Blackhole baby
// (RISC-V) data-movement core, bit-exact vs the generated golden
// (cross_entropy_golden_cases.h, itself drawn verbatim from golden/cross_entropy.json).
//
// THE ML PRIMITIVE:  for a target distribution p and a model/predicted
//   distribution q over the same alphabet of N classes, cross-entropy
//     H(p,q) = -Sum_i p_i * log2(q_i)
//   is the per-sample loss minimized by every softmax classifier. KL(p||q) =
//   H(p,q) - H(p) and Shannon H(p) = H(p,p) are the sibling primitives.
//
// THE NUMERICS (why this is the crown jewel, not a float dot):
//   log2q_i = bposit16_log2(q_i)           (HOST-precomputed; passed in as operand)
//   term    = bposit16_mul(p_i, log2q_i)   (EXACT bp16 product = p_i*log2(q_i))
//   H_q    += quire(term)                  (EXACT 256-bit accumulation, QFRAC=96)
//   H_q     = negate(H_q)                  (H(p,q) = -Sum, so positive loss)
//   H_bp32  = bp32_encode_quire256(H_q)    (the SINGLE rounding: readout only)
//   The accumulation is in the EXACT 256-bit Kulisch quire, so it is
//   order-independent AND the tiny-probability tails (q_i small => log2(q_i)
//   large-negative) are summed with FULL fixed-point precision — a float dot
//   silently drops those tail terms once the partial sum dominates, which is
//   exactly the regime cross-entropy cares about. This exact-quire SUM is the
//   load-bearing, rounding-stable result and it runs 100% ON-DEVICE.
//
// DEVICE/HOST SPLIT (honest):  the per-element log2(q_i) is precomputed on the
//   HOST (bposit16_log2) and handed to this kernel as the second bp16 operand
//   (log2q[]). The reason is hardware, not numeric: the full 65536-entry log2 LUT
//   does NOT fit the BRISC baby core's tiny local DATA region (~0x11d0 bytes) and
//   big-L1 is a separate CB/buffer address space, not kernel .data — a large
//   static LUT in the kernel TU overflows the .elf segment at load. With log2
//   precomputed, the device kernel is PURE EXACT-QUIRE: bp16 multiply +
//   256-bit-quire accumulate + negate + bposit32 readout. The host-precomputed
//   log2 is itself the canonical oracle (bposit16_log2 == the LUT entry), so the
//   result is bit-exact vs golden regardless of the split.
//
// This is the elementwise-then-reduce sibling of the causet/holographic kernels.
// Marshalling is the SAME proven pattern as causet/holographic/quire_reduce/dot:
// ONE page per buffer (page_size == whole buffer), one bulk noc_async_read per
// operand, one bulk noc_async_write out. An input-readback diagnostic echoes
// P[] then LOG2Q[] back to dbg DRAM so the host can confirm both operands arrived.

#include <cstdint>

// === REUSED golden-exact numerics (absolute path; do NOT reimplement) ========
// The SAME freestanding RV32IM-clean headers the proven cross_entropy_kernel.c
// #includes and the causet/holographic/reduce/dot/matmul kernels use
// (byte-identical x86 == qemu-rv32 == ttsim-BRISC). bp16_mul.h::bposit16_mul is
// the exact p*log2(q) product; bp16_quire.h::{bp16_to_q256,q256_add,q256_negate}
// carry the entropy sum exactly; bp32_encode.h::bp32_encode_quire256 is the
// quire->bposit32 readout. NOTE: the log2 LUT (bp16_log2_lut.h) is deliberately
// NOT included here — log2(q_i) is host-precomputed and arrives as the log2q[]
// operand (see DEVICE/HOST SPLIT above), so the kernel stays tiny + LUT-free.
//
// TODO(include-path): same caveat as the causet/holographic kernels — if the
//   kernel-compile sandbox rejects the absolute #include, pass the dir via the
//   CreateKernel defines/-I mechanism or have run_on_silicon.sh copy the headers
//   into kernels/. Best guess: absolute include works (freestanding, no libc).
#include "../../../kernel/bp16_quire.h"     // bp16_to_q256, q256_add, q256_negate
#include "../../../kernel/bp16_mul.h"       // bposit16_mul
#include "../../../kernel/bp32_encode.h"    // bp32_encode_quire256

#define QLIMBS 8  // 8x uint32 = 256-bit quire, fraction point at bit 96

void kernel_main() {
    // ---- runtime args (set by the host SetRuntimeArgs, same order) ----------
    uint32_t p_dram = get_arg_val<uint32_t>(0);       // DRAM base: N target probs p_i, 1 uint32 each
    uint32_t log2q_dram = get_arg_val<uint32_t>(1);   // DRAM base: N host-precomputed log2(q_i), 1 uint32 each
    uint32_t out_dram = get_arg_val<uint32_t>(2);     // DRAM base: 8 negated-quire limbs + 1 bp32 readout
    uint32_t p_l1 = get_arg_val<uint32_t>(3);         // L1 scratch for the P codes
    uint32_t log2q_l1 = get_arg_val<uint32_t>(4);     // L1 scratch for the LOG2Q codes
    uint32_t out_l1 = get_arg_val<uint32_t>(5);       // L1 scratch for the output words
    uint32_t n_dim = get_arg_val<uint32_t>(6);        // N (number of classes)
    uint32_t dbg_dram = get_arg_val<uint32_t>(7);     // [DEBUG] echo 2*N codes: P[] then LOG2Q[]

    constexpr uint32_t WORD = sizeof(uint32_t);
    const uint32_t n = n_dim;
    const uint32_t prob_bytes = n * WORD;             // N codes per operand

    // ---- 1. NoC-read BOTH operands DRAM -> L1 (one bulk read each) -----------
    InterleavedAddrGen<true> p_gen = {.bank_base_address = p_dram, .page_size = prob_bytes};
    InterleavedAddrGen<true> log2q_gen = {.bank_base_address = log2q_dram, .page_size = prob_bytes};
    noc_async_read(p_gen.get_noc_addr(0), p_l1, prob_bytes);
    noc_async_read(log2q_gen.get_noc_addr(0), log2q_l1, prob_bytes);
    noc_async_read_barrier();  // both operands resident in L1

    // ---- 1b. DEBUG (removable): echo P[] then LOG2Q[] back to dbg_dram -------
    const uint32_t dbg_count = 2u * n;
    InterleavedAddrGen<true> dbg_gen = {.bank_base_address = dbg_dram, .page_size = dbg_count * WORD};
    noc_async_write(p_l1, dbg_gen.get_noc_addr(0), prob_bytes);                                  // P[]     -> [0..N)
    noc_async_write(log2q_l1, dbg_gen.get_noc_addr(0) + (uint64_t)prob_bytes, prob_bytes);       // LOG2Q[] -> [N..2N)
    noc_async_write_barrier();
    // ---- end DEBUG ----

    // ---- 2. EXACT scalar cross-entropy accumulation (the crown jewel) -------
    //   for each class i:
    //     term = bposit16_mul(p_i, log2q_i)        (EXACT p_i*log2(q_i))
    //     H_q += quire(term)                       (EXACT 256-bit accumulation)
    //   then H_q = -H_q  (positive cross-entropy)  and  H_bp32 = encode(H_q).
    //   log2q_i is the host-precomputed bposit16_log2(q_i); the EXACT quire SUM
    //   (the rounding-stable, tail-lossless part) is entirely on-device here.
    volatile uint32_t* p_codes = (volatile uint32_t*)p_l1;      // length N
    volatile uint32_t* log2q_codes = (volatile uint32_t*)log2q_l1;  // length N
    volatile uint32_t* out = (volatile uint32_t*)out_l1;

    unsigned acc[QLIMBS] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    unsigned contrib[QLIMBS];
    for (uint32_t i = 0; i < n; ++i) {
        int pi = (int)(p_codes[i] & 0xFFFFu);
        int log2qi = (int)(log2q_codes[i] & 0xFFFFu);    // host-precomputed log2(q_i)
        int term = bposit16_mul(pi, log2qi);             // p_i * log2(q_i)  EXACT
        bp16_to_q256(term, contrib);                     // term -> 256-bit quire
        q256_add(acc, contrib);                          // exact accumulation
    }
    q256_negate(acc);                                    // H(p,q) = -Sum  (positive loss)

    // ---- 3. emit: 8 negated-quire limbs (primary gate) then the bp32 readout -
    for (int l = 0; l < QLIMBS; ++l) out[l] = acc[l];
    out[QLIMBS] = bp32_encode_quire256(acc);             // quire -> bposit32 (single rounding)

    // ---- 4. write all output words back: L1 -> DRAM (one bulk write) --------
    const uint32_t n_out = QLIMBS + 1u;                  // 8 quire limbs + 1 bp32 code
    InterleavedAddrGen<true> out_gen = {.bank_base_address = out_dram, .page_size = n_out * WORD};
    noc_async_write(out_l1, out_gen.get_noc_addr(0), n_out * WORD);
    noc_async_write_barrier();

    // TODO(parallel): the N classes are an embarrassingly-parallel reduction —
    //   split the class range across TRISC0/1/2 (each a partial quire), then one
    //   exact q256_add tree-merge (the quire add is associative, so the merge is
    //   bit-identical to the scalar order). The SFPU 32-lane lift (one quire per
    //   lane) is the shared Phase-2c with the holographic/matmul/causet kernels.
}
