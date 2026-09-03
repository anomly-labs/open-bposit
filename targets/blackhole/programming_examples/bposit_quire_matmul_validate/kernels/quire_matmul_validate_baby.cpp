// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: Apache-2.0
//
// quire_matmul_validate_baby.cpp — the FULL-VALIDATION MATMUL kernel: Anomly's
// EXACT 256-bit Kulisch quire b-posit16 MATMUL  C[M,N] = A[M,K] . B[K,N]
// (M=N=16, K=16) distributed across MANY Blackhole baby (RISC-V) data-movement
// cores, one (or a few) output element(s) per core. ONE dispatch computes ONE
// seed's matmul; the host re-uploads A,B and re-enqueues this SAME workload once
// per random seed, gating EVERY output of EVERY seed bit-exact vs the oracle
// golden (quire_matmul_validate_golden_cases.h).
//
// This is byte-for-byte the proven multi-core matmul kernel
// (bposit_quire_matmul_multicore/kernels/quire_matmul_mc_baby.cpp); the ONLY
// thing this example adds is the host-side SEED LOOP and full per-seed gate. The
// per-element math — exact bp16xbp16 product into a 256-bit quire, exact
// 256-bit add, single truncating readout — is unchanged and already on-silicon
// proven. Each output element C[i,j] is an INDEPENDENT exact-quire dot of length
// K (no cross-core reduction), so the M*N outputs are distributed one-(or-few)-
// per-core; the host hands each core a disjoint [out_start, out_end) slice.
//
//   for each flat output index o in [out_start, out_end):
//     i = o / N;  j = o % N
//     q = 0; for k:  q += bp16_prod_to_q256(A[i,k], B[k,j])   [exact, no rounding]
//     C[o] = bp16_encode_quire256(q)   [the single truncating readout]
//
// MARSHALLING (do NOT regress the proven rule):
//   - INPUTS A,B are READ-ONLY, ONE page each (page_size == whole matrix); each
//     core does ONE bulk noc_async_read of A and one of B into its own L1 — the
//     exact single-page/bulk pattern from quire_matmul_mc_baby.cpp.
//   - OUTPUT is a per-ELEMENT-paged buffer (page_size == 1 word, M*N pages); each
//     core writes ONLY its own output slot(s) via get_noc_addr(o), so concurrent
//     cores never collide.
//   - The GATE quire (8 limbs of the designated output) is written by the single
//     core that owns it, to a separate 8-word gate buffer (one page).

#include <cstdint>

// === REUSED golden-exact numerics (absolute path; do NOT reimplement) ========
// The SAME freestanding RV32IM-clean headers the single-/multi-core reduce/dot/
// matmul kernels use (proven byte-identical x86 == qemu-rv32 == ttsim-BRISC).
// bp16_prod_to_q256 is the EXACT bp16xbp16 FMA-into-quire primitive
// (bp16_quire.h:43-60).
#include "../../../kernel/bp16_quire.h"   // bp16_prod_to_q256, q256_add
#include "../../../kernel/bp16_encode.h"  // bp16_encode_quire256 (QFRAC=96, truncation)

#define QLIMBS 8  // 8x uint32 = 256-bit quire, fraction point at bit 96

void kernel_main() {
    // ---- runtime args (set by the host SetRuntimeArgs, same order) ----------
    uint32_t a_dram = get_arg_val<uint32_t>(0);     // DRAM base: M*K a-codes (row-major), shared
    uint32_t b_dram = get_arg_val<uint32_t>(1);     // DRAM base: K*N b-codes (row-major), shared
    uint32_t c_dram = get_arg_val<uint32_t>(2);     // DRAM base: M*N readouts, ONE word per page
    uint32_t a_l1 = get_arg_val<uint32_t>(3);       // L1 scratch for the A codes (per-core)
    uint32_t b_l1 = get_arg_val<uint32_t>(4);       // L1 scratch for the B codes (per-core)
    uint32_t c_l1 = get_arg_val<uint32_t>(5);       // L1 scratch for this core's output words
    uint32_t m_dim = get_arg_val<uint32_t>(6);      // M (rows of A / C)
    uint32_t k_dim = get_arg_val<uint32_t>(7);      // K (contraction length)
    uint32_t n_dim = get_arg_val<uint32_t>(8);      // N (cols of B / C)
    uint32_t out_start = get_arg_val<uint32_t>(9);  // first flat output index this core owns
    uint32_t out_end = get_arg_val<uint32_t>(10);   // one-past-last flat output index (exclusive)
    uint32_t gate_o = get_arg_val<uint32_t>(11);    // flat output index whose full quire is gated
    uint32_t gate_dram = get_arg_val<uint32_t>(12); // DRAM base: 8 quire limbs of output gate_o

    constexpr uint32_t WORD = sizeof(uint32_t);
    const uint32_t a_count = m_dim * k_dim;  // A is M x K
    const uint32_t b_count = k_dim * n_dim;  // B is K x N

    // ---- 1. NoC-read BOTH operand matrices DRAM -> L1 (one bulk read each) ---
    // Each input buffer is ONE page (page_size == count*WORD), so its base is
    // bank-base-aligned and the codes are one contiguous blob — one bulk read.
    // Every core reads its OWN private copy into its OWN L1 (inputs read-only).
    InterleavedAddrGen<true> a_gen = {.bank_base_address = a_dram, .page_size = a_count * WORD};
    InterleavedAddrGen<true> b_gen = {.bank_base_address = b_dram, .page_size = b_count * WORD};
    noc_async_read(a_gen.get_noc_addr(0), a_l1, a_count * WORD);
    noc_async_read(b_gen.get_noc_addr(0), b_l1, b_count * WORD);
    noc_async_read_barrier();  // both operand matrices resident in this core's L1

    // ---- 2. EXACT scalar quire MATMUL over THIS core's output slice ----------
    volatile uint32_t* a_codes = (volatile uint32_t*)a_l1;  // row-major M x K
    volatile uint32_t* b_codes = (volatile uint32_t*)b_l1;  // row-major K x N
    volatile uint32_t* cout = (volatile uint32_t*)c_l1;     // this core's readout scratch

    // The output buffer is 1-word-per-page; write each readout straight to page o.
    InterleavedAddrGen<true> c_gen = {.bank_base_address = c_dram, .page_size = WORD};

    unsigned gate_q[QLIMBS] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    bool own_gate = false;

    for (uint32_t o = out_start; o < out_end; ++o) {
        uint32_t i = o / n_dim;  // output row
        uint32_t j = o % n_dim;  // output col
        unsigned q[QLIMBS] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};  // quire256_zero()
        for (uint32_t k = 0; k < k_dim; ++k) {
            unsigned contrib[QLIMBS];
            int ac = (int)(a_codes[i * k_dim + k] & 0xFFFFu);  // A[i,k]
            int bc = (int)(b_codes[k * n_dim + j] & 0xFFFFu);  // B[k,j]
            bp16_prod_to_q256(ac, bc, contrib);  // EXACT a*b into 8 limbs (bp16_quire.h:43-60)
            q256_add(q, contrib);                 // exact 256-bit ripple-carry add
        }
        int readout = bp16_encode_quire256(q);    // quire -> bp16, truncation (bp16_encode.h)
        cout[0] = (uint32_t)(readout & 0xFFFF);
        noc_async_write(c_l1, c_gen.get_noc_addr(o), WORD);
        noc_async_write_barrier();               // serialise this core's own writes

        if (o == gate_o) {                        // the one core that owns the gate captures its quire
            for (int l = 0; l < QLIMBS; ++l) gate_q[l] = q[l];
            own_gate = true;
        }
    }

    // ---- 3. the single gate-owning core writes the gated output's full quire --
    if (own_gate) {
        volatile uint32_t* gout = (volatile uint32_t*)c_l1;  // reuse c_l1 scratch (>=8 words)
        for (int l = 0; l < QLIMBS; ++l) gout[l] = gate_q[l];
        InterleavedAddrGen<true> gate_gen = {.bank_base_address = gate_dram, .page_size = QLIMBS * WORD};
        noc_async_write(c_l1, gate_gen.get_noc_addr(0), QLIMBS * WORD);
        noc_async_write_barrier();
    }
}
