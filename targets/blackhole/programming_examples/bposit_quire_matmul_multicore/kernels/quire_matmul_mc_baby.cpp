// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: Apache-2.0
//
// quire_matmul_mc_baby.cpp — the MULTI-CORE MATMUL kernel: Anomly's EXACT
// 256-bit Kulisch quire b-posit16 MATMUL  C[M,N] = A[M,K] · B[K,N]  (M=N=8,
// K=16) distributed across MANY Blackhole baby (RISC-V) data-movement cores, one
// (or a few) output element(s) per core, bit-exact vs the generated golden
// (quire_matmul_mc_golden_cases.h).
//
// This is the PARALLEL generalisation of quire_matmul_baby.cpp. The PER-ELEMENT
// math is byte-for-byte the SAME proven scalar exact-quire dot — the ONLY new
// thing is the dispatch: the host runs THIS identical kernel on a CoreRange grid
// and hands each core a disjoint [out_start, out_end) slice of the M*N output
// indices via SetRuntimeArgs. Each output is independent (no cross-core
// reduction), so the parallelisation is embarrassingly parallel and provably
// preserves the single-core result.
//
//   for each flat output index o in [out_start, out_end):
//     i = o / N;  j = o % N
//     q = 0; for k:  q += bp16_prod_to_q256(A[i,k], B[k,j])   [exact, no rounding]
//     C[o] = bp16_encode_quire256(q)   [the single truncating readout]
//
// Each per-(i,k,j) product is EXACT (no per-product rounding); q256_add is exact
// 256-bit ripple-carry, so the K-accumulation is order-independent; only the
// per-output quire->bp16 readout rounds.
//
// MARSHALLING (do NOT regress the proven rule):
//   - INPUTS A and B are READ-ONLY and shared by every core. Each is ONE page
//     (page_size == whole matrix), so each core does ONE bulk noc_async_read of A
//     and ONE of B into its own L1 scratch — the exact single-page/bulk pattern
//     from quire_matmul_baby.cpp. (A=M*K, B=K*N codes — ~128 words each, trivially
//     L1-resident on every core.)
//   - OUTPUT is a per-ELEMENT-paged buffer (page_size == 1 word, M*N pages). Each
//     core writes ONLY its own output slot(s) via get_noc_addr(o), so concurrent
//     cores never collide and never need a shared-page read-modify-write.
//   - The GATE quire (8 limbs of one designated output) is written by the single
//     core that owns it, to a separate 8-word gate buffer (one page).

#include <cstdint>

// === REUSED golden-exact numerics (absolute path; do NOT reimplement) ========
// The SAME freestanding RV32IM-clean headers the single-core reduce/dot/matmul
// kernels use (proven byte-identical x86 == qemu-rv32 == ttsim-BRISC).
// bp16_prod_to_q256 is the EXACT bp16xbp16 FMA-into-quire primitive
// (bp16_quire.h:43-60).
//
// TODO(include-path): same caveat as quire_matmul_baby.cpp — if the kernel-compile
//   sandbox rejects the absolute #include, pass the dir via the CreateKernel
//   defines/-I mechanism or have run_on_silicon.sh copy the headers into this
//   kernels/ dir. Best guess: absolute include works (freestanding, proven on the
//   single-core path).
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
    // bank-base-aligned and the codes are one contiguous blob — one bulk read,
    // no per-page stride to get wrong. Identical to quire_matmul_baby.cpp; every
    // core reads its OWN private copy into its OWN L1 (inputs are read-only).
    InterleavedAddrGen<true> a_gen = {.bank_base_address = a_dram, .page_size = a_count * WORD};
    InterleavedAddrGen<true> b_gen = {.bank_base_address = b_dram, .page_size = b_count * WORD};
    noc_async_read(a_gen.get_noc_addr(0), a_l1, a_count * WORD);
    noc_async_read(b_gen.get_noc_addr(0), b_l1, b_count * WORD);
    noc_async_read_barrier();  // both operand matrices resident in this core's L1

    // ---- 2. EXACT scalar quire MATMUL over THIS core's output slice ----------
    // Byte-for-byte the same inner math as the single-core matmul kernel; only the
    // OUTPUT index range differs per core.
    volatile uint32_t* a_codes = (volatile uint32_t*)a_l1;  // row-major M x K
    volatile uint32_t* b_codes = (volatile uint32_t*)b_l1;  // row-major K x N
    volatile uint32_t* cout = (volatile uint32_t*)c_l1;     // this core's readout scratch

    // The output buffer is 1-word-per-page, so each output index is its own page;
    // write each computed readout straight to page o via get_noc_addr(o).
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
        // stash in L1 then write this single word to output page o.
        cout[0] = (uint32_t)(readout & 0xFFFF);
        noc_async_write(c_l1, c_gen.get_noc_addr(o), WORD);
        noc_async_write_barrier();               // serialise this core's own writes

        if (o == gate_o) {                        // the one core that owns the gate captures its quire
            for (int l = 0; l < QLIMBS; ++l) gate_q[l] = q[l];
            own_gate = true;
        }
    }

    // ---- 3. the single gate-owning core writes the gated output's full quire --
    // gate_dram is a separate 8-word (one-page) buffer; only the owning core writes.
    if (own_gate) {
        volatile uint32_t* gout = (volatile uint32_t*)c_l1;  // reuse c_l1 scratch (>=8 words)
        for (int l = 0; l < QLIMBS; ++l) gout[l] = gate_q[l];
        InterleavedAddrGen<true> gate_gen = {.bank_base_address = gate_dram, .page_size = QLIMBS * WORD};
        noc_async_write(c_l1, gate_gen.get_noc_addr(0), QLIMBS * WORD);
        noc_async_write_barrier();
    }
}
