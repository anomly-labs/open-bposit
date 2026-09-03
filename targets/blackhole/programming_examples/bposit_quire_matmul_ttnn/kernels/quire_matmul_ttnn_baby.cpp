// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// quire_matmul_ttnn_baby.cpp — the EXACT-quire b-posit16 MATMUL kernel in the
// LAYOUT a ttnn ROW_MAJOR uint32 tensor uses: operands are *interleaved,
// row-paged* (A = M pages of K words, B = K pages of N words, C = M pages of N
// words) instead of the single-big-page layout of quire_matmul_mc_baby.cpp.
//
// This is the kernel I/O bridge for the native ttnn b-posit matmul op (task #5):
// the production op takes normal ttnn tensors (interleaved across DRAM banks,
// paged by row), NOT a single bulk page. The PER-ELEMENT exact-quire MATH here is
// BYTE-FOR-BYTE identical to quire_matmul_mc_baby.cpp — the ONLY change is the
// marshalling (how A/B are read in and C is written out). So a PASS here proves
// the proven exact-quire numerics survive the ttnn-native layout unchanged.
//
//   for each flat output index o in [out_start, out_end):
//     i = o / N;  j = o % N
//     q = 0; for k:  q += bp16_prod_to_q256(A[i,k], B[k,j])   [exact, no rounding]
//     C[i,j] = bp16_encode_quire256(q)   [the single truncating readout]
//
// MARSHALLING (ttnn ROW_MAJOR interleaved layout):
//   - INPUTS A (page=K words) and B (page=N words) are READ-ONLY and shared. Each
//     core reads ALL rows of A and B into its own L1 (one noc read per row/page;
//     the row pages may live in different DRAM banks — get_noc_addr(page) resolves
//     the bank). Tiny matrices → fully L1-resident on every core.
//   - OUTPUT C is row-paged (page=N words). Each core writes ONLY its own
//     element(s): the absolute address of C[i,j] is get_noc_addr(row i) + j*WORD.
//     Distinct cores write distinct byte addresses, so concurrent writes into the
//     same row page never collide and need no read-modify-write.
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
#include "../../../kernel/bp16_quire.h"  // bp16_prod_to_q256, q256_add
#include "../../../kernel/bp16_encode.h"  // bp16_encode_quire256 (QFRAC=96, truncation)

#define QLIMBS 8  // 8x uint32 = 256-bit quire, fraction point at bit 96

void kernel_main() {
    // ---- runtime args (set by the host SetRuntimeArgs, same order) ----------
    uint32_t a_dram = get_arg_val<uint32_t>(0);      // DRAM base: M*K a-codes (row-major), shared
    uint32_t b_dram = get_arg_val<uint32_t>(1);      // DRAM base: K*N b-codes (row-major), shared
    uint32_t c_dram = get_arg_val<uint32_t>(2);      // DRAM base: M*N readouts, ONE word per page
    uint32_t a_l1 = get_arg_val<uint32_t>(3);        // L1 scratch for the A codes (per-core)
    uint32_t b_l1 = get_arg_val<uint32_t>(4);        // L1 scratch for the B codes (per-core)
    uint32_t c_l1 = get_arg_val<uint32_t>(5);        // L1 scratch for this core's output words
    uint32_t m_dim = get_arg_val<uint32_t>(6);       // M (rows of A / C)
    uint32_t k_dim = get_arg_val<uint32_t>(7);       // K (contraction length)
    uint32_t n_dim = get_arg_val<uint32_t>(8);       // N (cols of B / C)
    uint32_t out_start = get_arg_val<uint32_t>(9);   // first flat output index this core owns
    uint32_t out_end = get_arg_val<uint32_t>(10);    // one-past-last flat output index (exclusive)
    uint32_t gate_o = get_arg_val<uint32_t>(11);     // flat output index whose full quire is gated
    uint32_t gate_dram = get_arg_val<uint32_t>(12);  // DRAM base: 8 quire limbs of output gate_o

    constexpr uint32_t WORD = sizeof(uint32_t);
    const uint32_t a_count = m_dim * k_dim;  // A is M x K
    const uint32_t b_count = k_dim * n_dim;  // B is K x N

    // ---- 1. NoC-read BOTH operand matrices DRAM -> L1 (one bulk read each) ---
    // Each input buffer is ONE page (page_size == count*WORD), so its base is
    // bank-base-aligned and the codes are one contiguous blob — one bulk read,
    // no per-page stride to get wrong. Identical to quire_matmul_baby.cpp; every
    // core reads its OWN private copy into its OWN L1 (inputs are read-only).
    // ttnn ROW_MAJOR layout: A is M pages of K words, B is K pages of N words.
    // Read every row/page into L1 contiguously (row-major), so the inner-loop
    // indexing a_codes[i*K+k] / b_codes[k*N+j] below is unchanged.
    // KEY SILICON FINDING (Blackhole DRAM has a 64-byte transfer granularity):
    // a per-row noc read/write works ONLY when the row stride is >= 64 bytes. Here
    // A's rows are K=16 codes = 64 bytes (exactly aligned) so A is read ROW-PAGED,
    // one page per row, proving the ttnn ROW_MAJOR interleaved read path is exact.
    // B's rows are N=8 codes = 32 bytes (sub-granularity) so B uses the bulk
    // single-page read as a fallback. In a real b-posit matmul the operand
    // row-stride (4*N or 4*K bytes) far exceeds 64B (N,K >= 16), so BOTH operands
    // are read row-paged on the production path; the 32B case is a tiny-dim edge.
    InterleavedAddrGen<true> a_gen = {.bank_base_address = a_dram, .page_size = k_dim * WORD};
    for (uint32_t i = 0; i < m_dim; ++i) {
        noc_async_read(a_gen.get_noc_addr(i), a_l1 + i * k_dim * WORD, k_dim * WORD);  // row-paged
    }
    InterleavedAddrGen<true> b_gen = {.bank_base_address = b_dram, .page_size = b_count * WORD};
    noc_async_read(b_gen.get_noc_addr(0), b_l1, b_count * WORD);  // bulk (32B rows)
    noc_async_read_barrier();                                     // both operand matrices resident in this core's L1

    // ---- 2. EXACT scalar quire MATMUL over THIS core's output slice ----------
    // Byte-for-byte the same inner math as the single-core matmul kernel; only the
    // OUTPUT index range differs per core.
    volatile uint32_t* a_codes = (volatile uint32_t*)a_l1;  // row-major M x K
    volatile uint32_t* b_codes = (volatile uint32_t*)b_l1;  // row-major K x N
    volatile uint32_t* cout = (volatile uint32_t*)c_l1;     // this core's readout scratch

    // ttnn ROW_MAJOR output: C is row-paged (page=N words). Element C[i,j] lives at
    // get_noc_addr(row i) + j*WORD; distinct cores write distinct addresses.
    InterleavedAddrGen<true> c_gen = {.bank_base_address = c_dram, .page_size = n_dim * WORD};

    unsigned gate_q[QLIMBS] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    bool own_gate = false;

    // ROW-PARALLEL over [out_start,out_end) = this core's ROW range (out_start/end
    // are ROW indices here, set by the host's split_work_to_cores over M). Each
    // core computes a WHOLE row of N readouts into L1, then writes the entire row
    // page in ONE aligned full-page noc write. This respects the DRAM 16-byte
    // write-alignment rule that single-element (4-byte, arbitrary-offset) writes
    // violate — the ttnn ROW_MAJOR layout pages by row, so a full-row write is the
    // natural aligned unit. Per-element exact-quire math is byte-identical.
    for (uint32_t i = out_start; i < out_end; ++i) {                // i = output ROW owned by this core
        for (uint32_t j = 0; j < n_dim; ++j) {                      // all N columns of the row
            unsigned q[QLIMBS] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};  // quire256_zero()
            for (uint32_t k = 0; k < k_dim; ++k) {
                unsigned contrib[QLIMBS];
                int ac = (int)(a_codes[i * k_dim + k] & 0xFFFFu);  // A[i,k]
                int bc = (int)(b_codes[k * n_dim + j] & 0xFFFFu);  // B[k,j]
                bp16_prod_to_q256(ac, bc, contrib);                // EXACT a*b into 8 limbs (bp16_quire.h:43-60)
                q256_add(q, contrib);                              // exact 256-bit ripple-carry add
            }
            cout[j] = (uint32_t)(bp16_encode_quire256(q) & 0xFFFF);  // build the full row in L1
            if (i * n_dim + j == gate_o) {                           // the core that owns the gate captures its quire
                for (int l = 0; l < QLIMBS; ++l) {
                    gate_q[l] = q[l];
                }
                own_gate = true;
            }
        }
        noc_async_write(c_l1, c_gen.get_noc_addr(i), n_dim * WORD);  // ONE aligned full-row write
        noc_async_write_barrier();                                   // serialise this core's own writes
    }

    // ---- 3. the single gate-owning core writes the gated output's full quire --
    // gate_dram is a separate 8-word (one-page) buffer; only the owning core writes.
    if (own_gate) {
        volatile uint32_t* gout = (volatile uint32_t*)c_l1;  // reuse c_l1 scratch (>=8 words)
        for (int l = 0; l < QLIMBS; ++l) {
            gout[l] = gate_q[l];
        }
        InterleavedAddrGen<true> gate_gen = {.bank_base_address = gate_dram, .page_size = QLIMBS * WORD};
        noc_async_write(c_l1, gate_gen.get_noc_addr(0), QLIMBS * WORD);
        noc_async_write_barrier();
    }
}
