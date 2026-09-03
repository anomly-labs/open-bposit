// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// quire_matmul_op.cpp — the ttnn-op data-movement kernel for Anomly's EXACT
// 256-bit Kulisch quire b-posit16 MATMUL  C[M,N] = A[M,K] · B[K,N], operating on
// normal ttnn ROW_MAJOR uint32 tensors (b-posit16 codes carried in the low 16
// bits of each word). Validated layout: see programming_examples/
// bposit_quire_matmul_ttnn (72/72 bit-exact on Blackhole, real silicon).
//
// The per-element exact-quire MATH is byte-for-byte the proven scalar kernel's
// (bp16_prod_to_q256 / q256_add / bp16_encode_quire256). This kernel is the
// gate-free production variant: the host's split_work_to_cores over M hands each
// core a [row_start,row_end) ROW slice; each core reads all of A,B into L1, then
// computes a WHOLE output row and writes it as ONE aligned full-row noc write.
//
// DRAM 64-byte transfer granularity (Blackhole): per-row reads/writes are exact
// only when the row stride >= 64B. Production dims (N,K >= 16 => rows >= 64B)
// satisfy this, so both operands are read row-paged. (Single-element 4B writes are
// dropped by the 16B write alignment — hence the full-row write unit.)
#include <cstdint>

// Freestanding RV32IM-clean exact-quire numerics (proven byte-identical
// x86 == qemu-rv32 == ttsim-BRISC); shared with all the bposit quire kernels.
#include "../../../../kernel/bp16_quire.h"   // bp16_prod_to_q256, q256_add
#include "../../../../kernel/bp16_encode.h"  // bp16_encode_quire256
#include "../../../../kernel/bp32_encode.h"  // bp32_encode_quire256

#define QLIMBS 8  // 8x uint32 = 256-bit quire, fraction point at bit 96

void kernel_main() {
    uint32_t a_dram = get_arg_val<uint32_t>(0);     // DRAM base: M*K a-codes (row-major)
    uint32_t b_dram = get_arg_val<uint32_t>(1);     // DRAM base: K*N b-codes (row-major)
    uint32_t c_dram = get_arg_val<uint32_t>(2);     // DRAM base: M*N readouts (row-major)
    uint32_t m_dim = get_arg_val<uint32_t>(3);      // M
    uint32_t k_dim = get_arg_val<uint32_t>(4);      // K
    uint32_t n_dim = get_arg_val<uint32_t>(5);      // N
    // Work is split into UNITS = (row, column-chunk). unit u -> row u/nchunks,
    // chunk u%nchunks covering columns [chunk*chunk_cols, min(+chunk_cols, N)).
    // Splitting columns lets small-M matmuls use more of the grid; each chunk is a
    // contiguous 16B-aligned sub-range of its row (chunk_cols is a multiple of 4 and
    // N % 4 == 0), so concurrent cores writing different chunks of the same row page
    // never violate the DRAM 16-byte write alignment.
    uint32_t unit_start = get_arg_val<uint32_t>(6);    // first work unit this core owns
    uint32_t unit_end = get_arg_val<uint32_t>(7);      // one-past-last work unit (exclusive)
    uint32_t nchunks = get_arg_val<uint32_t>(8);       // column chunks per row
    uint32_t chunk_cols = get_arg_val<uint32_t>(9);    // columns per chunk (multiple of 4)
    uint32_t readout_bp32 = get_arg_val<uint32_t>(10); // 0 = bp16 readout (low 16b), 1 = bp32 (full 32b)

    // Per-core L1 scratch comes from circular buffers the program factory reserves
    // (cb 0 = A codes, cb 1 = B codes, cb 2 = one output row). Used as plain L1
    // scratch (not a producer/consumer pipe), so get_write_ptr gives the base.
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_c = 2;
    uint32_t a_l1 = get_write_ptr(cb_a);
    uint32_t b_l1 = get_write_ptr(cb_b);
    uint32_t c_l1 = get_write_ptr(cb_c);

    constexpr uint32_t WORD = sizeof(uint32_t);

    // ---- read A and B (row-paged interleaved, ttnn ROW_MAJOR) into L1 --------
    InterleavedAddrGen<true> a_gen = {.bank_base_address = a_dram, .page_size = k_dim * WORD};
    for (uint32_t i = 0; i < m_dim; ++i) {
        noc_async_read(a_gen.get_noc_addr(i), a_l1 + i * k_dim * WORD, k_dim * WORD);
    }
    InterleavedAddrGen<true> b_gen = {.bank_base_address = b_dram, .page_size = n_dim * WORD};
    for (uint32_t k = 0; k < k_dim; ++k) {
        noc_async_read(b_gen.get_noc_addr(k), b_l1 + k * n_dim * WORD, n_dim * WORD);
    }
    noc_async_read_barrier();

    volatile uint32_t* a_codes = (volatile uint32_t*)a_l1;  // row-major M x K
    volatile uint32_t* b_codes = (volatile uint32_t*)b_l1;  // row-major K x N
    volatile uint32_t* cout = (volatile uint32_t*)c_l1;     // one output row scratch

    InterleavedAddrGen<true> c_gen = {.bank_base_address = c_dram, .page_size = n_dim * WORD};

    // ---- exact-quire MATMUL over this core's WORK UNITS (row, col-chunk) -----
    for (uint32_t u = unit_start; u < unit_end; ++u) {
        uint32_t i = u / nchunks;             // output row
        uint32_t c = u % nchunks;             // column chunk
        uint32_t j0 = c * chunk_cols;         // first column of this chunk (multiple of 4)
        uint32_t j1 = j0 + chunk_cols;
        if (j1 > n_dim) {
            j1 = n_dim;                       // last chunk may be short (N % 4 == 0 keeps it 16B-aligned)
        }
        for (uint32_t j = j0; j < j1; ++j) {
            unsigned q[QLIMBS] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};  // quire256_zero()
            for (uint32_t k = 0; k < k_dim; ++k) {
                int ac = (int)(a_codes[i * k_dim + k] & 0xFFFFu);  // A[i,k]
                int bc = (int)(b_codes[k * n_dim + j] & 0xFFFFu);  // B[k,j]
                bp16_madd_q256(q, ac, bc);   // fused windowed MADD: EXACT a*b into the quire (bit-exact)
            }
            // Readout: bp32 keeps far more of the exact accumulation (full 32-bit code);
            // bp16 (low 16 bits) is the compact default. The quire above is identical.
            cout[j - j0] = readout_bp32 ? (uint32_t)bp32_encode_quire256(q)
                                        : (uint32_t)(bp16_encode_quire256(q) & 0xFFFF);
        }
        // One aligned write of this chunk: row i, columns [j0,j1) at offset j0*WORD.
        noc_async_write(c_l1, c_gen.get_noc_addr(i) + (uint64_t)j0 * WORD, (j1 - j0) * WORD);
        noc_async_write_barrier();
    }
}
