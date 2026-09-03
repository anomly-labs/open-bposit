// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// quire_matmul_op_streamed.cpp — UNBOUNDED-operand variant of the exact-quire matmul
// kernel. The non-streamed kernel (quire_matmul_op.cpp) reads ALL of A (M*K) and B
// (K*N) into each core's L1, capping operands at K=N<=512. This variant removes that
// ceiling: with the 2D (row x column-chunk) work split, a core only owns columns
// [j0,j1) of its rows, so it only needs A's row (K words) and B's CHUNK COLUMNS
// (K*chunk_cols words) — NOT all of B. L1 use is independent of N (and of the other
// rows), so it handles arbitrary sizes incl. large model linears (K=N=2048).
//
// The 256-bit quire still accumulates EXACTLY over all K (no per-tile rounding) — the
// exact cross-tile/streamed accumulation the matrix engine can't do.
#include <cstdint>

#include "../../../../kernel/bp16_quire.h"   // bp16_prod_to_q256, q256_add
#include "../../../../kernel/bp16_encode.h"  // bp16_encode_quire256
#include "../../../../kernel/bp32_encode.h"  // bp32_encode_quire256

#include "bp16_bf16.h"  // bf16_to_bp16_code / bp16_code_to_bf16 (on-device float<->code)

#define QLIMBS 8

void kernel_main() {
    uint32_t a_dram = get_arg_val<uint32_t>(0);
    uint32_t b_dram = get_arg_val<uint32_t>(1);
    uint32_t c_dram = get_arg_val<uint32_t>(2);
    uint32_t m_dim = get_arg_val<uint32_t>(3);
    uint32_t k_dim = get_arg_val<uint32_t>(4);
    uint32_t n_dim = get_arg_val<uint32_t>(5);
    uint32_t unit_start = get_arg_val<uint32_t>(6);
    uint32_t unit_end = get_arg_val<uint32_t>(7);
    uint32_t nchunks = get_arg_val<uint32_t>(8);
    uint32_t chunk_cols = get_arg_val<uint32_t>(9);
    uint32_t readout_bp32 = get_arg_val<uint32_t>(10);
    uint32_t input_float = get_arg_val<uint32_t>(11);     // operands are bf16 2B (encode; blocked by ttnn layout)
    uint32_t output_float = get_arg_val<uint32_t>(12);    // output bf16 (decode in-kernel)
    uint32_t input_bf16bits = get_arg_val<uint32_t>(13);  // operands uint32 w/ bf16 bits in low 16 (encode; proven read)
    uint32_t output_bf16bits = get_arg_val<uint32_t>(14); // output uint32 w/ bf16 bits in low 16 (decode; proven write)
    uint32_t lut_allowed = get_arg_val<uint32_t>(15);     // enable fused codec LUT fast path (BPOSIT_NO_LUT=0)

    constexpr uint32_t cb_a = 0, cb_b = 1, cb_c = 2;
    uint32_t a_l1 = get_write_ptr(cb_a);
    uint32_t b_l1 = get_write_ptr(cb_b);
    uint32_t c_l1 = get_write_ptr(cb_c);

    // Element byte sizes: bf16 = 2, bposit16-code (uint32) = 4.
    const uint32_t asz = input_float ? 2u : 4u;
    const uint32_t csz = output_float ? 2u : 4u;
    // TensorAccessors read the operands' ACTUAL ttnn layout (bank/page mapping for bf16 OR
    // uint32) from compile-time args the factory appended — correct for any ttnn tensor,
    // unlike a hand-rolled InterleavedAddrGen page-size guess (which scrambled bf16).
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const auto a_gen = TensorAccessor(a_args, a_dram);   // page/layout from the buffer's actual config
    const auto b_gen = TensorAccessor(b_args, b_dram);
    const auto c_gen = TensorAccessor(c_args, c_dram);

    volatile uint32_t* a_codes = (volatile uint32_t*)a_l1;
    volatile uint32_t* b_codes = (volatile uint32_t*)b_l1;
    volatile uint16_t* a_bf = (volatile uint16_t*)a_l1;
    volatile uint16_t* b_bf = (volatile uint16_t*)b_l1;
    volatile uint32_t* cout = (volatile uint32_t*)c_l1;
    volatile uint16_t* cout_bf = (volatile uint16_t*)c_l1;

    uint32_t cur_chunk = 0xFFFFFFFFu;   // B chunk read + pre-packed once per chunk (column-major units)

    // Fused codec LUT (bf16 bits -> packed bp16 operand): collapses the per-element encode AND the
    // in-MADD decode into one table load. Only for input_bf16bits (uint32 A buffer holds the 32-bit
    // packed value; input_float's 16-bit buffer can't) and only when the per-core work amortizes the
    // 64K build. cb 3 exists for bf16 input paths (factory).
    // Gate on actual per-core element-ops (units x chunk_cols x k_dim) vs the 64K LUT build cost — the
    // old units*k_dim metric omitted chunk_cols and starved real decoder linears (smaller N -> few
    // units/core) of the LUT. This enables the LUT fast path for the whole transformer, not just the LM head.
    const bool use_lut = lut_allowed && input_bf16bits &&
                         ((unit_end - unit_start) * chunk_cols * k_dim >= (1u << 16));
    volatile uint32_t* lut = (input_float || input_bf16bits) ? (volatile uint32_t*)get_write_ptr(3) : (volatile uint32_t*)0;
    if (use_lut) {
        for (uint32_t x = 0; x < (1u << 16); ++x) {
            lut[x] = (uint32_t)bp16_pack(bf16_to_bp16_code(x));
        }
    }

    // COLUMN-MAJOR units (c outer, i inner): a core's contiguous unit range covers runs of rows for
    // the SAME chunk, so each B chunk is read + pre-packed ONCE and reused across all those rows
    // (up to M x fewer B reads/packs — the new bottleneck once the codec LUT removed per-element work).
    // A is small (one K-row) and read+packed per unit. Same total_units, so NO factory change.
    for (uint32_t u = unit_start; u < unit_end; ++u) {
        uint32_t c = u / m_dim;
        uint32_t i = u % m_dim;
        uint32_t j0 = c * chunk_cols;
        uint32_t j1 = j0 + chunk_cols;
        if (j1 > n_dim) {
            j1 = n_dim;
        }
        uint32_t w = j1 - j0;
        const bool new_chunk = (c != cur_chunk);

        if (new_chunk) {
            for (uint32_t k = 0; k < k_dim; ++k) {                     // B[:, j0:j1] (once per chunk)
                noc_async_read(b_gen.get_noc_addr(k, j0 * asz), b_l1 + k * w * asz, w * asz);
            }
        }
        noc_async_read(a_gen.get_noc_addr(i), a_l1, k_dim * asz);      // A[i, :] (per unit/row)
        noc_async_read_barrier();

        // Pre-pack the B chunk bf16->packed-bp16 ONCE per chunk (LUT path), reused across all rows of
        // the chunk — the inner loop then does zero codec work. Bit-exact: same codec, packed once.
        if (new_chunk) {
            if (use_lut) {
                for (uint32_t idx = 0; idx < k_dim * w; ++idx) {
                    b_codes[idx] = lut[b_codes[idx] & 0xFFFFu];
                }
            }
            cur_chunk = c;
        }

        // Pre-encode/pack the A row (per unit).
        for (uint32_t k = 0; k < k_dim; ++k) {
            if (use_lut) {
                a_codes[k] = lut[a_codes[k] & 0xFFFFu];                // PACKED bp16 operand
            } else if (input_float) {
                a_bf[k] = (uint16_t)bf16_to_bp16_code((unsigned)a_bf[k]);
            } else if (input_bf16bits) {
                a_codes[k] = (uint32_t)bf16_to_bp16_code(a_codes[k] & 0xFFFFu);
            }
        }

        for (uint32_t jj = 0; jj < w; ++jj) {
            unsigned q[QLIMBS] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
            if (use_lut) {
                // Both A and B are PACKED (pre-packed above); the hot loop is pure fused MADD — no
                // encode, no decode, no LUT load.
                for (uint32_t k = 0; k < k_dim; ++k) {
                    bp16_madd_q256_packed(q, (unsigned)a_codes[k], (unsigned)b_codes[k * w + jj]);
                }
            } else {
                for (uint32_t k = 0; k < k_dim; ++k) {
                    // A already holds a bp16 code (pre-encoded above). B -> code here.
                    int ac = input_float ? (int)a_bf[k] : (int)(a_codes[k] & 0xFFFFu);
                    int bc;
                    if (input_float) {
                        bc = bf16_to_bp16_code((unsigned)b_bf[k * w + jj]);
                    } else if (input_bf16bits) {
                        bc = bf16_to_bp16_code(b_codes[k * w + jj] & 0xFFFFu);
                    } else {
                        bc = (int)(b_codes[k * w + jj] & 0xFFFFu);
                    }
                    bp16_madd_q256(q, ac, bc);
                }
            }
            if (output_float) {
                cout_bf[jj] = (uint16_t)bp16_code_to_bf16(bp16_encode_quire256(q));  // bf16 2B (blocked path)
            } else if (output_bf16bits) {
                cout[jj] = (uint32_t)bp16_code_to_bf16(bp16_encode_quire256(q));      // uint32, bf16 bits in low 16
            } else {
                cout[jj] = readout_bp32 ? (uint32_t)bp32_encode_quire256(q)
                                        : (uint32_t)(bp16_encode_quire256(q) & 0xFFFF);
            }
        }
        noc_async_write(c_l1, c_gen.get_noc_addr(i, j0 * csz), w * csz);  // row i, col offset j0
        noc_async_write_barrier();
    }
}
