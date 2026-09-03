// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// read_mc2.cpp — multi-ROW-per-core reader. Loads the SHARED weights W (b_dram, K tiles) once,
// then for each of this core's rows: pushes a fresh 8-zero quire seed into cb_qA and the row's
// K a-tiles (a_dram pages [(row_start+r)*K .. +K-1]) into cb_a. The compute kernel loops rows.

#include <cstdint>

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t zero_addr = get_arg_val<uint32_t>(2);
    uint32_t k_len = get_arg_val<uint32_t>(3);
    uint32_t row_start = get_arg_val<uint32_t>(4);
    uint32_t n_rows = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_zero = tt::CBIndex::c_2;
    const uint32_t tile_bytes = get_tile_size(cb_a);

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);
    constexpr auto z_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const auto z = TensorAccessor(z_args, zero_addr);

    // zero seed (carry), once
    cb_reserve_back(cb_zero, 1);
    noc_async_read_page(0, z, get_write_ptr(cb_zero));
    // W, once (shared across all rows)
    cb_reserve_back(cb_b, k_len);
    uint32_t bwr = get_write_ptr(cb_b);
    for (uint32_t k = 0; k < k_len; ++k) noc_async_read_page(k, b, bwr + k * tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_zero, 1);
    cb_push_back(cb_b, k_len);

    for (uint32_t r = 0; r < n_rows; ++r) {
        const uint32_t row = row_start + r;
        // this row's A (K tiles). cb_qA is seeded compute-side (kept off the reader to avoid a race).
        cb_reserve_back(cb_a, k_len);
        uint32_t awr = get_write_ptr(cb_a);
        for (uint32_t k = 0; k < k_len; ++k) noc_async_read_page(row * k_len + k, a, awr + k * tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_a, k_len);
    }
}
