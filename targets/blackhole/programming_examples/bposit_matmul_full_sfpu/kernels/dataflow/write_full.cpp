// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// write_full.cpp — writer: for each of this core's rows, drains the row's quire (8 tiles) to
// q_dram pages [(row_start+r)*8 .. +7].

#include <cstdint>

void kernel_main() {
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    uint32_t row_start = get_arg_val<uint32_t>(1);
    uint32_t n_rows = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr auto q_args = TensorAccessorArgs<0>();
    const auto q = TensorAccessor(q_args, q_addr);

    for (uint32_t r = 0; r < n_rows; ++r) {
        const uint32_t base = (row_start + r) * 8;
        cb_wait_front(cb_out, 8);
        uint32_t rd = get_read_ptr(cb_out);
        for (uint32_t l = 0; l < 8; ++l) noc_async_write_page(base + l, q, rd + l * tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 8);
    }
}
