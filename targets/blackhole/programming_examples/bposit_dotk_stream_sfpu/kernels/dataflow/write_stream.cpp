// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// write_stream.cpp — writer for the streaming-K dot: drains the final quire (8 tiles) to q_dram.

#include <cstdint>

void kernel_main() {
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr auto q_args = TensorAccessorArgs<0>();
    const auto q = TensorAccessor(q_args, q_addr);
    cb_wait_front(cb_out, 8);
    uint32_t rd = get_read_ptr(cb_out);
    for (uint32_t l = 0; l < 8; ++l) noc_async_write_page(l, q, rd + l * tile_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 8);
}
