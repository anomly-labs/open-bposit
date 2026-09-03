// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// write_q.cpp — writer for the bp16 dot-accumulation step. Drains the updated running quire
// q_out (8 tiles) from cb_qout to q_dram (overwriting q_in for the next dispatch).

#include <cstdint>

void kernel_main() {
    uint32_t q_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_qout = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_qout);

    constexpr auto q_args = TensorAccessorArgs<0>();
    const auto q = TensorAccessor(q_args, q_addr);

    cb_wait_front(cb_qout, 8);
    uint32_t rd = get_read_ptr(cb_qout);
    for (uint32_t l = 0; l < 8; ++l) noc_async_write_page(l, q, rd + l * tile_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_qout, 8);
}
