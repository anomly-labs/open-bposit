// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// write_limbs8.cpp — writer for the lane-parallel bp16 placement example. Drains the 8
// quire-limb contribution tiles from cb_out to DRAM (one page per tile). The compute
// kernel pushes them in two groups of 4 (limbs 0..3 then 4..7); we wait for all 8.

#include <cstdint>

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t n_out = get_arg_val<uint32_t>(1);  // = 8

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr);

    cb_wait_front(cb_out, n_out);
    uint32_t rd = get_read_ptr(cb_out);
    for (uint32_t t = 0; t < n_out; ++t) {
        noc_async_write_page(t, out, rd + t * tile_bytes);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, n_out);
}
