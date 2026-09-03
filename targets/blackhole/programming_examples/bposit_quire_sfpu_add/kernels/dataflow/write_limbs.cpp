// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// write_limbs.cpp — writer for the lane-parallel q256_add SFPU example. Drains the
// 8 OUT-limb tiles from cb_out to DRAM (one page per tile). Mirrors write_tile.cpp
// from custom_sfpi_add, generalised to 8 tiles.

#include <cstdint>

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t n_limbs = get_arg_val<uint32_t>(1);  // = 8

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr);

    cb_wait_front(cb_out, n_limbs);
    uint32_t rd = get_read_ptr(cb_out);
    for (uint32_t limb = 0; limb < n_limbs; ++limb) {
        noc_async_write_page(limb, out, rd + limb * tile_bytes);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, n_limbs);
}
