// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// read_limbs.cpp — reader for the lane-parallel q256_add SFPU example. NoC-reads
// the 8 Q-limb tiles into cb_q, the 8 X-limb tiles into cb_x, and the single ZERO
// seed tile into cb_zero (so the compute kernel can seed CARRY = 0). One page per
// tile, interleaved, exactly the read_tiles.cpp pattern from custom_sfpi_add.

#include <cstdint>

void kernel_main() {
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    uint32_t x_addr = get_arg_val<uint32_t>(1);
    uint32_t zero_addr = get_arg_val<uint32_t>(2);
    uint32_t n_limbs = get_arg_val<uint32_t>(3);  // = 8

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_x = tt::CBIndex::c_1;
    constexpr uint32_t cb_zero = tt::CBIndex::c_2;

    const uint32_t tile_bytes = get_tile_size(cb_q);

    // TensorAccessor compile-time arg layout (set by the host, in this order):
    //   [0] q, [1] x, [2] zero
    constexpr auto q_args = TensorAccessorArgs<0>();
    const auto q = TensorAccessor(q_args, q_addr);
    constexpr auto x_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, x_addr);
    constexpr auto zero_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto zero = TensorAccessor(zero_args, zero_addr);

    // --- ZERO seed: one tile into cb_zero ------------------------------------
    cb_reserve_back(cb_zero, 1);
    noc_async_read_page(0, zero, get_write_ptr(cb_zero));
    noc_async_read_barrier();
    cb_push_back(cb_zero, 1);

    // --- Q and X limbs: 8 tiles each, in limb order 0..7 ---------------------
    cb_reserve_back(cb_q, n_limbs);
    cb_reserve_back(cb_x, n_limbs);
    uint32_t q_wr = get_write_ptr(cb_q);
    uint32_t x_wr = get_write_ptr(cb_x);
    for (uint32_t limb = 0; limb < n_limbs; ++limb) {
        noc_async_read_page(limb, q, q_wr + limb * tile_bytes);
        noc_async_read_page(limb, x, x_wr + limb * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_q, n_limbs);
    cb_push_back(cb_x, n_limbs);
}
