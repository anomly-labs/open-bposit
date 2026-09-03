// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// read_dot.cpp — reader for the bp16 dot-accumulation step. Loads the running quire q_in
// (8 tiles) into cb_qin, the code pair a_k/b_k (1 tile each) into cb_a/cb_b, and the zero
// carry-seed tile into cb_zero.

#include <cstdint>

void kernel_main() {
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    uint32_t a_addr = get_arg_val<uint32_t>(1);
    uint32_t b_addr = get_arg_val<uint32_t>(2);
    uint32_t zero_addr = get_arg_val<uint32_t>(3);
    uint32_t a_tile = get_arg_val<uint32_t>(4);   // which tile of a_dram (= k)
    uint32_t b_tile = get_arg_val<uint32_t>(5);   // which tile of b_dram (= k)

    constexpr uint32_t cb_qin = tt::CBIndex::c_4;
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_zero = tt::CBIndex::c_2;
    const uint32_t tile_bytes = get_tile_size(cb_qin);

    constexpr auto q_args = TensorAccessorArgs<0>();
    const auto q = TensorAccessor(q_args, q_addr);
    constexpr auto a_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);
    constexpr auto z_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const auto z = TensorAccessor(z_args, zero_addr);

    // zero seed
    cb_reserve_back(cb_zero, 1);
    noc_async_read_page(0, z, get_write_ptr(cb_zero));
    // code pair
    cb_reserve_back(cb_a, 1);
    noc_async_read_page(a_tile, a, get_write_ptr(cb_a));
    cb_reserve_back(cb_b, 1);
    noc_async_read_page(b_tile, b, get_write_ptr(cb_b));
    // running quire (8 tiles)
    cb_reserve_back(cb_qin, 8);
    uint32_t qwr = get_write_ptr(cb_qin);
    for (uint32_t l = 0; l < 8; ++l) noc_async_read_page(l, q, qwr + l * tile_bytes);

    noc_async_read_barrier();
    cb_push_back(cb_zero, 1);
    cb_push_back(cb_a, 1);
    cb_push_back(cb_b, 1);
    cb_push_back(cb_qin, 8);
}
