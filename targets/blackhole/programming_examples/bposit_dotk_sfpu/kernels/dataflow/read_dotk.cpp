// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// read_dotk.cpp — reader for the in-kernel-K-loop bp16 dot. Loads all K a/b code tiles into
// cb_a/cb_b, the zero carry-seed into cb_zero, and seeds the running-quire ping buffer cb_qA
// with 8 zero tiles (q starts at 0). The compute kernel then loops K internally.

#include <cstdint>

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t zero_addr = get_arg_val<uint32_t>(2);
    uint32_t k_len = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_zero = tt::CBIndex::c_2;
    constexpr uint32_t cb_qA = tt::CBIndex::c_6;
    const uint32_t tile_bytes = get_tile_size(cb_a);

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);
    constexpr auto z_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const auto z = TensorAccessor(z_args, zero_addr);

    cb_reserve_back(cb_zero, 1);
    noc_async_read_page(0, z, get_write_ptr(cb_zero));

    cb_reserve_back(cb_qA, 8);   // seed q = 0 (8 zero tiles)
    uint32_t qwr = get_write_ptr(cb_qA);
    for (uint32_t l = 0; l < 8; ++l) noc_async_read_page(0, z, qwr + l * tile_bytes);

    cb_reserve_back(cb_a, k_len);
    uint32_t awr = get_write_ptr(cb_a);
    for (uint32_t k = 0; k < k_len; ++k) noc_async_read_page(k, a, awr + k * tile_bytes);
    cb_reserve_back(cb_b, k_len);
    uint32_t bwr = get_write_ptr(cb_b);
    for (uint32_t k = 0; k < k_len; ++k) noc_async_read_page(k, b, bwr + k * tile_bytes);

    noc_async_read_barrier();
    cb_push_back(cb_zero, 1);
    cb_push_back(cb_qA, 8);
    cb_push_back(cb_a, k_len);
    cb_push_back(cb_b, k_len);
}
