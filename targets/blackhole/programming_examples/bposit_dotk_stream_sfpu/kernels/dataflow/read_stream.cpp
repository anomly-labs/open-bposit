// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// read_stream.cpp — STREAMING-K reader. Loads the zero carry-seed once, then streams the K
// (a_k, b_k) code pairs just-in-time (1 a-tile + 1 b-tile per k) into shallow ring buffers
// cb_a/cb_b. K is therefore unbounded by L1 (the old all-K-resident reader capped K ~100).

#include <cstdint>

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t zero_addr = get_arg_val<uint32_t>(2);
    uint32_t k_len = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_zero = tt::CBIndex::c_2;

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);
    constexpr auto z_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const auto z = TensorAccessor(z_args, zero_addr);

    cb_reserve_back(cb_zero, 1);
    noc_async_read_page(0, z, get_write_ptr(cb_zero));
    noc_async_read_barrier();
    cb_push_back(cb_zero, 1);

    for (uint32_t k = 0; k < k_len; ++k) {
        cb_reserve_back(cb_a, 1);
        noc_async_read_page(k, a, get_write_ptr(cb_a));
        cb_reserve_back(cb_b, 1);
        noc_async_read_page(k, b, get_write_ptr(cb_b));
        noc_async_read_barrier();
        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
    }
}
