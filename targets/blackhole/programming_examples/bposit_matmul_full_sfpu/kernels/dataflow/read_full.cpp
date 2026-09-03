// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// read_full.cpp — reader for the fully arbitrary-(M,N,K) exact matmul. For each of this core's
// rows, STREAMS the row's K (a_k, b_k) pairs just-in-time (a from a_dram[(row_start+r)*K+k],
// b=W from b_dram[k]) into shallow ring buffers cb_a/cb_b. Both A and W stream per-k -> K is
// unbounded by L1; W is re-read per row (the throughput tradeoff for arbitrary K).

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

    for (uint32_t r = 0; r < n_rows; ++r) {
        const uint32_t row = row_start + r;
        for (uint32_t k = 0; k < k_len; ++k) {
            cb_reserve_back(cb_a, 1);
            noc_async_read_page(row * k_len + k, a, get_write_ptr(cb_a));
            cb_reserve_back(cb_b, 1);
            noc_async_read_page(k, b, get_write_ptr(cb_b));
            noc_async_read_barrier();
            cb_push_back(cb_a, 1);
            cb_push_back(cb_b, 1);
        }
    }
}
