// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// read_two_codes.cpp — reader for the lane-parallel bp16 product example. NoC-reads the
// CODE_A tile into cb_a and the CODE_B tile into cb_b (each 32 codes, one per lane,
// replicated across rows). One page == one tile. Mirrors read_codes.cpp x2.

#include <cstdint>

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    cb_reserve_back(cb_a, 1);
    noc_async_read_page(0, a, get_write_ptr(cb_a));
    cb_reserve_back(cb_b, 1);
    noc_async_read_page(0, b, get_write_ptr(cb_b));
    noc_async_read_barrier();
    cb_push_back(cb_a, 1);
    cb_push_back(cb_b, 1);
}
