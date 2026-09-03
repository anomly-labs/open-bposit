// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// read_codes.cpp — reader for the lane-parallel bp16_decode SFPU example. NoC-reads
// the single CODES tile (32 bposit-16 codes, one per lane, replicated across rows)
// into cb_codes. One page == one tile. Mirrors read_limbs.cpp.

#include <cstdint>

void kernel_main() {
    uint32_t codes_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_codes = tt::CBIndex::c_0;

    constexpr auto codes_args = TensorAccessorArgs<0>();
    const auto codes = TensorAccessor(codes_args, codes_addr);

    cb_reserve_back(cb_codes, 1);
    noc_async_read_page(0, codes, get_write_ptr(cb_codes));
    noc_async_read_barrier();
    cb_push_back(cb_codes, 1);
}
