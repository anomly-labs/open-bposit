// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bf16_codec_baby.cpp — on-device validation of the bf16<->bposit16-code codec
// (bp16_bf16.h) on a real Blackhole baby (RISC-V) core. Reads N bf16 values, round-trips
// each through bf16_to_bp16_code -> bp16_code_to_bf16, writes the resulting bf16 back.
// The host compares against the SAME codec computed on x86 — confirming the RV32 execution
// is byte-identical (validation-ladder step 1 for the host-round-trip-free op, task #18).
#include <cstdint>
#include "../../../kernel/bp16_bf16.h"

void kernel_main() {
    uint32_t in_dram = get_arg_val<uint32_t>(0);   // DRAM base: N bf16 (uint16) values, one page
    uint32_t out_dram = get_arg_val<uint32_t>(1);  // DRAM base: N bf16 outputs, one page
    uint32_t in_l1 = get_arg_val<uint32_t>(2);     // L1 scratch in
    uint32_t out_l1 = get_arg_val<uint32_t>(3);    // L1 scratch out
    uint32_t n = get_arg_val<uint32_t>(4);         // element count
    uint32_t nbytes = get_arg_val<uint32_t>(5);    // N*2 rounded up for the page

    InterleavedAddrGen<true> in_gen = {.bank_base_address = in_dram, .page_size = nbytes};
    InterleavedAddrGen<true> out_gen = {.bank_base_address = out_dram, .page_size = nbytes};
    noc_async_read(in_gen.get_noc_addr(0), in_l1, nbytes);
    noc_async_read_barrier();

    volatile uint16_t* in = (volatile uint16_t*)in_l1;
    volatile uint16_t* out = (volatile uint16_t*)out_l1;
    for (uint32_t i = 0; i < n; ++i) {
        int code = bf16_to_bp16_code((unsigned)in[i]);
        out[i] = (uint16_t)bp16_code_to_bf16(code);
    }
    noc_async_write(out_l1, out_gen.get_noc_addr(0), nbytes);
    noc_async_write_barrier();
}
