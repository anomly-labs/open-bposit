// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_csa_cs42 — the 256-bit 4:2 carry-save compressor at the heart of the exact-quire
// reduction tree. Two carry-save pairs in, one carry-save pair out, value preserved exactly
// under 256-bit modular arithmetic (s + c == s0 + c0 + s1 + c1 mod 2^256). The full reduction
// and the redundant-quire accumulate are compositions of this single cell, which is why
// proving this cell exact for ALL inputs (formal_csa_compressor.v) proves the whole reduction
// structure exact. Extracted verbatim from the vector-MAC engine so the proof is self-contained.
`default_nettype none

module bposit_csa_cs42 (
    input  wire [255:0] s0, c0, s1, c1,
    output wire [255:0] s,  c
);
    wire [255:0] ts =  s0 ^ c0 ^ s1;
    wire [255:0] tc = ((s0 & c0) | (s0 & s1) | (c0 & s1)) << 1;
    assign s =  ts ^ tc ^ c1;
    assign c = ((ts & tc) | (ts & c1) | (tc & c1)) << 1;
endmodule

`default_nettype wire
