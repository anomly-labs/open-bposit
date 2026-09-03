// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// formal_csa_compressor.v — FORMAL proof that the 4:2 carry-save compressor at the heart of
// the exact-quire reduction tree (bposit_csa_cs42.v) is bit-exact for ALL inputs, not just the
// golden vectors.
//
// A vector testbench can only show that a candidate satisfies the visible harness, not the full
// specification; a SAT proof covers every input. The whole CSA reduction is a tree of bposit_csa_cs42 (4:2) compressors plus one
// more for the accumulate; if the single 4:2 compressor is proven exact (s + c == s0+c0+s1+c1,
// mod 2^256) then the entire reduction and the redundant-quire accumulate are exact by
// composition. This is the strongest possible correctness statement for a datapath bound for
// silicon — it covers every input, not a sample.
//
// Run (yosys SAT):
//   yosys -p 'read_verilog formal/bposit_csa_cs42.v formal/formal_csa_compressor.v
//             hierarchy -top formal_csa_compressor; proc; flatten
//             sat -prove resolved reference -verify'
// PASS  == "Proved 'resolved' == 'reference' ... SUCCESS!" (property holds for all inputs;
//          a failure would print a concrete counterexample assignment).
`default_nettype none

module formal_csa_compressor (
    input  wire [255:0] s0, c0, s1, c1,
    output wire [255:0] resolved,
    output wire [255:0] reference
);
    wire [255:0] s, c;
    bposit_csa_cs42 u (.s0(s0), .c0(c0), .s1(s1), .c1(c1), .s(s), .c(c));

    // The 4:2 compressor must preserve the value exactly under 256-bit modular arithmetic:
    // the resolved pair (s + c) equals the four addends summed (both sides wrap mod 2^256).
    assign resolved  = s + c;
    assign reference = s0 + c0 + s1 + c1;
endmodule
`default_nettype wire
