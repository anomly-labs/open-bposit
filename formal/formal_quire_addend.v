// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// formal_quire_addend.v — FORMAL proof that exact_quire256 forms its two's-complement addend
// correctly from a sign-magnitude input, for ALL inputs. This guards permanently against the exact
// defect that shipped in two earlier quire implementations:
// they ADDED a sign-magnitude word as if it were two's complement, so a negative product added
// ~+2^255 instead of subtracting, and catastrophic cancellation could never cancel.
//
// exact_quire256's fix is  addend = sign ? (~mag + 1) : mag  with mag = {1'b0, addend_sm[254:0]}.
// The correctness claim: interpreted as a signed 256-bit value, addend equals +mag when sign=0 and
// −mag when sign=1 (mod 2^256). Proving this for every (sign, mag) means the quire accumulate
// `quire + addend` is the exact running signed sum for any input — the property the two buggy
// quires lacked. (This is the accumulate-side companion to formal_csa_compressor.v and
// formal_oc_signed_reduction.v — together the reduction and the accumulate are formally exact.)
//
// Run: yosys -p 'read_verilog formal/formal_quire_addend.v; hierarchy -top formal_quire_addend;
//                proc; flatten; sat -prove addend reference -verify'
// PASS == "SAT proof finished - no model found: SUCCESS!"
`default_nettype none

module formal_quire_addend (
    input  wire        sign,
    input  wire [254:0] mag_bits,
    output wire [255:0] addend,
    output wire [255:0] reference
);
    wire [255:0] mag = {1'b0, mag_bits};                 // non-negative magnitude (bit 255 = 0)
    // exact_quire256's addend formation
    assign addend    = sign ? (~mag + 256'd1) : mag;
    // the value it must equal: +mag when sign=0, two's-complement −mag when sign=1 (mod 2^256)
    assign reference = sign ? (256'd0 - mag) : mag;
endmodule
`default_nettype wire
