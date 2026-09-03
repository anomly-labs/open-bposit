// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// formal_oc_signed_reduction.v — FORMAL proof of the one's-complement + popcount signed-reduction
// identity used by the exact-quire vector engine, for ALL inputs (not just the golden vectors).
//
// The engine replaces each lane's
// two's-complement conversion  tc = sign ? (~mag + 1) : mag  with the one's complement
// oc = mag ^ {256{sign}} plus a collective +popcount(signs) correction. Correctness rests on the
// per-lane identity  tc == oc + sign  (as 256-bit modular values); summing 16 lanes then gives
// Σ tc = Σ oc + Σ sign = Σ oc + popcount(signs) by linearity. Proving the single-lane identity for
// every (sign, mag) is therefore sufficient and is a small, full-width SAT problem.
//
// Run: yosys -p 'read_verilog formal/formal_oc_signed_reduction.v; hierarchy -top formal_oc;
//                proc; flatten; sat -prove tc oc_plus_sign -verify'
// PASS == "SAT proof finished - no model found: SUCCESS!"  (identity holds for all inputs).
`default_nettype none

module formal_oc (
    input  wire         sign,
    input  wire [254:0] mag_bits,          // 255-bit magnitude (matches {1'b0, prod_sm[254:0]})
    output wire [255:0] tc,
    output wire [255:0] oc_plus_sign
);
    wire [255:0] mag = {1'b0, mag_bits};
    // two's-complement form (what the original engine builds)
    assign tc = sign ? (~mag + 256'd1) : mag;
    // one's-complement + collective +sign (what the OC engine builds, per lane)
    wire [255:0] oc = mag ^ {256{sign}};
    assign oc_plus_sign = oc + {255'd0, sign};
endmodule
`default_nettype wire
