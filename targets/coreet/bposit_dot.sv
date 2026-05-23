// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_dot — a full b-posit GEMM cell: exact quire accumulate (bposit_qmac)
// then round-once encode (bposit_encode) -> bposit32. The synthesizable
// realization of one anomly_bposit_gemm(use_quire=true) output element.

module bposit_dot (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        clr_i,
  input  logic        en_i,
  input  logic [4:0]  a_i,
  input  logic [4:0]  b_i,
  output logic signed [255:0] quire_o,
  output logic [31:0] bp32_o
);
  bposit_qmac mac (.clk_i, .rst_ni, .clr_i, .en_i, .a_i, .b_i, .quire_o(quire_o));
  bposit_encode enc (.quire_i(quire_o), .bp32_o(bp32_o));
endmodule
