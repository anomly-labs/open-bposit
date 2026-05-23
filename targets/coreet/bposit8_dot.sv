// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit8_dot — W8A8 GEMM cell: exact bp8 quire MAC -> bposit32 (reuses the
// same bposit_encode as the aip5 path).

module bposit8_dot (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        clr_i,
  input  logic        en_i,
  input  logic [7:0]  a_i,
  input  logic [7:0]  b_i,
  output logic signed [255:0] quire_o,
  output logic [31:0] bp32_o
);
  bposit8_qmac mac (.clk_i, .rst_ni, .clr_i, .en_i, .a_i, .b_i, .quire_o(quire_o));
  bposit_encode enc (.quire_i(quire_o), .bp32_o(bp32_o));
endmodule
