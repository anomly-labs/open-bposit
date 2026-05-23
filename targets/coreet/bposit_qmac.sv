// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_qmac — one exact b-posit (aiposit5) quire multiply-accumulate.
//
// The float-free b-posit MAC, in synthesizable SystemVerilog: decode two aip5
// codes to Q16 fixed-point (a small ROM — maps onto CORE-ET's trans_*_rom
// scheme), multiply (integer — maps onto tima_top), and accumulate the product
// EXACTLY into a 256-bit quire (a wide tima_adder + the vpu_tensorc_rf
// accumulator). No rounding until a separate encode step, so the result is
// bit-identical on any hardware — verified against the same reference the
// GPU/RISC-V port passes (see bposit_qmac_tb.sv).
//
// Q16 product (Q32) is aligned to the quire's 96 fraction bits by <<64.

module bposit_qmac #(
  parameter int QUIRE_W = 256,
  parameter int FRAC    = 16            // Q16 decode; product is Q(2*FRAC)=Q32
) (
  input  logic                       clk_i,
  input  logic                       rst_ni,
  input  logic                       clr_i,    // clear the quire (start a new dot)
  input  logic                       en_i,     // accumulate one product this cycle
  input  logic [4:0]                 a_i,      // aip5 code
  input  logic [4:0]                 b_i,      // aip5 code
  output logic signed [QUIRE_W-1:0]  quire_o
);
  // decode ROM: aip5 code -> signed Q16 value (loaded from the reference).
  logic signed [31:0] dec_rom [0:31];
  initial $readmemh("aip5_dec.hex", dec_rom);

  logic signed [31:0] va, vb;
  assign va = dec_rom[a_i];
  assign vb = dec_rom[b_i];

  // exact integer product (Q32), sign-extended and shifted to the Q96 window.
  logic signed [63:0]        prod;
  logic signed [QUIRE_W-1:0] prod_q96;
  localparam int SH = 96 - (2*FRAC);    // 64: align Q32 -> Q96
  assign prod     = va * vb;
  assign prod_q96 = $signed({{(QUIRE_W-64){prod[63]}}, prod}) <<< SH;

  logic signed [QUIRE_W-1:0] quire_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)      quire_q <= '0;
    else if (clr_i)   quire_q <= '0;
    else if (en_i)    quire_q <= quire_q + prod_q96;
  end
  assign quire_o = quire_q;
endmodule
