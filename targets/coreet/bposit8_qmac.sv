// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit8_qmac — exact bposit8 (W8A8) quire MAC. The production rung; maps onto
// CORE-ET's tima_top (int8 MAC). bp8 spans 2^-24..2^24, so its products underflow
// a fixed Q16; instead each code decodes to an exact (signed mantissa, exponent)
// and the product (ma*mb) is placed at quire bit (96 + ea + eb) — a variable
// shift into the 256-bit accumulator. No rounding => bit-exact.

module bposit8_qmac #(
  parameter int QUIRE_W = 256
) (
  input  logic                       clk_i,
  input  logic                       rst_ni,
  input  logic                       clr_i,
  input  logic                       en_i,
  input  logic [7:0]                 a_i,      // bp8 code
  input  logic [7:0]                 b_i,
  output logic signed [QUIRE_W-1:0]  quire_o
);
  // decode ROMs (loaded from the reference): code -> signed mantissa / exponent.
  logic signed [15:0] mant_rom [0:255];
  logic signed [7:0]  exp_rom  [0:255];
  initial begin
    $readmemh("bp8_mant.hex", mant_rom);
    $readmemh("bp8_exp.hex",  exp_rom);
  end

  logic signed [15:0] ma, mb;
  logic signed [7:0]  ea, eb;
  assign ma = mant_rom[a_i];
  assign mb = mant_rom[b_i];
  assign ea = exp_rom[a_i];
  assign eb = exp_rom[b_i];

  // product (small: |ma*mb| <= ~2^16) placed at bit (96 + ea + eb).
  logic signed [31:0]        prod;
  logic signed [31:0]        bitpos;            // 96 + ea + eb, range ~48..144
  logic signed [QUIRE_W-1:0] prod_q;
  assign prod   = ma * mb;
  assign bitpos = 32'sd96 + ea + eb;
  // sign-extend product to the quire width, then shift left by bitpos.
  assign prod_q = $signed({{(QUIRE_W-32){prod[31]}}, prod}) <<< bitpos;

  logic signed [QUIRE_W-1:0] quire_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)    quire_q <= '0;
    else if (clr_i) quire_q <= '0;
    else if (en_i)  quire_q <= quire_q + prod_q;
  end
  assign quire_o = quire_q;
endmodule
