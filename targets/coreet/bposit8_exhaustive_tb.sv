// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit8_exhaustive_tb — EVERY bp8 x bp8 pair (all 65,536) through the real
// W8A8 MAC+encode (bposit8_dot, single product) vs the reference oracle. Full
// input-space coverage of the production rung.

module bposit8_exhaustive_tb;
  `include "bp8x_dims.svh"
  logic clk = 0, rst_n = 0, clr = 0, en = 0;
  logic [7:0] a, b;
  logic signed [255:0] quire;
  logic [31:0] bp32;
  bposit8_dot dut (.clk_i(clk), .rst_ni(rst_n), .clr_i(clr), .en_i(en),
                   .a_i(a), .b_i(b), .quire_o(quire), .bp32_o(bp32));
  always #5 clk = ~clk;

  logic [7:0]  a_mem [0:NPAIR-1];
  logic [7:0]  b_mem [0:NPAIR-1];
  logic [31:0] ebp_mem [0:NPAIR-1];
  int fails = 0;
  initial begin
    $readmemh("bp8x_a.hex", a_mem); $readmemh("bp8x_b.hex", b_mem); $readmemh("bp8x_ebp.hex", ebp_mem);
    rst_n = 0; @(posedge clk); @(posedge clk); rst_n = 1;
    for (int i = 0; i < NPAIR; i++) begin
      @(negedge clk); clr = 1; en = 0; @(negedge clk); clr = 0;   // fresh quire each pair
      a = a_mem[i]; b = b_mem[i]; en = 1; @(negedge clk);
      en = 0; @(negedge clk);
      if (bp32 !== ebp_mem[i]) begin
        fails++; if (fails <= 8) $display("  MISMATCH a=%02x b=%02x got=%08x exp=%08x", a_mem[i], b_mem[i], bp32, ebp_mem[i]);
      end
    end
    $display("=== bposit8 EXHAUSTIVE: all %0d bp8 product pairs -> bposit32 vs reference ===", NPAIR);
    $display("Summary: %0d/%0d bit-exact", NPAIR-fails, NPAIR);
    $display("RESULT: %s", (fails==0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
