// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit8_dot_tb — W8A8 GEMM cell vs reference: checks BOTH the exact 256-bit
// quire (bp8_eq.hex) and the bposit32 output (bp8_ebp.hex), incl. edge cases
// (NaR/zero/max/min). Oracle == reference == GPU/RISC-V.

module bposit8_dot_tb;
  `include "bp8_dims.svh"

  logic clk = 0, rst_n = 0, clr = 0, en = 0;
  logic [7:0] a, b;
  logic signed [255:0] quire;
  logic [31:0] bp32;

  bposit8_dot dut (.clk_i(clk), .rst_ni(rst_n), .clr_i(clr), .en_i(en),
                   .a_i(a), .b_i(b), .quire_o(quire), .bp32_o(bp32));
  always #5 clk = ~clk;

  logic [7:0]   a_mem [0:M8*K8-1];
  logic [7:0]   b_mem [0:M8*K8-1];
  logic [255:0] eq_mem [0:M8-1];
  logic [31:0]  ebp_mem [0:M8-1];

  int qfail = 0, bfail = 0;
  initial begin
    $readmemh("bp8_a.hex", a_mem); $readmemh("bp8_b.hex", b_mem);
    $readmemh("bp8_eq.hex", eq_mem); $readmemh("bp8_ebp.hex", ebp_mem);
    rst_n = 0; @(posedge clk); @(posedge clk); rst_n = 1;
    for (int m = 0; m < M8; m++) begin
      @(negedge clk); clr = 1; en = 0; @(negedge clk); clr = 0;
      for (int k = 0; k < K8; k++) begin
        a = a_mem[m*K8+k]; b = b_mem[m*K8+k]; en = 1; @(negedge clk);
      end
      en = 0; @(negedge clk);
      if (quire !== eq_mem[m]) begin qfail++; if (qfail<=4) $display("  Q MISMATCH %0d: got=%064x exp=%064x", m, quire, eq_mem[m]); end
      if (bp32  !== ebp_mem[m]) begin bfail++; if (bfail<=4) $display("  BP MISMATCH %0d: got=%08x exp=%08x", m, bp32, ebp_mem[m]); end
    end
    $display("=== bposit8_dot (W8A8): exact quire + bposit32 vs reference ===");
    $display("Summary: quire %0d/%0d, bposit32 %0d/%0d bit-exact", M8-qfail, M8, M8-bfail, M8);
    $display("RESULT: %s", (qfail==0 && bfail==0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
