// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_dot_tb — drives M dot products of length K and checks the bposit32
// output bit-for-bit against the reference (ebp32.hex == GPU/RISC-V/x86).

module bposit_dot_tb;
  `include "dims.svh"

  logic clk = 0, rst_n = 0, clr = 0, en = 0;
  logic [4:0] a, b;
  logic signed [255:0] quire;
  logic [31:0] bp32;

  bposit_dot dut (.clk_i(clk), .rst_ni(rst_n), .clr_i(clr), .en_i(en),
                  .a_i(a), .b_i(b), .quire_o(quire), .bp32_o(bp32));

  always #5 clk = ~clk;

  logic [7:0]  a_mem [0:M*K-1];
  logic [7:0]  b_mem [0:M*K-1];
  logic [31:0] ebp_mem [0:M-1];

  int fails = 0;
  initial begin
    $readmemh("a.hex", a_mem);
    $readmemh("b.hex", b_mem);
    $readmemh("ebp32.hex", ebp_mem);
    rst_n = 0; @(posedge clk); @(posedge clk); rst_n = 1;
    for (int m = 0; m < M; m++) begin
      @(negedge clk); clr = 1; en = 0; @(negedge clk); clr = 0;
      for (int k = 0; k < K; k++) begin
        a = a_mem[m*K+k][4:0]; b = b_mem[m*K+k][4:0]; en = 1;
        @(negedge clk);
      end
      en = 0; @(negedge clk);              // settle accumulate + comb encode
      if (bp32 !== ebp_mem[m]) begin
        fails++;
        if (fails <= 4)
          $display("  MISMATCH dot %0d: got=%08x exp=%08x", m, bp32, ebp_mem[m]);
      end
    end
    $display("=== bposit_dot: GEMM cell (quire MAC -> bposit32) vs reference ===");
    $display("Summary: %0d/%0d bposit32 results bit-exact", M-fails, M);
    $display("RESULT: %s", (fails==0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
