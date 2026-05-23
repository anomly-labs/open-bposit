// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_lowp_alu_tb — EXHAUSTIVE check of the computed bp4 & aip5 ALU: every
// operand pair (mul AND add) vs the independent reference (gen_alu_tables.py).
// 256 (bp4) + 1024 (aip5) pairs per op = 2,560 total.

module bposit_lowp_alu_tb;
  // aip5 DUT
  logic        a5_op; logic [4:0] a5_a, a5_b; logic [4:0] a5_y;
  bposit_lowp_alu #(.NBITS(5), .NINR(13), .MAXE(4), .NAR(8'h10), .ZERO(8'h00),
                    .MINP(8'h01), .MAXP(8'h0f),
                    .DEC_FILE("aip5_dec.hex"), .INRV_FILE("aip5_inrv.hex"), .INRC_FILE("aip5_inrc.hex"))
    aip5 (.op_add_i(a5_op), .a_i(a5_a), .b_i(a5_b), .y_o(a5_y));
  // bp4 DUT
  logic        b4_op; logic [3:0] b4_a, b4_b; logic [3:0] b4_y;
  bposit_lowp_alu #(.NBITS(4), .NINR(5), .MAXE(2), .NAR(8'h08), .ZERO(8'h00),
                    .MINP(8'h01), .MAXP(8'h07),
                    .DEC_FILE("bp4_dec.hex"), .INRV_FILE("bp4_inrv.hex"), .INRC_FILE("bp4_inrc.hex"))
    bp4 (.op_add_i(b4_op), .a_i(b4_a), .b_i(b4_b), .y_o(b4_y));

  logic [7:0] a5m [0:1023]; logic [7:0] a5ad [0:1023];
  logic [7:0] b4m [0:255];  logic [7:0] b4ad [0:255];
  int fa=0, fb=0;

  initial begin
    $readmemh("aip5_mul.hex", a5m); $readmemh("aip5_add.hex", a5ad);
    $readmemh("bp4_mul.hex",  b4m); $readmemh("bp4_add.hex",  b4ad);
    // aip5: all 32x32 pairs
    for (int a=0;a<32;a++) for (int b=0;b<32;b++) begin
      a5_a=a[4:0]; a5_b=b[4:0];
      a5_op=0; #1; if (a5_y !== a5m[a*32+b][4:0])  begin fa++; if(fa<=6) $display("  aip5 mul %0d,%0d got=%02x exp=%02x",a,b,a5_y,a5m[a*32+b]); end
      a5_op=1; #1; if (a5_y !== a5ad[a*32+b][4:0]) begin fa++; if(fa<=6) $display("  aip5 add %0d,%0d got=%02x exp=%02x",a,b,a5_y,a5ad[a*32+b]); end
    end
    // bp4: all 16x16 pairs
    for (int a=0;a<16;a++) for (int b=0;b<16;b++) begin
      b4_a=a[3:0]; b4_b=b[3:0];
      b4_op=0; #1; if (b4_y !== b4m[a*16+b][3:0])  begin fb++; if(fb<=6) $display("  bp4 mul %0d,%0d got=%01x exp=%02x",a,b,b4_y,b4m[a*16+b]); end
      b4_op=1; #1; if (b4_y !== b4ad[a*16+b][3:0]) begin fb++; if(fb<=6) $display("  bp4 add %0d,%0d got=%01x exp=%02x",a,b,b4_y,b4ad[a*16+b]); end
    end
    $display("=== bposit_lowp_alu: EXHAUSTIVE compute vs reference ===");
    $display("Summary: aip5 %0d/2048, bp4 %0d/512 bit-exact", 2048-fa, 512-fb);
    $display("RESULT: %s", (fa==0 && fb==0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
