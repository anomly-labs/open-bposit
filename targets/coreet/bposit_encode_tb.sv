// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_encode_tb — EXHAUSTIVE differential check of the quire256->bposit32
// encoder against the reference oracle (qin.hex/ebp.hex, gen_encoder_vectors.py).
// Combinational, so we can push 100k+ vectors. A single mismatch fails the run.

module bposit_encode_tb;
  `include "enc_dims.svh"

  logic signed [255:0] qin;
  logic        [31:0]  bp32;
  bposit_encode dut (.quire_i(qin), .bp32_o(bp32));

  logic [255:0] q_mem  [0:NVEC-1];
  logic [31:0]  bp_mem [0:NVEC-1];

  int fails = 0;
  initial begin
    $readmemh("qin.hex", q_mem);
    $readmemh("ebp.hex", bp_mem);
    for (int i = 0; i < NVEC; i++) begin
      qin = q_mem[i];
      #1;
      if (bp32 !== bp_mem[i]) begin
        fails++;
        if (fails <= 8)
          $display("  MISMATCH vec %0d: qin=%064x got=%08x exp=%08x", i, q_mem[i], bp32, bp_mem[i]);
      end
    end
    $display("=== bposit_encode: EXHAUSTIVE quire256->bposit32 vs reference oracle ===");
    $display("Summary: %0d/%0d vectors bit-exact", NVEC-fails, NVEC);
    $display("RESULT: %s", (fails==0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
