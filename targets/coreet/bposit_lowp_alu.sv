// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_lowp_alu — elementwise low-precision (bp4 / aip5) mul & add, COMPUTED
// (not a result lookup): decode (Q16 ROM) -> integer arith -> encode by total_e
// saturation + in-range magnitude-truncate threshold scan. The threshold encode
// is the non-trivial logic; it's verified exhaustively (all pairs) against the
// independent reference. Maps onto a small VPU lane / trans_*_rom on CORE-ET.

module bposit_lowp_alu #(
  parameter int    NBITS = 5,
  parameter int    NINR  = 13,
  parameter int    MAXE  = 4,
  parameter int    FRAC  = 16,
  parameter logic [7:0] NAR  = 8'h10,
  parameter logic [7:0] ZERO = 8'h00,
  parameter logic [7:0] MINP = 8'h01,
  parameter logic [7:0] MAXP = 8'h0f,
  parameter string DEC_FILE  = "aip5_dec.hex",
  parameter string INRV_FILE = "aip5_inrv.hex",
  parameter string INRC_FILE = "aip5_inrc.hex"
) (
  input  logic               op_add_i,        // 0 = mul, 1 = add
  input  logic [NBITS-1:0]   a_i,
  input  logic [NBITS-1:0]   b_i,
  output logic [NBITS-1:0]   y_o
);
  localparam logic [NBITS-1:0] MASK   = '1;
  // NBITS-wide special codes (narrow once at elaboration — no in-process selects).
  localparam logic [NBITS-1:0] NAR_c  = NAR[NBITS-1:0];
  localparam logic [NBITS-1:0] ZERO_c = ZERO[NBITS-1:0];
  localparam logic [NBITS-1:0] MINP_c = MINP[NBITS-1:0];
  localparam logic [NBITS-1:0] MAXP_c = MAXP[NBITS-1:0];

  logic signed [31:0]      dec  [0:(1<<NBITS)-1];
  logic        [31:0]      inrv [0:NINR-1];
  logic        [NBITS-1:0] inrc [0:NINR-1];      // codes fit in NBITS bits
  initial begin
    $readmemh(DEC_FILE,  dec);
    $readmemh(INRV_FILE, inrv);
    $readmemh(INRC_FILE, inrc);
  end

  // total_e saturation + in-range threshold encode of a signed Q16 value.
  function automatic logic [NBITS-1:0] thr(input logic signed [47:0] fx);
    logic               sign;
    logic        [47:0] u;
    int                 msb, te, i;
    logic [NBITS-1:0]   code;
    begin
      if (fx == 0) return ZERO_c;
      sign = (fx < 0);
      u    = sign ? -fx : fx;
      msb  = -1;
      for (i = 0; i < 48; i++) if (u[i]) msb = i;
      te = msb - FRAC;
      if (te >  MAXE) code = MAXP_c;
      else if (te < -MAXE) code = MINP_c;
      else begin
        code = inrc[0];
        for (i = 0; i < NINR; i++)
          if ({16'b0, inrv[i]} <= u) code = inrc[i];   // largest in-range value <= u
      end
      if (sign) code = (~code + {{(NBITS-1){1'b0}},1'b1}) & MASK;
      thr = code;
    end
  endfunction

  logic signed [31:0] va, vb;
  logic signed [63:0] prod;
  logic signed [47:0] prod_q16, sum;
  logic [NBITS-1:0]   mul_y, add_y;

  always_comb begin
    va = dec[a_i];
    vb = dec[b_i];
    prod     = va * vb;
    prod_q16 = prod >>> FRAC;                 // Q32 -> Q16
    sum      = va + vb;                        // signed; sign-extended to 48b by context

    // mul: zero before NaR (0*NaR=0); add: NaR first
    if (a_i == ZERO_c || b_i == ZERO_c) mul_y = ZERO_c;
    else if (a_i == NAR_c || b_i == NAR_c) mul_y = NAR_c;
    else mul_y = thr(prod_q16);

    if (a_i == NAR_c || b_i == NAR_c) add_y = NAR_c;
    else if (a_i == ZERO_c) add_y = b_i;
    else if (b_i == ZERO_c) add_y = a_i;
    else add_y = thr(sum);

    y_o = op_add_i ? add_y : mul_y;
  end
endmodule
