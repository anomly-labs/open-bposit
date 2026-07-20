// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_encode — combinational quire256 -> bposit32 encoder (round-once).
// Direct SystemVerilog port of quire256_to_bposit32 (bposit16_encode.cuh),
// the keystone encoder proven bit-exact vs the Python reference. Maps onto
// CORE-ET's vpu_tensorquant (QUANT) op, fed by the quire instead of an FP acc.

module bposit_encode (
  input  logic signed [255:0] quire_i,
  output logic        [31:0]  bp32_o
);
  localparam int FRAC = 96;

  function automatic logic [31:0] q2bp32(input logic signed [255:0] qin);
    logic [255:0] q;
    logic         sign;
    int           msb, scale, k, e, abs_s, abs_k, abs_e;
    logic [31:0]  mag;
    int           top_pos, regime_bits, ones_count, zeros_count, term_pos;
    int           exp_avail, exp_bits, frac_bits, hi, lo, width;
    logic [31:0]  e_aligned, frac_field;
    logic [255:0] shifted;
    begin
      if (qin == 0) return 32'h0000_0000;
      sign = qin[255];
      q    = sign ? (-qin) : qin;                 // magnitude

      msb = -1;
      for (int i = 0; i < 256; i++) if (q[i]) msb = i;   // highest set bit
      if (msb < 0) return 32'h0000_0000;

      scale = msb - FRAC;
      if (scale >  240) return sign ? 32'h8000_0001 : 32'h7FFF_FFFF; // MAXNEG/MAXPOS
      if (scale < -240) return sign ? 32'hFFFF_FFFF : 32'h0000_0001; // MINNEG/MINPOS

      if (scale >= 0) begin
        k = scale >>> 3;
        e = scale - (k <<< 3);
      end else begin
        abs_s = -scale; abs_k = abs_s >>> 3; abs_e = abs_s - (abs_k <<< 3);
        if (abs_e == 0) begin k = -abs_k;     e = 0;        end
        else            begin k = -abs_k - 1; e = 8 - abs_e; end
      end

      mag = 32'h0;
      if (k >= 0) begin
        ones_count = k + 1; if (ones_count > 30) ones_count = 30;
        regime_bits = ones_count + 1; if (regime_bits > 31) regime_bits = 31;
        mag = mag | (((32'h1 << ones_count) - 32'h1) << (31 - ones_count));
        top_pos = 30 - regime_bits;
      end else begin
        zeros_count = -k; if (zeros_count >= 31) zeros_count = 30;
        regime_bits = zeros_count + 1;
        term_pos = 30 - zeros_count;
        mag = mag | (32'h1 << term_pos);
        top_pos = term_pos - 1;
      end

      exp_avail = top_pos + 1;
      exp_bits  = (exp_avail >= 3) ? 3 : (exp_avail > 0 ? exp_avail : 0);
      if (exp_bits > 0) begin
        e_aligned = e[31:0] >> (3 - exp_bits);
        mag = mag | (e_aligned << (top_pos - exp_bits + 1));
        top_pos = top_pos - exp_bits;
      end

      frac_bits = (top_pos >= 0) ? (top_pos + 1) : 0;
      if (frac_bits > 0) begin
        hi = msb - 1;
        lo = msb - frac_bits;
        width = hi - lo + 1;            // == frac_bits
        if (lo >= 0) begin
          shifted    = q >> lo;
          frac_field = shifted[31:0] & (((32'h1 << width) - 32'h1));
        end else begin
          // extract bits [0..hi] then shift up by (-lo)
          frac_field = (q[31:0] & (((32'h1 << (hi+1)) - 32'h1))) << (-lo);
        end
        mag = mag | frac_field;
      end

      if (mag >= 32'h8000_0000) mag = 32'h7FFF_FFFF;
      if (sign) mag = ((~mag) + 32'h1) & 32'h7FFF_FFFF;
      q2bp32 = ({sign, 31'h0} | mag);
      if (q2bp32 == 32'h8000_0000 && !sign) q2bp32 = 32'h7FFF_FFFF;  // NaR guard
      return q2bp32;
    end
  endfunction

  assign bp32_o = q2bp32(quire_i);
endmodule
