#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tables + exhaustive expected results for the elementwise bp4/aip5 ALU RTL
(bposit_lowp_alu.sv). The RTL COMPUTES (decode -> fixed-point -> total_e
saturation + in-range threshold encode); these expected tables come from the
INDEPENDENT reference (bposit4_mul/add, aiposit5_mul/add) — so an all-pairs
match is a real equivalence proof, not a table replay.

Per format emits: <f>_dec.hex (decode Q16, 32-bit), <f>_inrv.hex (in-range
values Q16, 32-bit), <f>_inrc.hex (in-range codes), <f>_mul.hex / <f>_add.hex
(expected, all pairs), <f>_alu.svh (params).
"""
from __future__ import annotations
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bposit_ref as r  # noqa: E402

FRAC = 16
HERE = Path.cwd()

FMTS = {
    "aip5": dict(nbits=5, decode=r.decode_aiposit5, tofrac=r.decoded_to_fraction_5,
                 mul=r.aiposit5_mul, add=r.aiposit5_add, max_e=r.MAXE_5,
                 NAR=r.NAR_5, ZERO=r.ZERO_5),
    "bp4":  dict(nbits=4, decode=r.decode_bposit4, tofrac=r.decoded_to_fraction_4,
                 mul=r.bposit4_mul, add=r.bposit4_add, max_e=r.MAXE_4,
                 NAR=r.NAR_4, ZERO=r.ZERO_4),
}


def main():
    for name, f in FMTS.items():
        n = 1 << f["nbits"]
        dec, pos = [], []
        for c in range(n):
            d = f["decode"](c)
            v = 0 if d.is_special else int(f["tofrac"](d) * (1 << FRAC))
            dec.append(v & 0xFFFFFFFF)
            if not d.is_special and f["tofrac"](d) > 0:
                pos.append((f["tofrac"](d), int(f["tofrac"](d) * (1 << FRAC)), c))
        pos.sort()
        minc, maxc = pos[0][2], pos[-1][2]
        lo, hi = Fraction(1, 1 << f["max_e"]), Fraction(1 << f["max_e"])
        inr = [(fx, c) for v, fx, c in pos if lo <= v <= hi]

        (HERE / f"{name}_dec.hex").write_text("\n".join(f"{v:08x}" for v in dec) + "\n")
        (HERE / f"{name}_inrv.hex").write_text("\n".join(f"{fx:08x}" for fx, _ in inr) + "\n")
        (HERE / f"{name}_inrc.hex").write_text("\n".join(f"{c:02x}" for _, c in inr) + "\n")
        mul = [f["mul"](a, b) & (n - 1) for a in range(n) for b in range(n)]
        add = [f["add"](a, b) & (n - 1) for a in range(n) for b in range(n)]
        (HERE / f"{name}_mul.hex").write_text("\n".join(f"{v:02x}" for v in mul) + "\n")
        (HERE / f"{name}_add.hex").write_text("\n".join(f"{v:02x}" for v in add) + "\n")
        (HERE / f"{name}_alu.svh").write_text(
            f"localparam int {name.upper()}_NBITS = {f['nbits']};\n"
            f"localparam int {name.upper()}_N = {n};\n"
            f"localparam int {name.upper()}_NINR = {len(inr)};\n"
            f"localparam int {name.upper()}_MAXE = {f['max_e']};\n"
            f"localparam logic [7:0] {name.upper()}_NAR = 8'h{f['NAR']:02x};\n"
            f"localparam logic [7:0] {name.upper()}_ZERO = 8'h{f['ZERO']:02x};\n"
            f"localparam logic [7:0] {name.upper()}_MINP = 8'h{minc:02x};\n"
            f"localparam logic [7:0] {name.upper()}_MAXP = 8'h{maxc:02x};\n")
        print(f"{name}: dec[{n}] inr[{len(inr)}] mul/add[{n*n}] tables written")


if __name__ == "__main__":
    main()
