#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""EXHAUSTIVE bposit8 single-product verification vectors: every one of the
65,536 bp8 x bp8 operand pairs -> exact quire -> bposit32, from the reference
oracle. Run through the real RTL (bposit8_dot, K=1) this covers the ENTIRE bp8
input space of the production W8A8 MAC+encode path — no sampling.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bposit_ref as r  # noqa: E402

HERE = Path.cwd()


def main():
    A, B, EBP = [], [], []
    for a in range(256):
        for b in range(256):
            A.append(a); B.append(b)
            q = int(r.code_value(a, "bp8") * r.code_value(b, "bp8") * (1 << r.QUIRE_FRAC_BITS))
            EBP.append(r.quire256_to_bposit32(q) & 0xFFFFFFFF)
    (HERE / "bp8x_a.hex").write_text("\n".join(f"{c:02x}" for c in A) + "\n")
    (HERE / "bp8x_b.hex").write_text("\n".join(f"{c:02x}" for c in B) + "\n")
    (HERE / "bp8x_ebp.hex").write_text("\n".join(f"{v:08x}" for v in EBP) + "\n")
    (HERE / "bp8x_dims.svh").write_text(f"localparam int NPAIR = {len(A)};\n")
    print(f"wrote {len(A)} exhaustive bp8 product vectors (all 256x256 pairs)")


if __name__ == "__main__":
    main()
