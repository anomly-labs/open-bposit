#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""RE-grade EXHAUSTIVE verification vectors for the quire256->bposit32 encoder
(bposit_encode.sv) — the riskiest block. Oracle = bposit16_reference.
quire256_to_bposit32 (proven bit-exact vs the Python encode + the GPU/RISC-V
port). We don't sample; we systematically cover every behavioural transition,
then add a large random sweep for breadth — the discipline that prevents a
"half-working" block reaching silicon.

Coverage (per the encoder's control flow):
  - every scale s in [-52, 52]  → all regime k / exponent e transitions AND the
    ±48/±49 saturation boundaries (MAXPOS/MINPOS/MAXNEG/MINNEG)
  - per scale: fraction patterns that stress every field-width boundary —
    zero, all-ones, single-bit walk across the top bits below the MSB,
    alternating, and randoms (regime→exp 3/2/1/0 and frac→0 transitions)
  - both signs; the lo<0 fraction branch; zero quire
  - a large uniform-random 256-bit sweep for anything systematic misses

Emits qin.hex (256-bit) / ebp.hex (32-bit) and prints a coverage report.
"""
from __future__ import annotations
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bposit_ref as r  # noqa: E402

FRAC = 96
MASK256 = (1 << 256) - 1
HERE = Path.cwd()
rng = random.Random(20260522)


def emit(qsigned, vecs):
    q = qsigned & MASK256
    bp = r.quire256_to_bposit32(qsigned) & 0xFFFFFFFF
    vecs.append((q, bp))


def main():
    vecs = []
    scales_covered = set()
    # zero
    emit(0, vecs)
    # systematic: every scale, boundary fraction patterns, both signs
    for s in range(-52, 53):
        msb = FRAC + s
        if msb < 0:
            continue
        scales_covered.add(s)
        base = 1 << msb
        fr_positions = list(range(max(0, msb - 34), msb))   # top ~34 frac bits
        patterns = [0, base - 1 if msb > 0 else 0]          # none, all-ones-below
        for p in fr_positions:                              # single-bit walk
            patterns.append(1 << p)
        if msb > 1:
            patterns.append(((base - 1) & 0xAAAA_AAAA_AAAA_AAAA_AAAA_AAAA))  # alternating
        for _ in range(24):                                 # randoms below msb
            patterns.append(rng.getrandbits(msb) if msb > 0 else 0)
        for fr in patterns:
            mag = base | (fr & (base - 1 if msb > 0 else 0))
            emit(mag, vecs)
            emit(-mag, vecs)
    # large uniform-random 256-bit sweep (full width, both interpretations)
    for _ in range(120000):
        bits = rng.getrandbits(256)
        q = bits - (1 << 256) if (bits >> 255) & 1 else bits   # signed
        emit(q, vecs)

    (HERE / "qin.hex").write_text("\n".join(f"{q:064x}" for q, _ in vecs) + "\n")
    (HERE / "ebp.hex").write_text("\n".join(f"{bp:08x}" for _, bp in vecs) + "\n")
    (HERE / "enc_dims.svh").write_text(f"localparam int NVEC = {len(vecs)};\n")
    print(f"wrote {len(vecs)} encoder vectors")
    print(f"coverage: scales {min(scales_covered)}..{max(scales_covered)} "
          f"({len(scales_covered)} incl. ±48/±49 saturation), both signs, "
          f"single-bit/all-ones/alt/random fractions, +120k uniform-random 256-bit")


if __name__ == "__main__":
    main()
