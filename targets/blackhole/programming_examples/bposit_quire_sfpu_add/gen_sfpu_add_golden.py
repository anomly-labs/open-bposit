#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate the LANE-PARALLEL q256_add golden for the SFPU milestone.

This bakes 32 lanes x (q[8], x[8], expected_sum[8]) for the first SFPU-vectorized
exact-quire primitive: 32 INDEPENDENT 256-bit accumulators (one per SFPU lane),
each doing  q += x  with full inter-limb ripple carry.

The expected sum is computed by REPLICATING THE SCALAR REFERENCE  q256_add  EXACTLY
(kernel/bp16_quire.h:62-65):

    unsigned long long carry = 0;
    for i in 0..7:
        t = (u64)q[i] + x[i] + carry
        q[i] = (u32)t
        carry = t >> 32

i.e. plain 256-bit two's-complement (mod 2^256) addition. The vector kernel must
be bit-identical to this PER LANE. We deliberately do NOT use the bp16 decode /
product path here — this milestone isolates and proves the carry vector alone.

Belt-and-suspenders: the HOST program re-derives the same golden at startup by
calling the real C q256_add over each lane (so a drifted baked golden is caught
before the device run), exactly like the working bposit_quire_reduce example.

Adversarial lanes (the carry chain is everything):
  - zero + zero
  - single-limb add, no carry
  - 0xFFFFFFFF + 1 in limb0 -> carry ripples ALL THE WAY to limb7 (the worst case)
  - all-limbs 0xFFFFFFFF + 1 in limb0 -> full 256-bit wraparound to 0 (mod 2^256)
  - two's-complement negative (high bit set) + positive -> signed accumulate
  - carry that dies mid-chain (limb3 absorbs it)
  - alternating patterns, max+max per limb (each limb overflows AND eats a carry-in,
    proving BOTH carry sources are detected: q_i+x_i overflow, and +carry overflow)
"""
import sys
import json
import random

LIMBS = 8
MASK32 = 0xFFFFFFFF
MASK256 = (1 << 256) - 1


def limbs_to_int(q):
    v = 0
    for i in range(LIMBS):
        v |= (q[i] & MASK32) << (32 * i)
    return v


def int_to_limbs(v):
    v &= MASK256
    return [(v >> (32 * i)) & MASK32 for i in range(LIMBS)]


def q256_add_ref(q, x):
    """EXACT replica of the C scalar q256_add (bp16_quire.h:62-65)."""
    out = [0] * LIMBS
    carry = 0
    for i in range(LIMBS):
        t = (q[i] & MASK32) + (x[i] & MASK32) + carry
        out[i] = t & MASK32
        carry = t >> 32
    return out


def neg_limbs(v_pos):
    """Two's-complement negation over 256 bits, as limbs."""
    return int_to_limbs((-v_pos) & MASK256)


# --- 32 adversarial lanes: (q[8], x[8]) -------------------------------------
ALL1 = MASK32
lanes = []

# 0: zero + zero
lanes.append(([0] * 8, [0] * 8))
# 1: single-limb, no carry
lanes.append(([1, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0]))
# 2: limb0 overflow -> carry into limb1 only
lanes.append(([ALL1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]))
# 3: carry ripples ALL the way limb0..limb7 (the worst case)
lanes.append(([ALL1, ALL1, ALL1, ALL1, ALL1, ALL1, ALL1, 0],
              [1, 0, 0, 0, 0, 0, 0, 0]))
# 4: FULL 256-bit wraparound to 0 (mod 2^256): all-ones + 1
lanes.append(([ALL1] * 8, [1, 0, 0, 0, 0, 0, 0, 0]))
# 5: all-ones + all-ones  => 0xFFFF...FFFE (every limb overflows AND eats carry-in)
lanes.append(([ALL1] * 8, [ALL1] * 8))
# 6: carry dies at limb3 (limb3 is not all-ones, absorbs the carry)
lanes.append(([ALL1, ALL1, ALL1, 0x7FFFFFFF, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0]))
# 7: two's-complement NEGATIVE (-5) + positive (+3) => -2
lanes.append((neg_limbs(5), [3, 0, 0, 0, 0, 0, 0, 0]))
# 8: negative + negative => more negative (-7 + -9 = -16)
lanes.append((neg_limbs(7), neg_limbs(9)))
# 9: fraction-point region: place at limb3 (bit 96) like a real quire contribution
lanes.append(([0, 0, 0, 0x00000001, 0, 0, 0, 0],
              [0, 0, 0, 0xFFFFFFFF, 0, 0, 0, 0]))
# 10: high limb (sign) set, two's-comp big negative + small positive
lanes.append(([0, 0, 0, 0, 0, 0, 0, 0x80000000],
              [1, 0, 0, 0, 0, 0, 0, 0]))
# 11: carry-in overflow source ONLY: limb_i = 0xFFFFFFFF + 0 + carry(=1) -> overflow
#     limb0 overflows producing carry; limb1 = 0xFFFFFFFF + 0 + 1 -> overflow (the
#     "+carry overflowed" path, distinct from q_i+x_i overflow).
lanes.append(([ALL1, ALL1, ALL1, ALL1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0]))
# 12: alternating limbs
lanes.append([0xAAAAAAAA, 0x55555555, 0xAAAAAAAA, 0x55555555,
              0xAAAAAAAA, 0x55555555, 0xAAAAAAAA, 0x55555555]) \
    if False else lanes.append(
        ([0xAAAAAAAA, 0x55555555, 0xAAAAAAAA, 0x55555555,
          0xAAAAAAAA, 0x55555555, 0xAAAAAAAA, 0x55555555],
         [0x55555555, 0xAAAAAAAA, 0x55555555, 0xAAAAAAAA,
          0x55555555, 0xAAAAAAAA, 0x55555555, 0xAAAAAAAA]))
# 13: max each limb + 1 each limb (each limb: 0xFFFFFFFF + 1 = overflow + carry chain)
lanes.append(([ALL1] * 8, [1, 1, 1, 1, 1, 1, 1, 1]))
# 14: a realistic edge-set quire (from bp16_quire_edge.json) + its negation -> 0
edge_q = [0x00000000, 0x00000000, 0x80000100, 0x00fffffe, 0, 0, 0, 0]
lanes.append((edge_q, neg_limbs(limbs_to_int(edge_q))))
# 15: same edge quire + itself (doubling, exercises mid-limb carries)
lanes.append((edge_q, list(edge_q)))

# 16..31: deterministic pseudo-random adversarial lanes (seeded, reproducible),
# biased toward all-ones limbs so carries are common.
rng = random.Random(0x5DC0FFEE)


def rand_limb():
    r = rng.random()
    if r < 0.30:
        return ALL1            # bias to all-ones (carry generators)
    if r < 0.45:
        return 0
    if r < 0.55:
        return 0x80000000      # sign bit
    return rng.getrandbits(32)


for _ in range(32 - len(lanes)):
    q = [rand_limb() for _ in range(LIMBS)]
    x = [rand_limb() for _ in range(LIMBS)]
    lanes.append((q, x))

assert len(lanes) == 32, len(lanes)

# --- compute expected sums via the EXACT scalar reference --------------------
exp = [q256_add_ref(q, x) for (q, x) in lanes]

# Cross-check against Python big-int mod-2^256 add (independent oracle).
for (q, x), e in zip(lanes, exp):
    big = (limbs_to_int(q) + limbs_to_int(x)) & MASK256
    assert limbs_to_int(e) == big, "ref q256_add disagrees with big-int mod 2^256"

# --- emit the C header -------------------------------------------------------
# Layout in the device buffers is LANE-MAJOR within each limb-vector, i.e. for the
# SFPU a "limb vector" is 32 lanes of one limb. So Q[limb][lane] and X[limb][lane]
# and EXP[limb][lane]. The host marshals one 32-lane vector per limb.
NLANES = 32


def fmt_u32(v):
    return "0x%08xu" % (v & MASK32)


def emit_table(name, getter):
    out = []
    out.append("static const uint32_t %s[%d][%d] = {" % (name, LIMBS, NLANES))
    for limb in range(LIMBS):
        row = ", ".join(fmt_u32(getter(lane, limb)) for lane in range(NLANES))
        out.append("    { %s }," % row)
    out.append("};")
    return "\n".join(out)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout"
    lines = []
    lines.append("/* sfpu_add_golden_cases.h — GENERATED by gen_sfpu_add_golden.py.")
    lines.append(" * DO NOT hand-edit. 32 lanes x (q[8], x[8], expected[8]) for the")
    lines.append(" * LANE-PARALLEL exact q256_add SFPU milestone. Expected sums are the")
    lines.append(" * EXACT scalar q256_add (bp16_quire.h:62-65) per lane, cross-checked")
    lines.append(" * vs Python big-int mod-2^256. Layout is [limb][lane] (limb-vector major)")
    lines.append(" * so the host can marshal one 32-lane vector per limb to the SFPU.")
    lines.append(" */")
    lines.append("#ifndef SFPU_ADD_GOLDEN_CASES_H")
    lines.append("#define SFPU_ADD_GOLDEN_CASES_H")
    lines.append("")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("#define QSFPU_LIMBS %d" % LIMBS)
    lines.append("#define QSFPU_LANES %d" % NLANES)
    lines.append("")
    lines.append(emit_table("QSFPU_Q", lambda lane, limb: lanes[lane][0][limb]))
    lines.append("")
    lines.append(emit_table("QSFPU_X", lambda lane, limb: lanes[lane][1][limb]))
    lines.append("")
    lines.append(emit_table("QSFPU_EXP", lambda lane, limb: exp[lane][limb]))
    lines.append("")
    lines.append("#endif /* SFPU_ADD_GOLDEN_CASES_H */")
    text = "\n".join(lines) + "\n"
    if path == "/dev/stdout":
        sys.stdout.write(text)
    else:
        with open(path, "w") as f:
            f.write(text)
    # human-readable summary to stderr
    sys.stderr.write("generated %d lanes x %d limbs; sample lane3 (full ripple): "
                     "q=%s x=%s exp=%s\n" % (
                         NLANES, LIMBS,
                         [hex(v) for v in lanes[3][0]],
                         [hex(v) for v in lanes[3][1]],
                         [hex(v) for v in exp[3]]))


if __name__ == "__main__":
    main()
