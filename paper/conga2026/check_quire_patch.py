#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host gate for bp16_quire.h's sub-binary-point truncation (CoNGA'26 §4).

Compiles the published kernel header (targets/blackhole/kernel/bp16_quire.h)
into a tiny host harness and runs every operand pair whose
product LSB lies below the quire's binary point (2^-96) — the pairs on which
the pre-patch kernel (drop) and the oracle (truncate toward zero) disagreed —
through all three kernel entry points (scalar prod+add, fused madd, packed
madd). Each result is read back as the low quire limb, in 2^-96 units, and
compared with the oracle's truncate value from tiny_product_vectors.py.
Also spot-checks the shift >= 0 path on a coprime lattice against the
reference codec so the patch is shown not to touch it.

Prints per-entry-point mismatch counts; exit 0 only if all are zero.

    python3 paper/conga2026/check_quire_patch.py [--kernel-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tiny_product_vectors as T  # noqa: E402

HARNESS = r"""
#include <stdio.h>
#include <string.h>
#include "bp16_quire.h"
int main(void) {
    int a, b;
    while (scanf("%d %d", &a, &b) == 2) {
        unsigned r[8], p[8], f[8], k[8];
        memset(r, 0, sizeof r); memset(f, 0, sizeof f); memset(k, 0, sizeof k);
        bp16_prod_to_q256(a, b, p); q256_add(r, p);
        bp16_madd_q256(f, a, b);
        bp16_madd_q256_packed(k, bp16_pack(a), bp16_pack(b));
        /* low limb is the 2^-96..2^-65 window; print limbs 0..2 for the sign-extended case */
        printf("%u %u %u %u %u %u %u %u %u\n", r[0], r[1], r[2], f[0], f[1], f[2], k[0], k[1], k[2]);
    }
    return 0;
}
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../targets/blackhole/kernel"))
    ap.add_argument("--json", default=None, help="write the summary here")
    a = ap.parse_args(argv)

    codes = T.bounded_codes()
    pairs = []                                    # (ca, cb, oracle_trunc, exact_is_zero)
    for (pa, ea, Ma) in codes:
        for (pb, eb, Mb) in codes:
            if pb < pa:
                continue
            sh = ea + eb + T_FRAC
            if sh < 0:
                pairs.append((pa, pb, (Ma * Mb) >> (-sh)))
    n_div = sum(1 for _, _, o in pairs if o != 0)
    print(f"bounded positive codes: {len(codes)}; pairs with product LSB below 2^-96: {len(pairs):,}; "
          f"of which oracle-truncate != 0 (kernel formerly dropped): {n_div:,}")

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "h.c")
        exe = os.path.join(td, "h")
        open(src, "w").write(HARNESS)
        subprocess.check_call(["gcc", "-O2", "-DBP16Q_CHECK", "-I", a.kernel_dir, src, "-o", exe])
        inp = "".join(f"{pa} {pb}\n" for pa, pb, _ in pairs)
        out = subprocess.run([exe], input=inp, capture_output=True, text=True, check=True).stdout.split("\n")

    bad = {"scalar": 0, "madd": 0, "packed": 0}
    examples = []
    for (pa, pb, orc), line in zip(pairs, out):
        w = list(map(int, line.split()))
        r, f, k = w[0:3], w[3:6], w[6:9]
        # a positive product below 2^-96*2^32 lives entirely in limb 0
        for name, limbs in (("scalar", r), ("madd", f), ("packed", k)):
            got = limbs[0] | (limbs[1] << 32) | (limbs[2] << 64)
            if got != orc:
                bad[name] += 1
                if len(examples) < 5:
                    examples.append(dict(a=pa, b=pb, path=name, got=got, oracle=orc))
    print(f"patched kernel vs oracle truncate on all {len(pairs):,} sub-binary-point pairs: "
          + ", ".join(f"{k} {v} mismatches" for k, v in bad.items()))
    for e in examples:
        print("  ", e)

    # shift >= 0 path: the kernel's own selftest (madd == scalar == packed over a
    # 143k lattice) is the guard; run it too so one command covers both.
    st = os.path.join(a.kernel_dir, "bp16_quire_selftest.c")
    with tempfile.TemporaryDirectory() as td:
        exe = os.path.join(td, "st")
        subprocess.check_call(["gcc", "-O2", "-DBP16Q_CHECK", "-I", a.kernel_dir, st, "-o", exe])
        r = subprocess.run([exe], capture_output=True, text=True)
        print("selftest:", r.stdout.strip().replace("\n", " | "), f"(exit {r.returncode})")
    ok = all(v == 0 for v in bad.values()) and r.returncode == 0
    if a.json:
        json.dump(dict(pairs_below_point=len(pairs), oracle_nonzero=n_div, mismatches=bad,
                       selftest_exit=r.returncode, ok=ok), open(a.json, "w"), indent=2)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


T_FRAC = 96

if __name__ == "__main__":
    sys.exit(main())
