#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""Regression for the 2026-09-24 quire-grid finding.

THE BUG, as it was
``bposit16_to_quire`` ended in ``int(f)``, which silently dropped every bit below
the quire's 2^-96 grid. Only SIX of the 65,534 non-special bposit16 codes have
values that fine — the extremes of the unbounded envelope — but for all six the
loss was total: the quire came out all-zero. So ``bposit16_add(0x0001, 0x0001)``
returned ZERO. Two positive numbers added to nothing.

A rounding inside an accumulator whose entire purpose is that it does not round is
the one defect this file must not have, however few inputs reach it.

THE FIX
``_to_quire_grid`` raises :class:`QuireGridError` instead, and the two internal
callers that can reach it (``bposit16_add`` and ``bposit16_mul_via_log_exp2``) fall
back to the exact rational answer, with a comment saying that a DEVICE with a Q96
quire genuinely cannot — which is a direct argument for the bounded envelope, where
minpos x minpos is exactly one quire LSB and nothing can fall off the bottom.

Every assertion below is checked against exact ``Fraction`` arithmetic, not against
another code path in this file.

    python3 reference/quire_grid_test.py
"""
from fractions import Fraction

import bposit_ref as bp

QF = bp.QUIRE_FRAC_BITS
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def exact(code):
    d = bp.decode_bposit16(code)
    return Fraction(0) if d.is_special else bp.decoded_to_fraction(d)


def on_grid(value):
    return (value * (1 << QF)).denominator == 1


def main():
    print("quire_grid_test: the 2^-96 grid is respected rather than truncated")

    # --- which codes are affected, stated as a number rather than "a few" -----
    subgrid = [c for c in range(1 << 16) if c not in (bp.ZERO, bp.NAR) and not on_grid(exact(c))]
    check("exactly six bposit16 codes are finer than the grid", len(subgrid) == 6, str(len(subgrid)))
    check(
        "and they are the extremes of the unbounded envelope",
        all(abs(exact(c)) <= Fraction(1, 1 << 96) or abs(exact(c)) >= (1 << 96) for c in subgrid),
    )

    # --- the conversion refuses rather than truncates -------------------------
    raised = 0
    for c in subgrid:
        try:
            bp.bposit16_to_quire(c)
        except bp.QuireGridError:
            raised += 1
    check("every one of them raises QuireGridError", raised == len(subgrid), f"{raised}/{len(subgrid)}")
    check(
        "strict=False still truncates, for a caller that has measured it does not care",
        bp.bposit16_to_quire(subgrid[0], strict=False) == 0,
    )

    # --- everything else converts exactly ------------------------------------
    exact_count = wrong = 0
    for c in range(1 << 16):
        if c in (bp.ZERO, bp.NAR):
            continue
        try:
            q = bp.bposit16_to_quire(c)
        except bp.QuireGridError:
            continue
        if Fraction(q, 1 << QF) == exact(c):
            exact_count += 1
        else:
            wrong += 1
    check(
        "every other code converts to its EXACT rational value",
        wrong == 0 and exact_count == 65534 - len(subgrid),
        f"{exact_count} exact, {wrong} wrong",
    )

    # --- the bug itself, in its clearest form --------------------------------
    got = bp.bposit16_add(0x0001, 0x0001)
    want = bp.encode_bposit16(exact(0x0001) * 2, mode="truncate")
    check(
        "add(minpos, minpos) is minpos and not ZERO (it returned ZERO before the fix)",
        got == want and got != bp.ZERO,
        f"0x{got:04x} vs 0x{want:04x}",
    )
    bad = [c for c in subgrid if bp.bposit16_add(c, c) != bp.encode_bposit16(exact(c) * 2, mode="truncate")]
    check("add(c, c) is exact for all six sub-grid codes", not bad, str([hex(c) for c in bad]))

    # --- and nothing in the normal range moved -------------------------------
    import random

    rng = random.Random(20260924)
    bad_add = bad_mul = 0
    for _ in range(4000):
        a = rng.randrange(1, 1 << 16)
        b = rng.randrange(1, 1 << 16)
        if bp.NAR in (a, b):
            continue
        if bp.bposit16_add(a, b) != bp.encode_bposit16(exact(a) + exact(b), mode="truncate"):
            bad_add += 1
        if bp.bposit16_mul(a, b) != bp.encode_bposit16(exact(a) * exact(b), mode="truncate"):
            bad_mul += 1
    check("4000 random add pairs match exact rational arithmetic", bad_add == 0, str(bad_add))
    check("4000 random mul pairs match exact rational arithmetic", bad_mul == 0, str(bad_mul))

    # --- the narrower widths cannot reach it, measured rather than argued ----
    for label, codes, conv in (
        ("bposit8", range(1 << 8), bp.bposit8_to_quire),
        ("bposit4", range(1 << 4), bp.bposit4_to_quire),
        ("aiposit5", range(1 << 5), bp.aiposit5_to_quire),
    ):
        hits = 0
        for c in codes:
            try:
                conv(c)
            except bp.QuireGridError:
                hits += 1
            except Exception:  # NaR and friends are not this test's business
                pass
        check(f"no {label} code falls below the grid", hits == 0, f"{hits} of {len(codes)}")

    # --- the bounded envelope, which is the point ----------------------------
    minpos48 = Fraction(1, 1 << 48)
    check(
        "bounded minpos x minpos is exactly one quire LSB, so nothing bounded can fall off",
        (minpos48 * minpos48 * (1 << QF)) == 1,
    )

    print("quire_grid_test: " + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURE(S): {FAILURES}"))
    raise SystemExit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
