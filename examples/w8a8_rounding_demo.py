# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""w8a8_rounding_demo.py — why b-posit8 W8A8 is BOTH reproducible AND accurate.

Two properties of b-posit8 W8A8 inference, on a self-contained synthetic deep MLP
(deterministic seed, no model download):

1. ROUNDING MODE MATTERS. b-posit's encode rounds toward zero, a *biased* rounding
   that systematically shrinks every value; the bias compounds layer over layer.
   Rounding to the NEAREST representable posit (the standard, unbiased choice) keeps
   the error flat. We push a signal through L layers of bp8 W8A8 (power-of-two
   per-row scales) and print the output error vs an fp64 reference for each mode.

2. REPRODUCIBILITY IS FREE. Round-to-nearest against the fixed posit lattice is
   deterministic — the same integer codes on any hardware / any tiling order — so the
   accuracy gain costs nothing in the bit-reproducibility that is b-posit's whole point.

Run:  python examples/w8a8_rounding_demo.py   (needs numpy)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))
import bposit_ref as bp                      # noqa: E402
import bposit_fast as bf                     # noqa: E402

_GRID = np.array(sorted(set(abs(float(bp.code_value(c, "bp8"))) for c in range(256) if c != 0x80)))
_V2C = {float(bp.code_value(c, "bp8")): c for c in range(256) if c != 0x80}


def _po2(M, axis):
    rms = np.sqrt(np.mean(M ** 2, axis=axis, keepdims=True))
    return np.where(rms > 0, np.round(np.log2(np.where(rms > 0, rms, 1.0))), 0.0)


def w8a8(x, W, *, nearest):
    """x[T,in] @ W[out,in]^T in bp8 W8A8 with power-of-two scales (fp accumulate)."""
    xe, we = 2.0 ** _po2(x, 1), 2.0 ** _po2(W, 1)
    xq = bf.quantize_bp8(x / xe, nearest=nearest) * xe
    wq = bf.quantize_bp8(W / we, nearest=nearest) * we
    return xq @ wq.T


def codes_of(x, *, nearest):
    v = bf.quantize_bp8(x, nearest=nearest)
    return np.vectorize(lambda t: _V2C[float(t)])(v)


def main() -> int:
    rng = np.random.default_rng(0)
    L, d, T = 8, 128, 16
    Ws = [rng.standard_normal((d, d)) * (1.0 / np.sqrt(d)) for _ in range(L)]
    x0 = rng.standard_normal((T, d)) * 0.5

    def run(mode):
        x = x0.copy()
        for W in Ws:
            if mode == "fp64":
                x = x @ W.T
            else:
                x = w8a8(x, W, nearest=(mode == "nearest"))
            x = np.maximum(x, 0.0)                       # ReLU
        return x

    ref = run("fp64")

    def relerr(y):
        return float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-30))

    print(f"synthetic MLP: {L} layers, width {d}, {T} tokens, bp8 W8A8, power-of-two scales\n")
    print(f"  output rel-err vs fp64 after {L} layers:")
    et, en = relerr(run("truncate")), relerr(run("nearest"))
    print(f"    truncate-toward-zero (encode default) : {et:.4e}")
    print(f"    round-to-nearest     (posit standard) : {en:.4e}")
    print(f"    -> round-to-nearest is {et/max(en,1e-30):.1f}x more accurate\n")

    # reproducibility: same codes across a re-run and a shuffled tiling
    c1 = codes_of(x0, nearest=True)
    c2 = codes_of(x0, nearest=True)
    flat = x0.ravel(); idx = np.arange(flat.size); rng.shuffle(idx)
    out = np.empty_like(flat)
    for s in range(0, flat.size, 97):
        j = idx[s:s + 97]; out[j] = bf.quantize_bp8(flat[j], nearest=True)
    c3 = np.vectorize(lambda t: _V2C[float(t)])(out).reshape(x0.shape)
    repro = bool(np.array_equal(c1, c2) and np.array_equal(c1, c3))
    print(f"  reproducibility (round-to-nearest): identical codes across re-run + shuffled tiling: {repro}")

    ok = en < et and repro
    print("\nw8a8_rounding_demo:", "PASS (nearest more accurate AND reproducible)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
