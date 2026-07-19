# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
# b-posit / AI-Posit — FAST exact-quire dot, numpy-vectorized.
#
# Bit-identical to bposit_ref.dot_at_precision (the pure-Python per-element quire),
# but ~60x faster, so exact-quire evaluation is tractable on real models. The trick:
# every code's value is dyadic (mantissa * 2^exponent), so a product lands at a fixed
# bit-shift in the 256-bit quire. We vectorize the decode with a per-format
# (mantissa, exponent) LUT, group products by bit-shift, and combine each shift group
# with a single big-int shift-add. Underflowing terms (shift < 0) are truncated
# toward zero, exactly as the reference's per-term int() does.
#
# Pure integer + numpy; the result is identical on any GPU / CPU / RISC-V core --
# the whole point of b-posit's reproducibility.
from __future__ import annotations
import numpy as np
from fractions import Fraction

import bposit_ref as _bp

QUIRE_FRAC_BITS = _bp.QUIRE_FRAC_BITS
_NBITS = {"bp4": 4, "aip5": 5, "bp8": 8}

_LUT_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _lut(fmt: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-format (signed mantissa, exponent) LUT: value(code) == mant[code] * 2**exp[code].
    Zero and NaR map to mant=0 (they contribute nothing, matching code_value)."""
    if fmt not in _LUT_CACHE:
        n = 1 << _NBITS[fmt]
        mant = np.zeros(n, dtype=np.int64)
        exp = np.zeros(n, dtype=np.int64)
        for c in range(n):
            f = _bp.code_value(c, fmt)  # exact Fraction (0 for zero/NaR)
            if f == 0:
                continue
            den = f.denominator
            assert den & (den - 1) == 0, f"non-dyadic {fmt} value for code {c}: {f}"
            mant[c] = f.numerator
            exp[c] = -(den.bit_length() - 1)
        _LUT_CACHE[fmt] = (mant, exp)
    return _LUT_CACHE[fmt]


def dot_quire(a_codes, b_codes, fmt: str) -> int:
    """Exact 256-bit quire accumulator for the dot of two code vectors — identical to
    the integer `q` inside bposit_ref.dot_at_precision, computed vectorized."""
    mant, exp = _lut(fmt)
    mask = (1 << _NBITS[fmt]) - 1
    a = np.asarray(a_codes, dtype=np.int64) & mask
    b = np.asarray(b_codes, dtype=np.int64) & mask
    P = mant[a] * mant[b]                       # signed product mantissa
    shift = exp[a] + exp[b] + QUIRE_FRAC_BITS
    q = 0
    pos = shift >= 0
    sp, ss = P[pos], shift[pos]
    if ss.size:
        order = np.argsort(ss, kind="stable")
        sp, ss = sp[order], ss[order]
        uniq, idx = np.unique(ss, return_index=True)
        bnd = list(idx) + [ss.size]
        for k in range(len(uniq)):
            s = int(uniq[k])
            tot = int(sp[bnd[k]:bnd[k + 1]].sum())
            if tot:
                q += tot << s
    # shift < 0: truncate toward zero per term (matches reference int())
    for Pi, si in zip(P[~pos].tolist(), shift[~pos].tolist()):
        sh = -si
        q += -((-Pi) >> sh) if Pi < 0 else (Pi >> sh)
    return q


def dot(a_codes, b_codes, fmt: str, out_fmt: str = "bp32") -> int:
    """Exact dot, rounded once to out_fmt — identical to dot_at_precision(...)."""
    return _bp._ENCODE_OUT[out_fmt](dot_quire(a_codes, b_codes, fmt))


def matmul_quire(A_codes, B_codes, fmt: str):
    """Exact quire per output of A[T,K] @ B[N,K]^T (object array of Python ints)."""
    A = np.asarray(A_codes, dtype=np.int64)
    B = np.asarray(B_codes, dtype=np.int64)
    out = np.empty((A.shape[0], B.shape[0]), dtype=object)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            out[i, j] = dot_quire(A[i], B[j], fmt)
    return out


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    ok = True
    for fmt in ("bp4", "aip5", "bp8"):
        n = 1 << _NBITS[fmt]
        for _ in range(300):
            K = int(rng.integers(1, 400))
            a = rng.integers(0, n, size=K, dtype=np.int64)
            b = rng.integers(0, n, size=K, dtype=np.int64)
            ref = _bp.dot_at_precision(a.tolist(), b.tolist(), fmt, "bp32")
            got = dot(a, b, fmt, "bp32")
            if ref != got:
                ok = False
                print(f"MISMATCH fmt={fmt} K={K}: ref={ref} got={got}")
                break
        print(f"  {fmt}: bit-exact vs dot_at_precision {'OK' if ok else 'FAIL'}")
        if not ok:
            break
    # speed sanity (bp8)
    K, N = 512, 256
    A = rng.integers(0, 256, size=(N, K), dtype=np.int64)
    B = rng.integers(0, 256, size=K, dtype=np.int64)
    t0 = time.time()
    for i in range(N):
        dot_quire(A[i], B, "bp8")
    tv = time.time() - t0
    t0 = time.time()
    for i in range(N):
        _bp.dot_at_precision(A[i].tolist(), B.tolist(), "bp8", "bp32")
    tr = time.time() - t0
    print(f"  speed ({N} dots K={K}): fast {tv*1e3:.0f} ms vs reference {tr*1e3:.0f} ms -> {tr/tv:.0f}x")
    print("bposit_fast:", "ALL BIT-EXACT" if ok else "FAILED")
