# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""Exact accumulation makes a parallel scan reproducible; a blocked fp32 scan is not.

Written for the discussion in tenstorrent/tt-metal#55542, and general to any tiled prefix sum.

#55542 shows ttnn.cumsum is a naive one-at-a-time fp32 running sum, ~2949x torch's error,
growing as ~T^1.5. The standard fix is a blocked / pairwise scan, which is what torch does.

This measures a consequence of that fix nobody has stated in the thread: a blocked fp32 scan's
OUTPUT DEPENDS ON THE BLOCKING. Change the block size (or the core count, or the tile width that
drives it) and the bits change, because floating-point addition is not associative. So a
multi-core cumsum would not agree with a single-core one, and a 32-wide tile would not agree with
a 1024-wide one, on the same input.

An exactly-accumulated scan does not have that property: every prefix is the correctly rounded
value of the exact sum, so every blocking gives the identical answer, bit for bit.

Exact reference: each fp32 is a dyadic rational, so scale by 2^149 (the smallest subnormal) and
prefix-sum in Python integers. No rounding at all until the final round-to-nearest-even back to
fp32 at each output position.

No silicon needed; this is a host-side statement about the algorithm.
"""
from __future__ import annotations

import argparse
import struct

import numpy as np

SCALE = 149  # 2^-149 is the smallest positive fp32 subnormal


def f32_to_exact_int(x: np.ndarray) -> list[int]:
    """Exact integer representation of each fp32 value, in units of 2^-149."""
    out = []
    for v in x.tolist():
        m, e = np.frexp(np.float32(v))
        # v = m * 2^e with 0.5 <= |m| < 1; m * 2^24 is an exact integer for fp32
        mi = int(round(float(m) * (1 << 24)))
        sh = int(e) - 24 + SCALE
        out.append(mi << sh if sh >= 0 else mi >> (-sh))
    return out


def exact_int_to_f32(n: int) -> np.float32:
    """Round an exact integer (units of 2^-149) to fp32, nearest-even."""
    if n == 0:
        return np.float32(0.0)
    neg = n < 0
    n = -n if neg else n
    bl = n.bit_length()
    keep = 24
    if bl > keep:                       # round to 24 significant bits, nearest-even
        drop = bl - keep
        head = n >> drop
        rem = n & ((1 << drop) - 1)
        half = 1 << (drop - 1)
        if rem > half or (rem == half and (head & 1)):
            head += 1
            if head.bit_length() > keep:
                head >>= 1
                drop += 1
        n, e = head, drop - SCALE
    else:
        e = -SCALE
    v = np.float32(np.ldexp(float(n), e))
    return np.float32(-v) if neg else v


def exact_cumsum(x: np.ndarray) -> np.ndarray:
    ints = f32_to_exact_int(x)
    out = np.empty(len(ints), dtype=np.float32)
    acc = 0
    for i, n in enumerate(ints):
        acc += n
        out[i] = exact_int_to_f32(acc)
    return out


def naive_cumsum(x: np.ndarray) -> np.ndarray:
    """One element at a time in fp32 — what #55542 shows the device does."""
    out = np.empty_like(x)
    acc = np.float32(0.0)
    for i in range(x.size):
        acc = np.float32(acc + x[i])
        out[i] = acc
    return out


def blocked_cumsum(x: np.ndarray, block: int) -> np.ndarray:
    """The standard fix, and the shape any tiled / multi-core scan takes: each block is scanned
    locally, the block totals are scanned to give per-block offsets, and the offset is added once
    to each local prefix. `block` is the tile width or the per-core span."""
    n = x.size
    nb = (n + block - 1) // block
    local = np.empty_like(x)
    totals = np.empty(nb, dtype=np.float32)
    for b in range(nb):
        s0, s1 = b * block, min((b + 1) * block, n)
        acc = np.float32(0.0)
        for i in range(s0, s1):
            acc = np.float32(acc + x[i])
            local[i] = acc
        totals[b] = acc
    offs = np.empty(nb, dtype=np.float32)          # exclusive scan of block totals
    acc = np.float32(0.0)
    for b in range(nb):
        offs[b] = acc
        acc = np.float32(acc + totals[b])
    out = np.empty_like(x)
    for b in range(nb):
        s0, s1 = b * block, min((b + 1) * block, n)
        out[s0:s1] = (offs[b] + local[s0:s1]).astype(np.float32)
    return out


def _ordered(v: np.ndarray) -> np.ndarray:
    """Map fp32 bits to a monotonic integer so a difference counts representable steps."""
    u = v.astype(np.float32).view(np.uint32).astype(np.int64)
    mag = u & 0x7FFFFFFF
    return np.where(u >> 31 != 0, -mag, mag)


def ulp_err(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.abs(_ordered(got) - _ordered(ref))


def bits_differ(a: np.ndarray, b: np.ndarray) -> int:
    return int((a.view(np.int32) != b.view(np.int32)).sum())


def make_signal(T: int, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "iid":
        return rng.standard_normal(T).astype(np.float32)
    # #55542's real case: nn.Upsample(scale_factor=256), mode nearest -> piecewise constant
    base = rng.standard_normal((T + 255) // 256).astype(np.float32)
    return np.repeat(base, 256)[:T].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-T", type=int, default=72192, help="scan length (#55542 uses 72192)")
    ap.add_argument("--kind", default="upsampled", choices=["iid", "upsampled"])
    ap.add_argument("--blocks", default="32,64,256,1024")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    x = make_signal(a.T, a.kind, a.seed)
    ref = exact_cumsum(x)
    blocks = [int(b) for b in a.blocks.split(",")]

    peak = float(np.abs(ref).max())
    print(f"T={a.T} signal={a.kind} seed={a.seed}  peak |prefix| = {peak:.4g}")
    print(f"{'scan':>22}  {'max abs err':>12}  {'mean ULP':>9}  {'sign flips':>10}")

    def row(name, o):
        flips = int(((np.sign(o.astype(np.float64)) * np.sign(ref.astype(np.float64))) < 0).sum())
        print(f"{name:>22}  {np.abs(o - ref).max():>12.4e}  {ulp_err(o, ref).mean():>9.2f}  {flips:>10d}")

    naive = naive_cumsum(x)
    row("naive serial fp32", naive)
    outs = {}
    for b in blocks:
        o = blocked_cumsum(x, b)
        outs[b] = o
        row("blocked fp32 (%d)" % b, o)
    print(f"{'exact accumulation':>22}  {0.0:>12.4e}  {0.0:>9.2f}  {0:>10d}   (correctly rounded)")
    print("\n  ULP is reported as a mean: where a prefix passes near zero the relative error is\n"
          "  unbounded, so a max-ULP figure there says more about the zero crossing than the scan.")

    print("\ndo two blockings of the SAME input agree bit for bit?")
    base = blocks[0]
    for b in blocks[1:]:
        d = bits_differ(outs[base], outs[b])
        print(f"  block {base:>4} vs block {b:>4}: {d:>7d} / {a.T} outputs differ  ({100.0*d/a.T:5.2f}%)")
    print(f"  exact accumulation: 0 / {a.T} differ for ANY blocking — the sum is order-independent")


if __name__ == "__main__":
    main()
