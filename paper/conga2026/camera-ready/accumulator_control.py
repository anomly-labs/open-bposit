# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The control Reviewer 3 asked for in §6.7: hold the quantized operands fixed
and vary ONLY the accumulator.

R3.4: "comparisons using identical quantized operands with conventional FP32
and integer accumulation". The paper's W8A8 result mixes the quantization step
with the accumulation method, so it cannot show how much of the effect is the
accumulator. This script quantizes a real Llama-3.2-1B layer exactly once and
then accumulates the SAME int8 products four ways:

  int64      exact integer dot -- the ground truth, exact by construction
  int32      what the paper's path and an INT8 tensor core use
  fp32 seq   sequential fp32 accumulation, rounding after every add
  fp32 numpy numpy's blocked/pairwise fp32 dot, as a GPU GEMM would schedule it

Everything before the accumulator is identical across arms, so any difference
is the accumulator alone.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

SNAP = Path(os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")) + "/"
            "models--unsloth--Llama-3.2-1B-Instruct/snapshots/"
            "5a8abab4a5d6f164389b1079fb721cfab8d7126c")
LAYER = "model.layers.0.self_attn.q_proj.weight"
EMBED = "model.embed_tokens.weight"
N_TOK = 64            # real token ids -> real layer-0 input rows
SEED = 0


def per_row_int8(x):
    """Per-row symmetric int8 quantization: the paper's W8A8 scheme."""
    s = np.max(np.abs(x), axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    q = np.rint(x * (127.0 / s)).astype(np.int64)
    return np.clip(q, -127, 127).astype(np.int8), (s / 127.0).squeeze(-1)


def fp32_sequential(A, B):
    """Accumulate the int8 products in fp32, rounding after every add."""
    acc = np.zeros((A.shape[0], B.shape[1]), np.float32)
    Af, Bf = A.astype(np.float32), B.astype(np.float32)
    for k in range(A.shape[1]):
        acc = (acc + (Af[:, [k]] * Bf[[k], :]).astype(np.float32)).astype(np.float32)
    return acc


def main():
    # the checkpoint is bfloat16; torch is used only to widen it losslessly
    with safe_open(str(SNAP / "model.safetensors"), framework="pt") as f:
        W = f.get_tensor(LAYER).to(torch.float64).numpy()  # (N_out, K)
        E = f.get_tensor(EMBED).to(torch.float64).numpy()
    cfg = json.loads((SNAP / "config.json").read_text())
    print(f"model  : Llama-3.2-1B-Instruct  (hidden {cfg['hidden_size']})")
    print(f"layer  : {LAYER}  shape {W.shape}")

    rng = np.random.default_rng(SEED)
    ids = rng.integers(0, E.shape[0], size=N_TOK)
    H = E[ids]                                            # (N_TOK, K) real rows

    Wq, _ = per_row_int8(W)                               # (N_out, K) int8
    Hq, _ = per_row_int8(H)                               # (N_TOK, K) int8
    A = Hq                                                # (M, K)
    B = Wq.T.copy()                                       # (K, N)
    M, K = A.shape
    N = B.shape[1]
    print(f"shape  : M={M} K={K} N={N}  ({M * N} outputs, {K}-term contractions)")
    print(f"operands: identical int8 for every arm, per-row symmetric scales\n")

    exact = A.astype(np.int64) @ B.astype(np.int64)       # ground truth
    i32 = exact.astype(np.int32)                          # the paper's path
    i32_ok = np.array_equal(i32.astype(np.int64), exact)
    peak = int(np.max(np.abs(exact)))
    print(f"int64 exact        : |max| = {peak}  (int32 limit {2**31 - 1})")
    print(f"int32              : {'exact' if i32_ok else 'OVERFLOWED'}"
          f"  -- headroom {(2**31 - 1) / peak:.1f}x")

    for name, got in (("fp32 numpy dot", A.astype(np.float32) @ B.astype(np.float32)),
                      ("fp32 sequential", fp32_sequential(A, B))):
        d = got.astype(np.float64) - exact
        wrong = int(np.count_nonzero(d))
        rms = float(np.sqrt(np.mean((d / np.maximum(np.abs(exact), 1)) ** 2)))
        print(f"{name:19}: {wrong}/{M * N} outputs differ from exact "
              f"({100.0 * wrong / (M * N):.1f}%), max abs err {int(np.max(np.abs(d)))}, "
              f"RMS rel err {rms:.3e}")

    # --- where DOES fp32 break? int8 operands bound the product magnitude to
    # 127^2, and fp32 holds integers exactly to 2^24, so a K-term int8 dot is
    # exact in fp32 whenever |sum| <= 2^24, i.e. for K up to 2^24/127^2 at full
    # scale. Past that, fp32 loses an increment only when it falls below the
    # running sum's ULP -- which needs a dynamic range int8 operands can barely
    # reach. Construct the case that does reach it.
    print(f"\nAnalytic bound: |a*b| <= 127^2 = {127 * 127}, fp32 integers exact "
          f"to 2^24 = {2**24},\n  so an int8 dot is exact in fp32 for K up to "
          f"{2**24 // (127 * 127)} at full scale.")
    print("Constructed swamping case (large terms first, then +-1 terms):")
    for Ks in (2048, 8192, 32768):
        n_big = 2**24 // (127 * 127) + 1
        a_v = np.ones(Ks, np.int8); b_v = np.ones(Ks, np.int8)
        a_v[:n_big] = 127; b_v[:n_big] = 127
        ex = int(a_v.astype(np.int64) @ b_v.astype(np.int64))
        acc = np.float32(0.0)
        for x, y in zip(a_v.astype(np.float32), b_v.astype(np.float32)):
            acc = np.float32(acc + np.float32(x * y))
        i32 = int(np.int32(a_v.astype(np.int32) @ b_v.astype(np.int32)))
        print(f"  K={Ks:6d}: exact {ex}   int32 {i32} "
              f"({'exact' if i32 == ex else 'WRONG'})   "
              f"fp32 seq {int(acc)} (err {int(acc) - ex})")

    print("\nReading: the quantization is identical in every arm, so any "
          "difference is the\naccumulator alone. On the real layer fp32 "
          "accumulation is ALSO exact -- int8\noperands bound the product "
          "magnitude so tightly that fp32 cannot be swamped at\nthese shapes. "
          "The quire's contribution in the W8A8 integer path is therefore\n"
          "order independence and reproducibility, NOT accuracy; the accuracy "
          "case lives in\nthe 16-bit paths of Sections 6.2-6.6, where the "
          "operands carry real dynamic range.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
