#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mixed_precision_demo.py — mix rungs in ONE exact quire, stay bit-reproducible.

The point of this demo is the property float can't offer: you can quantize some
weight channels to 8-bit b-posit and others to 5-bit AI-posit, accumulate the
mixed-rung dot product in a single exact 256-bit quire, and the result is still
**bit-identical on any GPU/CPU/RISC-V**. Mixed-precision *and* reproducible.

Strategy here (one simple, honest policy): put the highest-energy output channels
in bp8 and the rest in aip5. That interpolates accuracy between the uniform rungs
at a lower average bit-width than uniform bp8. Accuracy is reported as-measured —
the headline is the reproducibility, not a free accuracy lunch.

    python mixed_precision_demo.py [--model <hf-id>] [--bp8-frac 0.25]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))
import bposit_ref as bp                       # noqa: E402
from bposit_quantize import recommended_exponents  # noqa: E402

QF = bp.QUIRE_FRAC_BITS
BITS = {"aip5": 5, "bp8": 8}


def _bp32_to_float(code: int) -> float:
    d = bp.decode_bposit32(code)
    return 0.0 if d.is_special else float(bp.decoded_to_fraction_32(d))


def mixed_dot(x_codes, w_codes, x_fmt, w_fmt) -> int:
    """Exact dot of differently-typed operands into one quire -> bposit32 code."""
    q = 0
    for cx, cw in zip(x_codes, w_codes):
        q += int(bp.code_value(cx, x_fmt) * bp.code_value(cw, w_fmt) * (1 << QF))
    return bp.quire256_to_bposit32(q)


def matmul(W, X, w_fmt_per_row):
    """X @ W^T; activations in bp8, each weight row in its assigned rung."""
    w_exp, x_exp = recommended_exponents(W), recommended_exponents(X)
    x_codes = [bp.quantize_vec(X[t] / (2.0 ** int(x_exp[t])), "bp8") for t in range(X.shape[0])]
    w_codes = [bp.quantize_vec(W[o] / (2.0 ** int(w_exp[o])), w_fmt_per_row[o])
               for o in range(W.shape[0])]
    Y = np.empty((X.shape[0], W.shape[0]), dtype=np.float64)
    for t in range(X.shape[0]):
        for o in range(W.shape[0]):
            c = mixed_dot(x_codes[t], w_codes[o], "bp8", w_fmt_per_row[o])
            Y[t, o] = _bp32_to_float(c) * (2.0 ** int(x_exp[t] + w_exp[o]))
    return Y


def load_weight(model_id):
    try:
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        for name, p in m.named_parameters():
            if p.ndim == 2 and "mlp" in name and "proj" in name:
                return f"{model_id} [{name}]", p.detach().numpy().astype(np.float64)
    except Exception as e:
        print(f"  (transformers/model unavailable — {type(e).__name__}: synthetic)\n")
    rng = np.random.default_rng(0)
    return "synthetic", rng.standard_normal((4864, 896)) * 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--bp8-frac", type=float, default=0.25)
    args = ap.parse_args()

    label, Wfull = load_weight(args.model)
    W = np.ascontiguousarray(Wfull[:128, :512]).astype(np.float64)
    X = (np.random.default_rng(1).standard_normal((8, 512)) * 0.5).astype(np.float64)
    Yref = X @ W.T
    relerr = lambda Y: float(np.linalg.norm(Y - Yref) / np.linalg.norm(Yref))  # noqa: E731

    n = W.shape[0]
    # assign bp8 to the highest-output-energy channels, aip5 to the rest
    energy = np.linalg.norm(Yref, axis=0)            # per output channel
    k = max(1, int(round(args.bp8_frac * n)))
    bp8_rows = set(np.argsort(energy)[::-1][:k].tolist())
    fmt = ["bp8" if o in bp8_rows else "aip5" for o in range(n)]
    avg_bits = (k * 8 + (n - k) * 5) / n

    print(f"=== open-bposit mixed-precision demo — {label}  W[{n},512] ===\n")
    Y_aip5 = matmul(W, X, ["aip5"] * n)
    Y_bp8 = matmul(W, X, ["bp8"] * n)
    Y_mix = matmul(W, X, fmt)
    print(f"{'policy':24s} {'avg bits':>8s} {'relerr':>9s}")
    print("-" * 45)
    print(f"{'uniform aip5':24s} {5.0:8.2f} {relerr(Y_aip5):9.4f}")
    print(f"{f'mixed ({k}/{n} bp8)':24s} {avg_bits:8.2f} {relerr(Y_mix):9.4f}")
    print(f"{'uniform bp8':24s} {8.0:8.2f} {relerr(Y_bp8):9.4f}")
    print()
    repro = np.array_equal(Y_mix, matmul(W, X, fmt))
    print(f"mixed-rung result bit-reproducible across runs: {'YES' if repro else 'NO'}")
    print("All three accumulate in the SAME exact 256-bit quire — mixing rungs does")
    print("not break reproducibility, which is the property float/INT8 can't give.")


if __name__ == "__main__":
    main()
