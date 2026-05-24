#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""rung_sweep.py — reproduce the accuracy_table.md numbers from scratch.

Quantizes a real HuggingFace layer (or a synthetic transformer-shaped weight if
transformers isn't installed) to the bp4 / aip5 / bp8 rungs, with the same
reproducibility-safe per-channel power-of-two recipe, and prints relerr per rung.
Everything routes through the reference oracle's exact quire, so each result is
bit-identical on any GPU/CPU/RISC-V.

    python rung_sweep.py [--model <hf-id>] [--layers N]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))
import bposit_ref as bp                       # noqa: E402
from bposit_quantize import recommended_exponents  # noqa: E402

BITS = {"bp4": 4, "aip5": 5, "bp8": 8}


def _bp32_to_float(code: int) -> float:
    d = bp.decode_bposit32(code)
    return 0.0 if d.is_special else float(bp.decoded_to_fraction_32(d))


def matmul_rung(W, X, fmt):
    w_exp, x_exp = recommended_exponents(W), recommended_exponents(X)
    w_codes = [bp.quantize_vec(W[o] / (2.0 ** int(w_exp[o])), fmt) for o in range(W.shape[0])]
    x_codes = [bp.quantize_vec(X[t] / (2.0 ** int(x_exp[t])), fmt) for t in range(X.shape[0])]
    Y = np.empty((X.shape[0], W.shape[0]), dtype=np.float64)
    for t in range(X.shape[0]):
        for o in range(W.shape[0]):
            c = bp.dot_at_precision(x_codes[t], w_codes[o], fmt, "bp32")
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
        W = next(p for p in m.parameters() if p.ndim == 2)
        return f"{model_id} [first 2-D]", W.detach().numpy().astype(np.float64)
    except Exception as e:
        print(f"  (transformers/model unavailable — {type(e).__name__}: synthetic fallback)\n")
        rng = np.random.default_rng(0)
        return "synthetic transformer-shaped weight", rng.standard_normal((4864, 896)) * 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    args = ap.parse_args()

    label, Wfull = load_weight(args.model)
    W = np.ascontiguousarray(Wfull[:128, :512]).astype(np.float64)
    X = (np.random.default_rng(1).standard_normal((8, 512)) * 0.5).astype(np.float64)
    Yref = X @ W.T
    relerr = lambda Y: float(np.linalg.norm(Y - Yref) / np.linalg.norm(Yref))  # noqa: E731

    print(f"=== open-bposit rung sweep — {label}  W[128,512] ===\n")
    print(f"{'rung':6s} {'bits':>4s} {'vs fp16':>8s} {'relerr':>9s}")
    print("-" * 32)
    for fmt in ("bp8", "aip5", "bp4"):
        Y = matmul_rung(W, X, fmt)
        repro = "✓" if np.array_equal(Y, matmul_rung(W, X, fmt)) else "✗"
        print(f"{fmt:6s} {BITS[fmt]:4d} {100*(1-BITS[fmt]/16):7.0f}% {relerr(Y):9.4f}   reproducible:{repro}")
    print("\nbp8 is the deployable W8A8 rung (~9%); aip5/bp4 are coarse — mixed/dynamic precision only.")


if __name__ == "__main__":
    main()
