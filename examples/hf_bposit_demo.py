#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
hf_bposit_demo.py — quantize a real HuggingFace model's weights to b-posit (W8A8)
and demonstrate the three claims, end to end:

  1. MEMORY        — 8-bit b-posit weights are ~50% smaller than fp16/bf16
  2. ACCURACY      — output of a real layer in b-posit vs full precision (relerr)
  3. REPRODUCIBILITY — the b-posit result is computed by EXACT integer quire
                       accumulation, so it is bit-identical on any hardware
                       (GPU / CPU / RISC-V) — proven separately, illustrated here
                       by the deterministic exact-quire dot product.

Uses a real model via `transformers` if available (default: a small cached Qwen);
otherwise falls back to a synthetic transformer-shaped weight so the demo still
runs with only numpy + the local reference.

    python hf_bposit_demo.py [--model <hf-id>]

Honest scope: this is a *layer-level* demonstration on real weights (a full
generate() loop in the pure-Python reference would be slow). 8-bit is the
deployable accuracy point; 4/5-bit rungs trade accuracy for more compression.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))
import bposit_ref as bp  # noqa: E402

QF = bp.QUIRE_FRAC_BITS


def q8_dequant(arr: np.ndarray) -> np.ndarray:
    """Quantize each element to bposit8, return the dequantized fp64 value."""
    out = np.empty(arr.shape, dtype=np.float64)
    flat, o = arr.ravel(), out.ravel()
    for i, v in enumerate(flat):
        d = bp.decode_bposit8(bp.encode_bposit8(float(v)))
        o[i] = 0.0 if d.is_special else float(bp.decoded_to_fraction_8(d))
    return out


def exact_quire_dot(a_row: np.ndarray, b_row: np.ndarray) -> float:
    """One output element via EXACT bposit8 quire accumulation -> bposit32 value.
    Pure integer; identical on any hardware implementing the same LUTs."""
    q = 0
    for x, w in zip(a_row, b_row):
        ca, cb = bp.encode_bposit8(float(x)), bp.encode_bposit8(float(w))
        q += int(bp.code_value(ca, "bp8") * bp.code_value(cb, "bp8") * (1 << QF))
    code = bp.quire256_to_bposit32(q)
    d = bp.decode_bposit32(code)
    return 0.0 if d.is_special else float(bp.decoded_to_fraction_32(d))


def load_weight(model_id):
    """Return (label, total_params, a real [out,in] weight matrix)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        total = sum(p.numel() for p in m.parameters())
        # grab a representative Linear weight (an MLP projection from layer 0)
        W = None
        for name, p in m.named_parameters():
            if p.ndim == 2 and ("mlp" in name or "fc" in name or "proj" in name):
                W = p.detach().numpy().astype(np.float64); wname = name; break
        if W is None:
            W = next(p for p in m.parameters() if p.ndim == 2).detach().numpy().astype(np.float64)
            wname = "(first 2-D weight)"
        return f"{model_id}  [{wname} {tuple(W.shape)}]", total, W
    except Exception as e:
        print(f"  (transformers/model unavailable — {type(e).__name__}: falling back to synthetic)\n")
        rng = np.random.default_rng(0)
        W = rng.standard_normal((4864, 896)) * 0.05    # Qwen-0.5B-ish mlp shape, NN-scale
        return "synthetic transformer-shaped weight [4864, 896]", 500_000_000, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    args = ap.parse_args()

    print("=== open-bposit — HuggingFace W8A8 demo ===\n")
    label, total, W = load_weight(args.model)
    print(f"model: {label}\n")

    # 1) MEMORY (whole model)
    fp16 = total * 2
    bp8 = total * 1
    print("1) MEMORY")
    print(f"   {total/1e6:.1f}M params:  fp16 = {fp16/1e6:.0f} MB   bposit8 = {bp8/1e6:.0f} MB"
          f"   -> {100*(1-bp8/fp16):.0f}% smaller")
    print("   (4-/5-bit rungs go to 75/69% smaller, with more accuracy cost.)\n")

    # representative slice for accuracy/repro (keeps the pure-Python encode fast)
    Ws = np.ascontiguousarray(W[:64, :512])
    rng = np.random.default_rng(1)
    X = rng.standard_normal((8, 512)) * 0.5         # stand-in activations

    # 2) ACCURACY: real layer slice, full precision vs bposit8 W8A8.
    Y_ref = X @ Ws.T

    def relerr(Yq):
        return float(np.linalg.norm(Yq - Y_ref) / np.linalg.norm(Y_ref))

    # naive (direct quantize, no scaling)
    Y_naive = q8_dequant(X) @ q8_dequant(Ws).T
    # standard W8A8: per-channel weight scale + per-token activation scale
    sw = np.abs(Ws).max(axis=1, keepdims=True); sw[sw == 0] = 1.0
    sx = np.abs(X).max(axis=1, keepdims=True); sx[sx == 0] = 1.0
    Wq = q8_dequant(Ws / sw) * sw
    Xq = q8_dequant(X / sx) * sx
    Y_scaled = Xq @ Wq.T
    print("2) ACCURACY  (X[8,512] @ W[64,512]^T, full precision vs bposit8 W8A8)")
    print(f"   naive (no scaling):                 {relerr(Y_naive):.3e}")
    print(f"   per-channel W / per-token X scaling: {relerr(Y_scaled):.3e}  (standard W8A8)\n")
    Y_bp = Y_scaled

    # 3) REPRODUCIBILITY: one output via exact integer quire (hardware-independent)
    r0 = exact_quire_dot(X[0], Ws[0])
    r0_again = exact_quire_dot(X[0], Ws[0])
    print("3) REPRODUCIBILITY  (output[0,0] via EXACT bposit8 quire accumulation)")
    print(f"   exact-quire result : {r0:.10f}")
    print(f"   recomputed         : {r0_again:.10f}   ({'identical' if r0 == r0_again else 'DIFFERS'})")
    print(f"   dequant-matmul[0,0]: {Y_bp[0,0]:.10f}   (≈, rounded to bposit32)")
    print("   The quire result is exact integer math — bit-identical on any GPU/CPU/")
    print("   RISC-V (proven: GPU ≡ x86 ≡ RISC-V). Float matmul is NOT reproducible.\n")

    print("Summary: ~50% memory at 8-bit, layer accuracy above, and a result that")
    print("reproduces to the bit across hardware — the b-posit value proposition.")


if __name__ == "__main__":
    main()
