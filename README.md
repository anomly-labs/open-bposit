# open-bposit

**An open, bit-reproducible low-precision number format for AI inference.**
Run models with **~30–50% less memory** *and* results that are **identical to the
bit on any GPU, CPU, or RISC-V core** — something INT8/AWQ/GPTQ quantization
cannot give you.

b-posit (bounded posit) / AI-Posit is a pure-integer, float-free number format.
This repo is the **open standard**: the reference implementation, an exhaustive
conformance suite, and a verified hardware block (SystemVerilog for AiNEKKO's
CORE-ET, with more targets to come).

## Why it matters

Quantizing a model to fewer bits saves memory — that part isn't new. What *is*
new here is **reproducibility**:

- **Less memory.** Weights at **8-bit b-posit** are ~50% smaller than fp16/bf16;
  end-to-end you typically land in the **30–50%** range depending on what you
  quantize. The 4- and 5-bit rungs go further when you can tolerate/calibrate
  the accuracy.
- **Bit-for-bit reproducible — the differentiator.** Every result is *identical*
  across an NVIDIA GPU, x86, and RISC-V. Ordinary float quantization isn't: IEEE
  rounding + non-associative tensor-core reductions make the low bits drift
  across chips, drivers, and even runs. b-posit accumulates in an exact 256-bit
  integer **quire** with no intermediate rounding and one final round — so the
  answer depends only on the inputs, not the hardware. That's what regulated,
  auditable, and forensic AI need, and what makes a number format a *trustable
  standard* (the conformance suite here defines "correct").
- **No FPU required.** The compute is integer-only — it runs on small,
  FPU-less cores (smaller die, lower power), which is the edge-AI silicon angle.

## Applying it to a HuggingFace model

The recipe is the standard W8A8 path with b-posit as the number format: quantize
each weight (and, for W8A8, activations) to b-posit, run the matmuls with exact
quire accumulation, round once. `examples/hf_bposit_demo.py` does this on a real
HuggingFace model and reports memory, accuracy, and reproducibility.

> **Status / honesty (read this).** What b-posit gives is **reproducibility +
> memory + FPU-free hardware** — *not* best-in-class low-bit accuracy. Measured
> across 7 real Qwen-0.5B layers (`examples/accuracy_table.md`): naive **8-bit
> b-posit ≈ 12.5% relative error**, which a **reproducibility-safe per-channel
> power-of-two scale** (`reference/bposit_quantize.py`) brings to **≈ 9%** — and
> an OpenEvolve search couldn't beat ~9%, so that's the intrinsic 8-bit floor.
> Note: per-channel *max* scaling (the INT8 trick) barely helps posits — they're
> tapered, not uniform; the win is **RMS-centering** each channel into the dense
> high-precision band, via an exact power-of-two shift that keeps the result
> bit-reproducible. So if you need near-lossless low-bit accuracy, INT8/AWQ wins.
> **b-posit's niche is when you need results that are bit-identical across
> hardware** (audit, regulated/forensic, multi-vendor, pre-silicon validation) —
> which INT8/AWQ cannot give. For accuracy-sensitive work, **16-bit b-posit is
> bf16-class AND reproducible**. This repo is the format standard + reference +
> conformance + hardware (RTL); a turnkey HF runtime is roadmap.

## What's in here

```
reference/        format reference oracle (decode/encode/quire/mul/add) + conformance generators
                  + bposit_quantize.py (reproducibility-safe W8A8 recipe)
targets/coreet/   CORE-ET (AiNEKKO ET-Minion) SystemVerilog block + testbenches + VERIFICATION.md
targets/coreet/synth/   sky130 synthesis flow (yosys area + OpenLane P&R/STA) for the cloud labs
examples/         HuggingFace W8A8 demo + measured accuracy table
hdl/              (planned) SpinalHDL source that generates the SystemVerilog
```

## Verify it (open tools: iverilog + python3)

```bash
cd targets/coreet && make verify-full
```

Regenerates conformance vectors from the reference and checks the hardware
bit-for-bit: the `quire→bposit32` encoder over **132,811** vectors (every scale,
saturation edge, and field-width transition), **all 65,536** bp8 product pairs,
all **2,560** bp4/aip5 ALU pairs, plus GEMM-cell edge cases. Lint-clean. The
reference is itself bit-identical to the same compute on GPU/x86/RISC-V, so a
pass here means the hardware agrees with all of them.

See `targets/coreet/VERIFICATION.md` for the full coverage matrix and the honest
gap list (no synthesis/STA yet; not yet wired into the core pipeline).

## License

Apache-2.0. Copyright (c) 2026 Anomly, Inc.

## Authors

Created by Ry Bruscoe at Anomly, Inc. Contributions welcome.
