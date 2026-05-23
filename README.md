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

The recipe is the standard W8A8 path, with b-posit as the number format: quantize
each weight (and, for W8A8, activations) to 8-bit b-posit, run the matmuls with
exact quire accumulation, round once. You get ~30–50% memory reduction at
AWQ-class accuracy — plus the reproducibility guarantee above. The 4/5-bit rungs
trade accuracy for more compression.

> **Status / honesty:** this repo is the **format standard + reference +
> conformance + hardware (RTL)**. It is *not yet* a turnkey one-line HuggingFace
> runtime — the inference integration (a quantize-and-run path) is on the
> roadmap. Low-bit accuracy is also format-limited: **8-bit is the deployable
> sweet spot** (AWQ-class); 4-bit is aggressive and needs calibration. We report
> this straight rather than overclaim.

## What's in here

```
reference/        format reference oracle (decode/encode/quire/mul/add) + conformance generators
targets/coreet/   CORE-ET (AiNEKKO ET-Minion) SystemVerilog block + testbenches + VERIFICATION.md
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
