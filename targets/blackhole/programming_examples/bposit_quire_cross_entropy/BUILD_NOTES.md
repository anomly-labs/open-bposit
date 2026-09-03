<!-- Copyright (c) 2026 Anomly, Inc. SPDX-License-Identifier: Apache-2.0 -->
# bposit_quire_cross_entropy — cross-entropy (classification loss) on a Blackhole baby core

A REAL information-theoretic ML primitive computed in b-posit16 through
Anomly's EXACT 256-bit Kulisch quire on a single Tenstorrent Blackhole baby
(RISC-V) data-movement core, bit-exact vs the published oracle golden. The
load-bearing exact-quire loss SUM runs on-device; the per-element `log2(q_i)` is
host-precomputed (an honest split, forced by the baby-core DATA-region size limit
— see "The log story" below).

## The ML primitive

**cross-entropy** — the standard classification training loss:

```
H(p, q) = -Sum_i  p_i * log2(q_i)        for a target p and a predicted q over N classes
```

For a target distribution `p` (one-hot or soft label) and a model/predicted
distribution `q` (softmax output) over the same alphabet of `N` classes,
cross-entropy is the per-sample loss every softmax classifier minimizes. The
siblings: `KL(p||q) = H(p,q) - H(p)` and Shannon `H(p) = H(p,p)`. This is genuine
ML breadth for the catalog — the actual loss, not a synthetic tile.

### Why it maps onto the proven exact-quire kernel

Cross-entropy is a sum-of-products `Sum_i p_i * log2(q_i)`, exactly the shape the
exact quire was built for. Per element:

```
log2q_i = bposit16_log2(q_i)          # log2(q_i)  [HOST-precomputed — see split]
term    = bposit16_mul(p_i, log2q_i)  # EXACT bp16 product = p_i*log2(q_i)  [DEVICE]
H_q    += bp16_to_q256(term)          # EXACT 256-bit accumulation (QFRAC=96) [DEVICE]
H_q     = q256_negate(H_q)            # H(p,q) = -Sum (positive loss)         [DEVICE]
H_bp32  = bp32_encode_quire256(H_q)   # the SINGLE rounding: the readout      [DEVICE]
```

- `bp16_to_q256` places each `term` into the 256-bit quire with **no per-product
  rounding** (`kernel/bp16_quire.h`).
- `q256_add` is exact 256-bit two's-complement ripple-carry, so the
  N-accumulation is **order-independent** (forward == reverse).
- The **single** rounding is the final quire->bposit32 readout
  (`bp32_encode_quire256`, `kernel/bp32_encode.h`).

**Why this matters for a loss (not just a correctness tile):** the
tiny-probability tails — `q_i` small, so `log2(q_i)` is large-negative — are the
terms cross-entropy actually cares about (a confident-wrong prediction). A float
dot **silently drops** those tail terms once the partial sum dominates (the
classic catastrophic-cancellation / absorption failure); the exact quire sums
them **losslessly**, so the loss is rounding-stable and reproducible. This
exact-quire SUM is the load-bearing result and it runs **on-device**.

## The log story — HONEST host/device split (and why)

The CRITICAL question was whether the base-2 log can run on the baby core. The
only freestanding log2 available is a **65536-entry LUT** (`kernel/bp16_log2_lut.h`,
`log2_lut[q] = bposit16_log2(q)`) — there is no small closed-form `bp16_log2`.

A first cut compiled the LUT INTO the kernel; it **built but FAILED at kernel load
on real silicon**:

```
TT_THROW: .../cross_entropy_baby/.../brisc/brisc.elf: segment[1] [0xffb00d30,+0x20000)
overflows region:1 limit of 0x11d0 bytes, reduce the size of thread_local variables
```

Root cause: the LUT (~128 KB as `u16`) lands in the kernel TU's static-data
segment, but the BRISC baby core's **local DATA region is only `0x11d0` (~4.5 KB)**.
Big-L1 (1.5 MB) is a **separate CB/buffer address space**, not the kernel's
`.data`/`.rodata` — so a large static LUT in the kernel binary can never fit.

**Fix (this version):** move the per-element `log2(q_i)` to the **HOST**
(`bposit16_log2` == `BP16_LOG2_LUT[q_i]`) and hand the device a small `log2q[]`
operand alongside `p[]`. The device kernel is then **pure exact-quire**:
`term = bp16_mul(p_i, log2q_i)` -> 256-bit-quire accumulate -> negate -> bposit32
readout. This is an HONEST split, clearly documented: the **elementwise log2 is
host-precomputed**, while the **load-bearing exact-quire SUM** (the rounding-stable,
tail-lossless part that the whole result hinges on) runs **100% on-device**. The
host log2 is itself the canonical oracle, so the answer is bit-exact vs golden.

Verified by cross-compiling the (now LUT-free) kernel compute body freestanding for
`rv32im` (`riscv64-linux-gnu-gcc -march=rv32im -mabi=ilp32 -nostdlib
-ffreestanding`): **zero libgcc helper calls** (`__ashldi3`, `__muldi3`, ...),
**zero undefined external symbols**, and — the point — **`.data = 0`, `.bss = 0`,
no `.rodata`**, so the kernel binary cannot overflow the baby-core DATA region.

## Device vs host split (honest accounting)

- **DEVICE** (Blackhole baby core, `kernels/cross_entropy_baby.cpp`): the EXACT
  loss accumulation — every EXACT `bp16` product `term = p_i*log2(q_i)`, the exact
  256-bit quire accumulation, the negate, AND the quire->bposit32 readout. The
  **load-bearing, rounding-stable, tail-lossless quire SUM is 100% on-device.**
- **HOST** (`bposit_quire_cross_entropy.cpp`): precomputes the per-element
  `log2(q_i)` (`bposit16_log2` == `BP16_LOG2_LUT[q_i]`, host-side only because the
  LUT does not fit the baby core), marshals `p[]` + `log2q[]` (DRAM<->L1 MeshBuffer,
  one page per buffer), runs a belt-and-suspenders self-check (host `log2(q_i)` ==
  baked `QGOLD_CE_LOG2Q`, plus the full quire + bp32 re-derived from the reused
  EXACT headers) before the device run, then gates the device output (8
  negated-quire limbs + the bp32 readout) byte-/code-identical vs golden.

## What is reused (not reimplemented)

The kernel and host `#include` the **same freestanding RV32IM-clean headers** the
proven `cross_entropy_kernel.c` / reduce / dot / matmul / holographic / causet
kernels use (proven byte-identical x86 == qemu-rv32 == ttsim-BRISC), and the
device kernel mirrors the proven RV32 reference `kernel/cross_entropy_kernel.c` on
the exact-quire accumulation (with log2 host-precomputed instead of LUT'd):

- `the scalar-kernel/deploy/tt-metalium-bposit/kernel/bp16_mul.h` — `bposit16_mul` (the `p*log2(q)` product) **[DEVICE + HOST]**
- `the scalar-kernel/deploy/tt-metalium-bposit/kernel/bp16_quire.h` — `bp16_to_q256`, `q256_add`, `q256_negate` **[DEVICE + HOST]**
- `the scalar-kernel/deploy/tt-metalium-bposit/kernel/bp32_encode.h` — `bp32_encode_quire256` (the readout) **[DEVICE + HOST]**
- `the scalar-kernel/deploy/tt-metalium-bposit/kernel/bp16_log2_lut.h` — `BP16_LOG2_LUT[65536]` (the log2) **[HOST ONLY — NOT compiled into the kernel]**
- `the scalar-kernel/deploy/tt-metalium-bposit/kernel/bp16_decode.h` — pulled in transitively

The host / marshalling / gate idiom mirrors the sibling
`programming_examples/bposit_quire_causet/bposit_quire_causet.cpp`: 1x1 unit mesh
on device 0, one page per buffer, one bulk NoC read per operand, one bulk NoC
write out, an input-readback diagnostic, and a byte-level gate of the full negated
quire + the bposit32 readout.

## Golden + oracle citations

- **golden**: `the scalar-kernel/deploy/tt-metalium-bposit/golden/cross_entropy.json`
  (`p_codes` N, `q_codes` N, `quire_le_hex` 32-byte LE negated quire, `bp32_code`, `n`).
  Canonical case: `p = 1/2, 1/4, 1/8, 1/8`; `q = 1/8, 1/8, 1/4, 1/2`; **H = 2.625 bits**.
- **oracle**: `mosyne-bposit/kernels/bposit16_reference.py`
  (`bposit16_log2`, `bposit16_mul`, `bposit16_to_quire`, `quire256_to_bposit32`,
  `decode_bposit32`, `decoded_to_fraction_32`, `QUIRE_FRAC_BITS=96`).
- **C reference**: `the scalar-kernel/deploy/tt-metalium-bposit/kernel/cross_entropy_kernel.c`
  (the proven RV32IM-clean kernel this device kernel mirrors on the exact-quire
  accumulation — `term = bposit16_mul(p, log2q) -> quire -> negate` — with log2
  host-precomputed instead of LUT'd in-kernel, plus metalium marshalling).
- **canonical producer**: `the scalar-kernel/deploy/tt-metalium-bposit/scripts/gen_golden_vectors.py`
  (`build_cross_entropy_golden` / `_cross_entropy_quire`).

## Files

| file | role |
|---|---|
| `bposit_quire_cross_entropy.cpp` | host driver: host log2 precompute + marshalling (p, log2q) + self-check + device dispatch + golden gate |
| `kernels/cross_entropy_baby.cpp` | baby-core kernel: pure exact-quire accumulation of `p_i*log2(q_i)` -> negated quire + bposit32 (LUT-free, on-device) |
| `gen_cross_entropy_golden.py` | re-derive the baked golden header from `cross_entropy.json` + oracle |
| `cross_entropy_golden_cases.h` | **generated** baked golden (p, q, host-precomputed log2q, per-element terms, expected negated quire, bp32 readout) |
| `CMakeLists.txt` | target `metal_example_bposit_quire_cross_entropy` |
| `run_on_silicon.sh` | regen golden if stale, build, run on physical Blackhole, gate on golden |

## Regenerate the golden

```bash
cd tt_metal/programming_examples/bposit_quire_cross_entropy
python3 gen_cross_entropy_golden.py > cross_entropy_golden_cases.h
```
The generator asserts, at generation time, that its oracle output equals the
published `golden/cross_entropy.json` `quire_le_hex` AND `bp32_code`. The host then
re-verifies the baked header against the reused C headers at startup before any
device run.

## Build

```bash
# in-tree, registered in programming_examples/CMakeLists.txt
cmake -S $TT_METAL_HOME \
      -B $TT_METAL_HOME/build_Release \
      -DBUILD_PROGRAMMING_EXAMPLES=ON          # only if build tree absent
ninja -C $TT_METAL_HOME/build_Release \
      metal_example_bposit_quire_cross_entropy
```

## Run on real silicon

```bash
./run_on_silicon.sh cross_entropy     # or "all"
```
Requires `ARCH_NAME=blackhole`, `TT_METAL_SLOW_DISPATCH_MODE=1`, **no**
`TT_METAL_SIMULATOR`, and a healthy Blackhole (tt-kmd + firmware + tt-smi). The
A power-cycled card needs a cold boot + tt-flash first — confirm healthy before
running. **This file is authoring + buildable draft ONLY; the build+run on
hardware is done by a separate process.**

Expected on PASS:
```
PASS: cross-entropy on a Blackhole baby core, bit-exact vs golden (real silicon)
      H(p,q) = -Sum_i p_i*log2(q_i) over N=4 classes ... H(p,q) = 2.625 bits (bp32 0x45400000), 9/9 match golden (case cross_entropy)
      ... SPLIT: log2(q_i) HOST-precomputed; the exact-quire SUM runs 100% on-device.
```
golden negated quire: `0000000000000000000000a00200000000000000000000000000000000000000`;
golden bp32: `0x45400000` (H = 2.625); host-precomputed `log2q = 0xba00 0xba00 0xbc00 0xc000`.

## Verification status (host-side, pre-silicon)

Verified locally with gcc/g++ reusing the EXACT the scalar-kernel C headers the kernel
`#include`s (`bposit16_mul` + `bp16_to_q256` + `q256_add` + `q256_negate` +
`bp32_encode_quire256`) plus the host-side `BP16_LOG2_LUT`, via a standalone
host-check that mirrors the host driver (host-precompute `log2q`, then the
device's LUT-free exact-quire path) and gates it the same way:

- the host-precomputed `log2(q_i)` == baked `QGOLD_CE_LOG2Q`
  (`0xba00, 0xba00, 0xbc00, 0xc000` = `bposit16_log2` of `q = 1/8,1/8,1/4,1/2`);
- the per-element term codes `term_i = bposit16_mul(p_i, log2q_i)` ==
  `QGOLD_CE_TERM` (`0xbe00, 0xc200, 0xc800, 0xcc00`);
- the negated 256-bit quire == `golden/cross_entropy.json` `quire_le_hex`
  (`0000000000000000000000a00200…`, bit-exact);
- the bposit32 readout == `golden/cross_entropy.json` `bp32_code`
  (`0x45400000`, H = 2.625).

Result: **9/9** (8 quire limbs + 1 bp32 readout) match golden, host-side. The
(now LUT-free) device kernel compute body additionally cross-compiles
**RV32IM-clean** (no libgcc helpers, no undefined symbols) with **`.data = 0`,
`.bss = 0`, no `.rodata`** — directly confirming the kernel-load overflow that
killed the first cut (LUT in `.rodata`) is gone. NOT yet run on the device — that
is the separate build+run process's job.

## Top risk

The kernel `#include`s the the scalar-kernel **math** headers (`bp16_mul.h`, `bp16_quire.h`,
`bp32_encode.h` — all tiny, no large data) by **absolute path**. If the tt-metal
kernel-compile sandbox (RISC-V cross-compile) rejects the absolute `#include` — the
same `TODO(include-path)` caveat flagged in the proven `causet/mera/holographic`
baby kernels — the fix is to pass the header dir via the `CreateKernel`
defines/`-I` mechanism or have `run_on_silicon.sh` copy the three headers into
`kernels/`. The numerics are proven bit-exact on host and the compute is
RV32IM-clean with zero static data, so the prior kernel-load overflow is resolved;
this remaining risk is purely include-path plumbing, identical to the already-
shipping exact-quire examples. Secondary: for very large alphabets N the
host-precomputed `log2q[]` operand grows with N (same as `p[]`/`q[]`) and is
marshalled the same one-page-per-buffer way; the device footprint stays tiny (the
exact quire is fixed 32 bytes), so it scales cleanly.
```
