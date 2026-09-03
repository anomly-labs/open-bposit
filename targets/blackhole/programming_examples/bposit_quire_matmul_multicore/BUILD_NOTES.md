<!-- Copyright (c) 2026 Anomly, Inc. SPDX-License-Identifier: Apache-2.0 -->
# bposit_quire_matmul_multicore — build & run notes

Exact 256-bit Kulisch quire **b-posit16 MATMUL across MANY Blackhole baby
(RISC-V) cores**, bit-exact vs the oracle golden. This is the **robust
throughput path**: it reuses the *already-bit-exact* scalar quire-dot kernel
verbatim and only adds the multi-core dispatch. The hard SFPU-vectorization path
is paused; this one is tractable because every core runs proven scalar code.

## What it computes

```
C[M,N] = A[M,K] · B[K,N]      M = N = 8,  K = 16   →  64 output elements
```

Each output `C[i,j]` is an **independent** exact-quire dot of length `K`:

```
C[i,j] = Σ_k bp16_prod_to_q256(A[i,k], B[k,j])      (exact, no per-product rounding)
         → bp16_encode_quire256(quire)               (the single truncating readout)
```

No cross-core reduction is needed (outputs are independent), so the 64 outputs are
distributed **one-per-core onto a CoreRange grid** (an 8×8 baby-core sub-grid; the
host uses `split_work_to_cores` so it also works on any grid the device exposes).

## Why this is correct by construction

The **per-element inner math is byte-for-byte identical** to the single-core,
on-silicon-proven kernel (`bposit_quire_reduce/kernels/quire_matmul_baby.cpp`,
mode `matmul`, 24/24 bit-exact). The only new code is the host-side dispatch and
the per-core output marshalling. Because each output is an isolated dot, the
parallel result provably equals the scalar result.

## Files

| File | Role |
|------|------|
| `bposit_quire_matmul_multicore.cpp` | Host: self-check, CoreRange dispatch, read-back, bit-exact gate, `PASS/FAIL` print |
| `kernels/quire_matmul_mc_baby.cpp` | The per-core kernel — SAME scalar exact-quire dot, parameterised by a `[out_start,out_end)` output slice |
| `gen_matmul_mc_golden.py` | Regenerates the golden header from the canonical oracle (`the scalar-kernel/.../bposit16_reference.py`) |
| `quire_matmul_mc_golden_cases.h` | Baked golden (operand matrices, 64 readouts, gate quire of output (7,7), per-element `bposit16_mul` cross-check codes) — **generated, do not hand-edit** |
| `CMakeLists.txt` | Target `metal_example_bposit_quire_matmul_multicore` |
| `run_on_silicon.sh` | Regen golden → build → run on physical Blackhole, gate on golden |

## Reused, unchanged (absolute `#include`)

```
<kernel>/bp16_quire.h    # bp16_prod_to_q256, q256_add
<kernel>/bp16_encode.h   # bp16_encode_quire256 (QFRAC=96, truncation)
```

## Marshalling rules (do NOT regress)

- **Inputs A, B** are read-only and shared. Each is **ONE page** (`page_size ==
  whole matrix`); every core does **one bulk `noc_async_read`** of A and one of B
  into its own L1 scratch — the proven single-page/bulk pattern. (A = 128 codes,
  B = 128 codes → ~512 B each, trivially L1-resident on every core.)
- **Output C** is a **1-word-per-page** buffer (`page_size == 4 B`, 64 pages). Each
  core writes **only its own output slot(s)** via `get_noc_addr(o)`, so concurrent
  cores never collide and never need a shared-page read-modify-write.
- **Gate quire** (8 limbs of output (7,7)) goes to a separate one-page buffer,
  written by the single core that owns that output index.

## Build

```bash
cd $TT_METAL_HOME
# golden is checked in; regenerate from the oracle if inputs change:
( cd tt_metal/programming_examples/bposit_quire_matmul_multicore && \
  python3 gen_matmul_mc_golden.py > quire_matmul_mc_golden_cases.h )

# configure once (if build_Release doesn't exist), then build the single target:
cmake -S . -B build_Release -DBUILD_PROGRAMMING_EXAMPLES=ON   # only if needed
ninja -C build_Release metal_example_bposit_quire_matmul_multicore
```

## Run (physical Blackhole only — no simulator)

```bash
./tt_metal/programming_examples/bposit_quire_matmul_multicore/run_on_silicon.sh
# Expect (on bit-exact match):
#   PASS: exact bp16 quire MATMUL (8x16x8) across <C> cores on Blackhole, 72/72 match golden
# (72 = 64 readouts + 8 gate quire limbs)
```

## Verification status

- **Host self-check: PASS (native gcc, against the scalar reference headers (kernel/)).** All 64
  C readouts, the gate (7,7) quire, every per-element exact product vs
  `bposit16_mul`, and K-order associativity (forward == reverse) all match the
  generated golden. This same gate runs in the host program *before* the device
  dispatch.
- **On-silicon: NOT YET RUN** (a separate process drives the hardware). Build is a
  draft authored to be buildable; do not build/run here.

## Main risk for the on-silicon run

The **1-word-per-page output buffer + 64 concurrent cores each writing one page**
is the only pattern that differs from the proven single-core path. Risks:
1. **Page granularity / NoC alignment** — `page_size == 4 B` is below the typical
   DRAM page/alignment the single-core path used (whole-buffer pages). If the NoC
   write alignment or `InterleavedAddrGen` page stride misbehaves at 4-byte pages,
   fall back to: each core writes to a per-core *region* of a coarser-paged output
   (e.g. one page per core), or have core (0,0) gather. The readouts are correct
   in L1 regardless; only the DRAM write granularity is at issue.
2. **Concurrent-write ordering** — distinct output pages mean no true data race,
   but confirm the read-back happens after `EnqueueMeshWorkload` completes (it is
   gated by the blocking `EnqueueReadMeshBuffer`).

If page-4B writes are rejected, the lowest-risk fix is a coarser output paging
(one page per core) with the host reshaping on read-back — the kernel's compute
loop and the golden stay identical.
