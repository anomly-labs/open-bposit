<!-- Copyright (c) 2026 Anomly, Inc. SPDX-License-Identifier: Apache-2.0 -->
# bposit_quire_matmul_multicore_perf — build & run notes

**Throughput** variant of the proven `bposit_quire_matmul_multicore`. Scales the
*already-bit-exact* scalar exact-quire dot to a LARGE matmul that saturates the
full ~140-core Blackhole baby (RISC-V) grid, and **times the device matmul** to
report the **exact-quire throughput** — Anomly's differentiator number.

The small `bposit_quire_matmul_multicore` example (C[8,8], 64 outputs one-per-core,
72/72 bit-exact on real silicon) is kept as the **correctness gate** and is left
untouched. This is an additive, separate example.

## What it computes

```
C[N,N] = A[N,K] · B[K,N]      N = 128,  K = 64   →  16384 output elements
```

- **16384** exact-quire dots (one per output `C[i,j]`), each of length **K = 64**
- **16384 × 64 = 1,048,576** EXACT bp16×bp16 products fed UNROUNDED into 256-bit
  Kulisch quires (`QUIRE_FRAC_BITS = 96`); only the per-output quire→bp16 readout
  rounds.

Each `C[i,j] = Σ_k bp16_prod_to_q256(A[i,k], B[k,j]) → bp16_encode_quire256(quire)`.
No cross-core reduction (outputs independent), so the 16384 outputs are split
across the device grid via `split_work_to_cores` — **~117 outputs/core** on the
14×10 = 140-core Blackhole grid.

## Why these dims

- **16384 outputs ÷ 140 cores ≈ 117 outputs/core** — every core does real,
  sustained work (not a one-shot), so the timed number reflects steady-state
  compute + NoC, not dispatch overhead.
- **K = 64** (4× the small example's K=16) makes each dot a meaningful inner loop.
- Operands stay small dyadics in the **quire golden zone**, so K=64 sums never
  approach the 256-bit quire bounds and every product is exact.
- The golden-sample gen and on-silicon run are both reasonable: the slow Fraction
  oracle only touches the **64 sampled** outputs (×64 products = 4096 oracle
  products), not all 1.05M.

## Throughput print (honest framing)

```
PERF: 128x64x128 exact-quire matmul across <C> cores, <T> ms, <dots/s> exact-quire-dots/s (<prods/s> exact-products/s); sample 64/64 bit-exact
```

These are **SCALAR-per-core exact-quire ops — NOT bf16-engine FLOPs.** The number
is the exact-quire dots/sec and exact-bp16-products/sec: every product is placed
unrounded into a 256-bit quire (the differentiator), and only the readout rounds.

## Timing methodology

`std::chrono::high_resolution_clock` around a **blocking** dispatch:

1. operands written once (`EnqueueWriteMeshBuffer`); program built once.
2. **WARM-UP**: one `EnqueueMeshWorkload(blocking=false)` + `distributed::Finish(cq)`
   — discarded. This triggers kernel JIT + program-cache fill + first dispatch.
3. **TIMED**: `t0 = now()`; `EnqueueMeshWorkload(blocking=false)`;
   `distributed::Finish(cq)`; `t1 = now()`. Times the steady-state matmul only.
4. read-back (`EnqueueReadMeshBuffer`) happens **after** timing, so the output DMA
   is excluded from the matmul time.

API used: `tt::tt_metal::distributed::EnqueueMeshWorkload` +
`tt::tt_metal::distributed::Finish` (declared in
`tt_metal/api/tt-metalium/distributed.hpp`), with `std::chrono` around the
blocking `Finish`.

## Correctness without a huge-golden bottleneck

The Fraction oracle is too slow to gate all 1.05M products, so the host gates a
**SAMPLE of S=64** outputs spread across the grid (4 corners + center + a strided
lattice) plus **one full gate quire** (output (127,127)), while TIMING the FULL
matmul. The kernel still computes all 16384 outputs; only the sampled subset is
checked bit-exact. The per-core inner math is byte-for-byte the on-silicon-proven
small-multicore kernel, so the un-sampled outputs are correct by construction.

## Files

| File | Role |
|------|------|
| `bposit_quire_matmul_multicore_perf.cpp` | Host: sample self-check, full-grid dispatch, warm-up + timed `Finish`, sampled bit-exact gate, `PERF:` + `PASS/FAIL` print |
| `kernels/quire_matmul_perf_baby.cpp` | Per-core kernel — SAME scalar exact-quire dot, looping its `[out_start,out_end)` slice (~117 outputs) |
| `gen_matmul_perf_golden.py` | Regenerates the SAMPLED golden header from the canonical oracle |
| `quire_matmul_perf_golden_cases.h` | Baked golden (full A/B operands, 64 sampled (i,j)+readouts, gate quire of (127,127)) — **generated, do not hand-edit** |
| `CMakeLists.txt` | Target `metal_example_bposit_quire_matmul_multicore_perf` |
| `run_on_silicon.sh` | Regen golden → build → run on physical Blackhole, time + gate sample |

## Reused, unchanged (absolute `#include`)

```
<kernel>/bp16_quire.h    # bp16_prod_to_q256, q256_add
<kernel>/bp16_encode.h   # bp16_encode_quire256 (QFRAC=96, truncation)
```

## Marshalling rules (do NOT regress)

- **Inputs A, B** are read-only and shared. Each is **ONE page**
  (`page_size == whole matrix`); every core does **one bulk `noc_async_read`** of A
  and one of B into its own L1 scratch. At N=128, K=64: A = N·K = 8192 codes = 32 KB,
  B = K·N = 8192 codes = 32 KB → **64 KB private per core**, well within the
  **1536 KB** Blackhole L1.
- **Output C** is a **1-word-per-page** buffer (`page_size == 4 B`, 16384 pages).
  Each core writes only its own output slots via `get_noc_addr(o)`, so concurrent
  cores never collide.
- **Gate quire** (8 limbs of output (127,127)) goes to a separate one-page buffer.

## Build

```bash
cd $TT_METAL_HOME
# golden is checked in; regenerate from the oracle if dims/inputs change:
( cd tt_metal/programming_examples/bposit_quire_matmul_multicore_perf && \
  python3 gen_matmul_perf_golden.py > quire_matmul_perf_golden_cases.h )

# configure once (if build_Release doesn't exist), then build the single target:
cmake -S . -B build_Release -DBUILD_PROGRAMMING_EXAMPLES=ON   # only if needed
ninja -C build_Release metal_example_bposit_quire_matmul_multicore_perf
```

## Run (physical Blackhole only — no simulator)

```bash
./tt_metal/programming_examples/bposit_quire_matmul_multicore_perf/run_on_silicon.sh
# Expect (on bit-exact match):
#   PERF: 128x64x128 exact-quire matmul across <C> cores, <T> ms, <dots/s> ...; sample 64/64 bit-exact
#   PASS: exact bp16 quire PERF MATMUL (128x64x128, 16384 dots, 1048576 exact products) across <C> cores ...
```

## Verification status

- **Host self-check (sampled subset): PASS** — verified natively with `g++
  -std=c++17 -Wall -Wextra -Werror -O2` against the REAL the scalar-kernel headers
  (`bp16_quire.h` / `bp16_encode.h`) + the generated golden: all 64 sampled
  readouts, K-order associativity (forward == reverse), and the gate (127,127)
  quire match. This same gate runs in the host program *before* the device
  dispatch.
- **On-silicon: NOT YET RUN** (a separate process drives the hardware). Build is a
  draft authored to be buildable; do not build/run here.

## Top risk for the on-silicon run

**Per-core L1 capacity for the full private A/B copy at scale.** Each core holds a
full private copy of A (32 KB) + B (32 KB) = 64 KB plus kernel/stack/CB reserve.
That is comfortably within the 1536 KB Blackhole L1, but it is the largest single
per-core L1 footprint of any example here. If a larger N is ever chosen and L1
pressure appears, the fix is to have each core **stream only the A-rows and B-cols
its output slice needs from DRAM** (read-on-demand via `InterleavedAddrGen` instead
of one full private copy) — the compute loop and golden stay identical.

Secondary risk (carried over from the small example): the **1-word-per-page output
buffer** at 16384 pages. Distinct pages mean no data race; if 4-byte page writes
misbehave on the NoC at this count, fall back to coarser per-core output paging
with host reshaping on read-back. Read-back is gated by the blocking
`EnqueueReadMeshBuffer` after `Finish`, so ordering is safe.
```
