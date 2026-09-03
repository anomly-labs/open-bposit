<!-- Copyright (c) 2026 Anomly, Inc. SPDX-License-Identifier: Apache-2.0 -->
# bposit_quire_matmul_validate — RIGOROUS full-validation of the exact-quire matmul

Hardens the headline correctness claim from **sampled** to **fully
validated across random data**: runs `SEEDS=8` distinct **random** matmuls
`C[16,16] = A[16,16] · B[16,16]` on real Blackhole silicon and gates **every**
one of the `M*N = 256` outputs of **every** seed bit-exact vs the exact 256-bit
Kulisch-quire oracle — **2048 fully-validated outputs and 32768 exact bp16
products** — plus a full 256-bit quire byte-gate on one output per seed. No
sampling, no tolerance: any mismatch FAILs the run.

## What it proves vs the existing examples

| example | size | validation |
| --- | --- | --- |
| `bposit_quire_matmul_multicore` | C[8,8]=A[8,16]·B[16,8] | FULL but engineered tile (64/64 + gate) |
| `bposit_quire_matmul_multicore_perf` | C[128,128]=A·B, K=64 | LARGE but SAMPLED (64/16384 + gate) |
| **`bposit_quire_matmul_validate`** | 8 × C[16,16]=A·B, K=16 | **FULL over RANDOM data** (2048/2048 + 8 gates) |

A paper reviewer / skeptic wants "we ran N random MxKxN matrices and ALL
`M*N*SEEDS` outputs were bit-exact vs the exact-quire reference." This example is
exactly that statement, executed on silicon.

## Files

- `bposit_quire_matmul_validate.cpp` — host: full host-side self-check over all
  seeds (re-derives every output/product from the reused C headers, FAILs on
  drift before any device run), then per-seed device dispatch with a **full**
  bit-exact gate of every output + per-seed full-quire byte-gate. Reuses the
  proven `bposit_quire_matmul_multicore` host path verbatim, wrapping a **seed
  loop** (re-upload A,B per seed, re-enqueue the cached workload — the
  `..._perf` re-dispatch pattern).
- `kernels/quire_matmul_validate_baby.cpp` — the scalar exact-quire dot kernel,
  **byte-for-byte** the on-silicon-proven `quire_matmul_mc_baby.cpp` (one
  dispatch = one seed's full matmul; outputs distributed one+/core).
- `gen_matmul_validate_golden.py` — generates `quire_matmul_validate_golden_cases.h`
  from the canonical oracle (`bposit16_reference.py`), one random A,B per seed
  from a realistic mix (golden-zone `N(0,0.3)` + wide-dynamic-range/cancellation
  seeds), bp16-quantized. Reseeds until every nonzero product is quire-exact
  (see below). Runs the same per-element / order-independence / `bp16_mul.json`
  cross-checks as the multicore generator, on every seed.
- `quire_matmul_validate_golden_cases.h` — **generated**, do not hand-edit.
- `CMakeLists.txt`, `run_on_silicon.sh`.

## Reused (cited)

- Exact-quire numerics: `the scalar-kernel/deploy/tt-metalium-bposit/kernel/bp16_quire.h`
  (`bp16_prod_to_q256`, `q256_add`), `bp16_encode.h` (`bp16_encode_quire256`),
  `bp16_decode.h` — the SAME freestanding RV32IM-clean headers the proven kernels
  use, included via absolute path by both host and kernel.
- Oracle: `mosyne-bposit/kernels/bposit16_reference.py`
  (`encode_bposit16`, `decode_bposit16`, `decoded_to_fraction`, `bposit16_mul`,
  `quire256_to_bposit16`, `QUIRE_FRAC_BITS=96`).
- Host/dispatch shape: `bposit_quire_matmul_multicore` (+ `_perf` for the
  re-enqueue-cached-workload pattern), `vecadd_multi_core` (grid + split).

## Why random data stays bit-exact (the load-bearing invariant)

The oracle's `int(prod * 2^96)` and the C kernel's `bp16_prod_to_q256` (which
places `M_a·M_b` at bit `E2_a+E2_b+96` and drops the whole term if that bit `< 0`)
agree **exactly** iff every nonzero product satisfies `E2_a + E2_b >= -96` (term
fully representable in the 256-bit quire) — or the product is exactly zero. The
generator computes `(M, E2)` per code (mirroring `bp16_decode.h`) and
**rejects-and-reseeds** any matrix pair that would land a nonzero term below the
quire's LSB. With the chosen value ranges this never triggers (observed reseed
count: 0 for all 8 seeds), but the check makes bit-exactness an **invariant of
the emitted data**, not a hope. The host then independently re-derives every
output from the codes using the reused C headers at startup.

## Build

Registered in `programming_examples/CMakeLists.txt`. Build the single target:

```bash
ninja -C $TT_METAL_HOME/build_Release metal_example_bposit_quire_matmul_validate
```

(Full configure if needed:
`cmake -S $TT_METAL_HOME -B $TT_METAL_HOME/build_Release -DBUILD_PROGRAMMING_EXAMPLES=ON`.)

## Run (real silicon only — no simulator)

```bash
./run_on_silicon.sh
```

Regenerates the golden from the oracle (the slow step — pure-Python Fraction over
32768 products, ~minutes), builds if missing, then runs on the physical Blackhole
(`ARCH_NAME=blackhole`, slow dispatch, `TT_METAL_SIMULATOR` unset). Expected tail:

```
VALIDATE: 8 seeds x 16x16x16 exact-quire matmul on <C> cores — 2048 outputs, 32768 exact products, ALL bit-exact vs oracle
PASS: full-validation 2048/2048 bit-exact (incl. 8 full-quire byte-gates) on real silicon
```

## Verified locally (no hardware)

- `gen_matmul_validate_golden.py` runs to completion; all 8 seeds pass the oracle
  self-checks (per-element product == `bposit16_mul`, order-independent quire,
  `bp16_mul.json` overlap), reseed count 0.
- The **host-side numeric core** (golden header + `bp16_quire.h`/`bp16_encode.h`,
  exactly as the host program includes them) compiles clean under
  `g++ -std=c++17 -Wall -Wextra -Werror -Wno-unused-function` and reports
  `2048 outputs / 32768 products`, `per_elem/readout/assoc/gate all = 1`.

## Top risk

The kernel `#include`s the bp16 headers by **absolute path** (same caveat as the
proven `quire_matmul_mc_baby.cpp`): if the kernel-compile sandbox rejects the
absolute include, copy the three headers into `kernels/` or pass the dir via the
`CreateKernel` defines/`-I`. The single-core/multicore paths build with the
absolute include today, so this is expected to work as-is.
