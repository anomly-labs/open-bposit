<!-- Copyright (c) 2026 Anomly, Inc. SPDX-License-Identifier: Apache-2.0 -->
# bposit_quire_gradient — build & verification notes

The **training** result on real silicon: on a Tenstorrent Blackhole baby
(RISC-V) core, accumulating a per-parameter **gradient** `g = Σ_t δ_t` (a sum of many
tiny per-step contributions) in an **exact 256-bit b-posit16 quire** recovers the
tiny-tail gradient mass that an IEEE **fp32** running accumulator *swallows*. This is
the training analogue of the inference cancellation demo in `bposit_quire_vs_float`.

## The training problem this demonstrates

In large-batch / many-step training, a parameter's gradient is a **sum of many tiny
per-example / per-step contributions**. A large early contribution inflates the
running accumulator to magnitude `M`; once the running sum is at `M`, each subsequent
tiny `δ_t` that is **below `ULP(M)`** adds nothing in low precision — it is *swamped*
and lost (the swamping / stale-gradient / gradient-underflow problem). The exact quire
is a fixed-point Kulisch register wide enough for the full b-posit dynamic range, so it
has **no ULP-relative-to-a-running-sum**: every `δ_t` is placed exactly and nothing is
dropped.

## The construction (genuine binary32 loss)

Per config, `g = Σ_t δ_t`:

| term | value | role |
|------|-------|------|
| `δ_0`          | `+2^14` (= 16384)     | large early spike → running sum `M = 2^14` |
| `δ_1 … δ_T`    | `+2^-14` each (`τ`)   | `T` tiny per-step gradients, **`τ ≪ ULP(M) = 2^-9`** so each is swamped by fp32 |
| `δ_{T+1}`      | `-2^14`               | spike cancelled by an opposite-sign later batch (running sum back to ~0) |
| `δ_{T+2}`      | `+1/128` (`g0`)       | coarse surviving gradient fp32 keeps |

Exact gradient `g = g0 + T·τ`. fp32 freezes at `g0 = 0.0078125` (every `τ` was lost
under `M`); the quire keeps all of `g0 + T·τ` exactly. The fraction fp32 loses is
`T·τ / (g0 + T·τ)`, which **grows with the step count T**:

| T   | N codes | exact `g`   | quire→bp16   | fp32 `g`    | **fp32 lost** |
|-----|---------|-------------|--------------|-------------|---------------|
| 64  | 67      | 0.01171875  | `0x2600` ✓   | 0.0078125   | **33.3 %**    |
| 128 | 131     | 0.01562500  | `0x2800` ✓   | 0.0078125   | **50.0 %**    |
| 192 | 195     | 0.01953125  | `0x2900` ✓   | 0.0078125   | **60.0 %**    |
| 256 | 259     | 0.02343750  | `0x2a00` ✓   | 0.0078125   | **66.7 %**    |

All `δ_t` are bp16-exact and the exact `g` is bp16-representable, so the quire readout
**equals the exact gradient** (quire error 0). The fp32 column is genuine IEEE-754
binary32 RTNE running-sum behaviour (verified bit-exact vs native C `float` and vs
Python `struct` round-trip).

## What is reused (cited)

Nothing in the numeric core is reimplemented — all of it already PASSes bit-exact on
this Blackhole:

- **Exact quire kernel path** — `bp16_to_q256`, `q256_add`, `bp16_encode_quire256`
  from `the scalar-kernel/deploy/tt-metalium-bposit/kernel/bp16_quire.h` + `bp16_encode.h` +
  `bp16_decode.h` (the proven `bposit_quire_reduce` reduce/sum path).
- **fp32 baseline** — `ieee_softfp32.h` (`fp32_add`) and the `bp16 code → fp32 bits`
  conversion shape, reused **verbatim from `bposit_quire_vs_float`** (host-verified
  bit-exact vs native C `float`).
- **Oracle** — `mosyne-bposit/kernels/bposit16_reference.py`
  (`encode/decode_bposit16`, `decoded_to_fraction`, `quire256_to_bposit16`,
  `QUIRE_FRAC_BITS`) is the exact-Fraction ground truth baked by
  `gen_gradient_golden.py`.

No large static LUT is included in the kernel (baby-core `.data` stays tiny, as in the
sibling examples).

## Files

- `kernels/quire_gradient_baby.cpp` — the baby-core kernel: accumulate the gradient
  both ways (exact quire + genuine fp32 running sum), write back 8 quire limbs + bp16
  readout + fp32 bits.
- `bposit_quire_gradient.cpp` — host: per-config golden self-check against the reused C
  headers, device run (single-page MeshBuffer marshalling), gate quire == exact
  gradient and fp32 == genuine baseline, report the lost fraction.
- `gen_gradient_golden.py` — generates `quire_gradient_cases.h` from the oracle
  (source of truth). Sweeps `T ∈ {64,128,192,256}`.
- `quire_gradient_cases.h` — **generated**; do not hand-edit.
- `CMakeLists.txt` — target `metal_example_bposit_quire_gradient`.
- `run_on_silicon.sh` — regenerate golden → check device → build → run on the physical
  Blackhole (no simulator).

## Build

Registered in `tt_metal/programming_examples/CMakeLists.txt`:

```
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/bposit_quire_gradient)
```

Build command:

```bash
cmake -S $TT_METAL_HOME \
      -B $TT_METAL_HOME/build_Release \
      -DBUILD_PROGRAMMING_EXAMPLES=ON          # only if build_Release not configured
ninja -C $TT_METAL_HOME/build_Release \
      metal_example_bposit_quire_gradient
```

Run on real silicon (ARCH_NAME=blackhole, slow dispatch, no simulator):

```bash
./run_on_silicon.sh            # whole T sweep
./run_on_silicon.sh 1          # single config index (0..3)
```

## Expected output (per config + headline)

```
T=64 running_sum~16384: quire_g=0.01171875 fp32_g=0.0078125 exact_g=0.01171875  fp32_lost=0.333 (33.3%)  [quire EXACT vs oracle, device fp32==baseline OK]
...
PASS: exact-quire gradient accumulation recovers up to 66.7% of gradient that fp32 drops; quire bit-exact vs oracle — real silicon (4/4 configs)
```

## Verification done host-side (before any device run)

Compiled a standalone harness including the **real** reused C headers
(`bp16_quire.h`, `bp16_encode.h`, `bp16_decode.h`, `ieee_softfp32.h`) and the generated
`quire_gradient_cases.h`, replicating the host self-check with **no** tt-metal device.
Under `-Wall -Wextra -Werror` it confirms, for all four configs: quire limbs match the
baked golden, the bp16 readout matches, the fp32 baseline matches, the readout decodes
to exactly the exact gradient (quire error 0), and fp32 genuinely loses 33/50/60/67 %.
**ALL HOST-SIDE CHECKS PASS.** The on-device run is what `run_on_silicon.sh` performs
(handled by the build+run process, not here).

## Known pitfalls watched (siblings hit these)

- **Distinct names for vars vs functions** — no shadowing; the per-config driver is
  `run_config(...)` and variables (`quire_g_val`, `fp32_lost_frac`, …) are all distinct.
- **All `EnqueueReadMeshBuffer` blocking=true** — both reads (result + debug) are
  `blocking=true`.
- **`-Werror` cleanliness** — header includes that pull in a wider op set than the host
  calls are wrapped in `#pragma GCC diagnostic ignored "-Wunused-function"` (same as the
  sibling host); the host self-check harness builds clean under `-Werror`.
- **No large static LUT in the kernel** — only the exact-quire + softfp32 integer paths.
- The largest config has **N = 259** input words; the single-page DRAM/L1 buffers are
  sized from `N` per config (`page_size == N*4`), keeping the proven one-page bulk-NoC
  marshalling.
```
