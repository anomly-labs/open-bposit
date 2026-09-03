<!-- Copyright (c) 2026 Anomly, Inc. SPDX-License-Identifier: Apache-2.0 -->
# bposit_quire_attention — build & verification notes

A -on-silicon result for the **attention** failure mode: on a **real
Tenstorrent Blackhole baby (RISC-V) core**, the EXACT 256-bit Kulisch b-posit16
quire computes the **correct** transformer attention scores `s_j = q·k_j` under
**massive-activation outliers** — and a **genuine IEEE fp32** naive dot does not,
**flipping the softmax argmax** onto the wrong key. Both dot products run on the
same core over the same inputs; the host gates both against the exact
mathematical ground truth, then softmaxes each score vector and shows the flip.

Why this is a result: transformer attention scores are corrupted when a few
activation dims are huge outliers ("massive activations", Sun et al. 2024,
arXiv:2402.17762, which drive attention sinks). A low-precision dot lets the
outlier products swamp / catastrophically cancel away the many small-but-meaningful
contributions, distorting the softmax. b-posit's tapered precision + the exact
256-bit quire keep the full dot product exactly. Framing: an SDK note (not published) `src/
spacetime/attention_precision.py`.

This is **authoring + buildable draft only** — it does NOT build or run on
hardware (another process does that). Everything below was verified host-side.

## The problem (head_dim D = 32, one query q, J = 3 keys)

Per vector, the 32 dims play three roles:

| dims | role | values |
|------|------|--------|
| 0 | +outlier | `q0 = 2^17`, `k_{j,0} = 2^16` → product **+2^33** |
| 1..29 | the **signal** | `q_i = 1/8`, `k_{j,i} ∈ {+1,−1}` → exact dot = (Σ signal)/8 |
| 30 | −outlier | `q30 = 2^17`, `k_{j,30} = −2^16` → product **−2^33** (exact cancel) |
| 31 | per-key **tail** | `q31 = 1/8`, `k_{j,31}` small |

The two outlier products (±2^33) **cancel exactly** in each key's dot, so the
exact score is the small-dim signal (+ tiny tail). All q,k values are
**bp16-exact** (asserted at gen time by a round-trip check).

The signal makes **key0** the true winner; the tails are ordered the *wrong* way.

| key | exact / quire score | fp32 naive score |
|-----|--------------------|------------------|
| `key0_strong_signal` | **3.6875** (`0x4760`) | 0.0625 |
| `key1_mid_signal` | 2.375 (`0x44c0`) | 0.25 |
| `key2_weak_signal_big_tail` | 2.625 (`0x4540`) | **1.0** |

Softmax over each score vector:

```
quire/exact softmax: [0.6193, 0.1667, 0.2140]   argmax = key0   (TRUE)
fp32        softmax: [0.2101, 0.2534, 0.5365]   argmax = key2   (WRONG)
```

**ARGMAX FLIP**: exact/quire pick key0, fp32 picks key2.
Divergence: **KL(exact‖fp32) = 0.403**, **max softmax-weight error = 0.409**.

Why fp32 fails: in the sequential fp32 naive sum (DIM ORDER), the +2^33 outlier
product lands first; the O(1) signal terms fall below its ULP (~2^10) and are
swallowed; then −2^33 returns the sum to ~0 having recorded **none** of the
signal. fp32 then ranks keys only by the tiny post-cancellation tail. The exact
256-bit quire represents ±2^33 exactly and cancels them exactly — the signal
survives.

Expected output shape:
```
s_0 (key0_strong_signal): quire=3.6875 fp32=0.0625 exact=3.6875  [quire EXACT, fp32==baseline OK, ...]
s_1 (key1_mid_signal):    quire=2.3750 fp32=0.2500 exact=2.3750  [quire EXACT, fp32==baseline OK, ...]
s_2 (key2_weak_signal_big_tail): quire=2.6250 fp32=1.0000 exact=2.6250  [quire EXACT, fp32==baseline OK, ...]

softmax weights:
  quire/exact: 0.6193 0.1667 0.214   argmax=key0
  fp32       : 0.2101 0.2534 0.5365  argmax=key2

PASS: exact-quire attention matches exact softmax (argmax/weights), fp32 distorts it — real silicon
  fp32 picks key2 (key2_weak_signal_big_tail); quire/exact pick key0 (key0_strong_signal) — the TRUE argmax. ARGMAX FLIP.
  divergence: KL(exact||fp32)=0.403  max softmax-weight error=0.4092
```

## What the kernel does (per key, one baby-core invocation)

1. **Exact quire dot** — `q256 = Σ_i bp16_prod_to_q256(q_i, k_i)`: the EXACT
   bp16×bp16 product placed un-rounded into the 256-bit Kulisch quire
   (QUIRE_FRAC_BITS=96), exact ripple-carry accumulate, single truncating readout
   (`bp16_encode_quire256`). This is the proven DOT path from
   `bposit_quire_reduce/kernels/quire_dot_baby.cpp`, reusing the scalar-reference
   `bp16_quire.h` / `bp16_encode.h` headers **verbatim**.
2. **Genuine fp32 naive dot** — `acc = fp32_add(acc, prod_i)` sequential running
   sum, RTNE, in **pure integer arithmetic** (`kernels/ieee_softfp32.h`). Each
   per-dim product `q_i·k_i` is formed **exactly** by `bp16_prod_to_fp32_bits`
   directly from the dyadic decode (M_q·M_k < 2^22 fits the fp32 significand
   losslessly), so the float error is **purely accumulation** — exactly what a
   real fp32 attention kernel does.
3. The host runs this once per key, gates the quire limbs + readout and the fp32
   bits against the golden, then softmaxes both score vectors and asserts the flip.

## Files

- `bposit_quire_attention.cpp` — host driver (mirrors `bposit_quire_vs_float.cpp`:
  single-page MeshBuffer marshalling, golden self-check before the device run,
  per-key gate, softmax + argmax + flip + PASS/FAIL gate). Query buffer shared
  across keys; key/out buffers per key.
- `kernels/quire_attention_baby.cpp` — the baby-core kernel (both dot products).
- `kernels/ieee_softfp32.h` — pure-integer IEEE-754 binary32 add (RTNE), reused
  **verbatim** from `bposit_quire_vs_float`.
- `attention_golden_cases.h` — **generated** golden (do not hand-edit).
- `gen_attention_golden.py` — regenerates the golden from the oracle
  (`mosyne-bposit/kernels/bposit16_reference.py`); asserts every value bp16-exact,
  quire readout == exact score, and the fp32 argmax flip.
- `CMakeLists.txt` — target `metal_example_bposit_quire_attention` (registered in
  `tt_metal/programming_examples/CMakeLists.txt`).
- `run_on_silicon.sh` — physical-device runner (no simulator, blackhole, slow
  dispatch; regenerates the golden, gates on PASS).

## Build / run

```bash
# regenerate golden (optional — host self-checks it anyway):
python3 gen_attention_golden.py

# build (real arch):
cmake -S "$TT_METAL_HOME" -B "$TT_METAL_HOME/build_Release" -DBUILD_PROGRAMMING_EXAMPLES=ON
ninja -C "$TT_METAL_HOME/build_Release" metal_example_bposit_quire_attention

# run on physical Blackhole (no simulator):
./run_on_silicon.sh
```

## Verification done host-side (no hardware touched)

1. **Construction genuinely flips** — the oracle (`gen_attention_golden.py`,
   exact `Fraction` arithmetic + genuine fp32 `struct` round-trip) asserts: every
   q,k value is bp16-exact; the quire-dot readout == the exact score for every key
   (quire error 0); and the fp32 naive-dot softmax argmax ≠ the exact argmax. If
   the construction ever failed to separate, gen raises — it did not.
2. **fp32 path is genuine fp32** — `fp32_add` is the reused `ieee_softfp32.h`
   already cross-checked bit-for-bit vs native C `float` over ~19.8M random pairs
   in `bposit_quire_vs_float`. The per-dim product `bp16_prod_to_fp32_bits` is
   **exact** (built from the dyadic decode, no rounding multiply) — confirmed:
   replaying the kernel's exact-product + fp32-add path host-side reproduces the
   golden fp32 bits (`0x3d800000`, `0x3e800000`, `0x3f800000`) for all 3 keys.
   So the float distortion is real fp32 behavior, not a contrivance.
3. **Full pipeline self-check** — replaying the kernel's quire-dot
   (`bp16_prod_to_q256` + `q256_add` + `bp16_encode_quire256`) host-side over the
   baked keys reproduces the golden quire limbs and bp16 readouts exactly; each
   readout decodes to the exact score. The host program embeds this same
   self-check and refuses to run the device if the baked golden ever drifts from
   the reused C headers.
4. **32-bit-integer-only kernel** — both the quire path and `bp16_prod_to_fp32_bits`
   use only 32-bit integer ops (the product `M_q·M_k` is an 11b×11b → 22b multiply
   fitting a single uint32 — **no 64-bit `__muldi3`**). The fp32 multiply was
   deliberately *removed*: forming the product exactly from the decode is both
   simpler and provably lossless, sidestepping a fragile soft-float multiply.
   Kernel math helpers compile clean under `g++ -Wall -Wextra -Werror`.

## Reused (cite)

- `bposit_quire_reduce/{bposit_quire_reduce.cpp, kernels/quire_dot_baby.cpp}` — the
  proven exact-quire DOT path (`bp16_prod_to_q256` accumulation) + single-page /
  bulk-NoC marshalling, host self-check, PASS/FAIL gate idiom.
- `bposit_quire_vs_float/{bposit_quire_vs_float.cpp, kernels/ieee_softfp32.h}` — the
  host driver shape and the pure-integer IEEE fp32 add (reused verbatim) for the
  honest float baseline.
- `the scalar-kernel/deploy/tt-metalium-bposit/kernel/{bp16_quire.h, bp16_encode.h, bp16_decode.h}`
  — the exact quire numerics (included verbatim by absolute path), already
  bit-exact on this Blackhole.
- `mosyne-bposit/kernels/bposit16_reference.py` — the oracle (exact `Fraction`
  arithmetic) that generates the ground-truth golden.
- SDK framing: `attention_precision.py` (not published) (attention
  under massive-activation outliers — posit escapes the range-vs-precision tradeoff).

## Top risk

The fp32 score depends on **accumulation order** (outliers first, then signal,
then the −outlier). The kernel sums in DIM ORDER over the L1 array, matching the
oracle's `gen_attention_golden.py` (which sums `zip(q, k)` in dim order). If the
device ever reordered the running sum (it does not — it is a scalar sequential
loop on one baby core), the fp32 result could differ; the host gates the device
fp32 bits == the baked baseline, so any reorder would be **caught**, not silently
accepted. Secondary risk (shared with the sibling examples): if the rv32 kernel
toolchain emits a libgcc helper for some construct, the link fails — mitigated by
the 32-bit-integer-only discipline above and absolute-path includes. No hardware
was touched; the first real-silicon run is the actual proof.
