<!-- Copyright (c) 2026 Anomly, Inc. SPDX-License-Identifier: Apache-2.0 -->
# Blackhole SFPU exact-quire kernels

The vectorized 256-bit exact quire for b-posit16 on the Tenstorrent Blackhole SFPU —
the implementation described in *A Vectorized Exact Quire for b-posit Arithmetic*
(CoNGA'26). Every kernel here ran on a physical Blackhole p150 and is gated bit-exact
against the scalar reference in `kernel/` (which is itself gated against the
arbitrary-precision oracle in `../../reference/bposit_ref.py`).

```
kernel/                 scalar reference: bp16 decode / encode, exact bp16×bp16 product
                        placement, 256-bit quire add/negate, quire→bp16/bp32 encode,
                        bf16↔bp16 codec, rounded bp16 multiply, log2 LUT.  Pure 32-bit
                        integer C, header-only, byte-identical on x86, RV32 (Blackhole baby
                        cores) and the GPU port.  bp16_quire_selftest.c (host, gcc) gates the
                        fused MADD entry points against the scalar path over 143,641 pairs.
programming_examples/   24 tt-metal programming examples (host driver + device kernels):
                        the paper's §4 lineage from lane-parallel decode to the multi-core
                        matmul, plus the baby-core evaluation kernels behind §6–§7.
golden/                 oracle-generated JSON goldens the gen_*.py scripts cross-check against.
ttnn_op/                ttnn.experimental.bposit_quire_matmul — the native op (§4.5).
tt-metal-integration.patch  registration hooks for the ttnn op (3 files, 60 lines).
```

## What each example proves

Milestone chain (paper §4, in order; each is gated bit-exact on silicon):

| example | proves |
|---|---|
| `bposit_quire_sfpu_add` | lane-parallel exact `q256_add` — 32 quires per SFPU pass; the compare-free carry trick (§4.3) |
| `bposit_decode_sfpu` | M1: lane-parallel `bp16_decode` → (sign, M, E2) |
| `bposit_product_sfpu` | M2: lane-parallel exact dyadic product (M_a·M_b, E_a+E_b) |
| `bposit_place_sfpu` | M3: windowed placement of a signed product into the 256-bit quire |
| `bposit_dot_sfpu` | M3.5: exact dot product, quire resident across K |
| `bposit_dotk_sfpu` | in-kernel K loop, quire resident in L1 (ping-pong CB) |
| `bposit_dotk_stream_sfpu` | streaming K (unbounded by L1); the §7 streaming microbenchmark |
| `bposit_matmul_sfpu` | M4: full exact-quire matmul, single core |
| `bposit_matmul_mc_sfpu` | multi-core, one output row per core |
| `bposit_matmul_mc2_sfpu` | multi-core, arbitrary M (row slices) |
| `bposit_matmul_full_sfpu` | multi-core, arbitrary (M, N, K) |
| `bposit_matmul_run` | file-I/O driver for the full matmul (callable from Python) |
| `bposit_cancel_sfpu` | catastrophic-cancellation dot: bf16 and fp32 wrong, exact quire right (§6.1) |
| `bposit_bf16_codec` | on-device bf16↔bp16 codec byte-identical to x86 (host-round-trip-free op path) |

Baby-core (scalar RISC-V) predecessors, kept because the paper's bit-exact validation
chain starts there:

| example | proves |
|---|---|
| `bposit_exact_dot` | arbitrary-length exact dot on one baby core vs fp32 cancellation |
| `bposit_exact_gemm` | arbitrary-size exact GEMM on one baby core |
| `bposit_exact_gemm_mc` | the same GEMM distributed over the Tensix grid |
| `bposit_quire_matmul_ttnn` | host driver for the ttnn op: 64-core exact matmul vs the scalar reference |

Baby-core evaluation kernels — the producers of the paper's §6/§7 numbers that are not
SFPU lineage (each ships its `gen_*_golden.py` oracle generator, its golden header, and
`BUILD_NOTES.md` with the on-silicon transcript):

| example | paper result |
|---|---|
| `bposit_quire_matmul_multicore` | multi-core exact matmul on the baby cores, one output row per core |
| `bposit_quire_matmul_multicore_perf` | §7.1: 128×64×128 on 130 cores in 2.653 ms, 395 M exact products/s, 64/64 sampled bit-exact |
| `bposit_quire_matmul_validate` | §7.1: 8 seeds × 16³, 2048/2048 outputs bit-exact incl. full 256-bit quire byte gates |
| `bposit_quire_attention` | §6.4: q·k = ±2^33 outliers — fp32 picks the wrong arg-max, the quire is correct (KL 0.403) |
| `bposit_quire_gradient` | §6.5: quire recovers 33/50/60/67 % of the gradient signal fp32 drops at T = 64/128/192/256 |
| `bposit_quire_cross_entropy` | §6.5: exact on-device cross-entropy (2.625 bits, 9/9 cases) via `bp16_log2_lut.h` |

## Building

These are tt-metal programming examples; they are not standalone. The tree was developed
against upstream `tenstorrent/tt-metal` at commit `3548ed05` (2026-06-05) on a Blackhole
p150 (tt-kmd/tt-smi stack as of June 2026). Newer tt-metal releases may need the usual
API adjustments (CB/kernel-creation signatures move), but the SFPU device kernels
(`programming_examples/*/kernels/`) are plain RISC-V C++ over `kernel/*.h` and are
API-independent.

```bash
git clone https://github.com/tenstorrent/tt-metal && cd tt-metal
git checkout 3548ed05 && git submodule update --init --recursive

# 1. Copy the examples in, and put kernel/ where the device kernels' relative includes
#    expect it: host drivers and example kernels include ../../kernel/*.h and
#    ../../../kernel/*.h from tt_metal/programming_examples/<example>/, the ttnn op
#    includes ../../../../kernel/*.h from its device/kernels/ directory.
cp -r <open-bposit>/targets/blackhole/programming_examples/* tt_metal/programming_examples/
cp -r <open-bposit>/targets/blackhole/kernel                  tt_metal/kernel
cp -r <open-bposit>/targets/blackhole/golden                  tt_metal/golden
ln -s ../../../../tt_metal/kernel ttnn/cpp/ttnn/operations/kernel      # only for step 3

# 2. Register the examples you want:
for d in bposit_quire_sfpu_add bposit_decode_sfpu bposit_product_sfpu bposit_place_sfpu \
         bposit_dot_sfpu bposit_dotk_sfpu bposit_dotk_stream_sfpu bposit_matmul_sfpu \
         bposit_matmul_mc_sfpu bposit_matmul_mc2_sfpu bposit_matmul_full_sfpu bposit_matmul_run \
         bposit_cancel_sfpu bposit_bf16_codec bposit_exact_dot bposit_exact_gemm \
         bposit_exact_gemm_mc bposit_quire_matmul_ttnn \
         bposit_quire_matmul_multicore bposit_quire_matmul_multicore_perf bposit_quire_matmul_validate \
         bposit_quire_attention bposit_quire_gradient bposit_quire_cross_entropy; do
  echo "add_subdirectory(\${CMAKE_CURRENT_SOURCE_DIR}/$d)" >> tt_metal/programming_examples/CMakeLists.txt
done

# 3. (optional) the native ttnn op
cp -r <open-bposit>/targets/blackhole/ttnn_op/bposit_quire_matmul ttnn/cpp/ttnn/operations/experimental/
git apply <open-bposit>/targets/blackhole/tt-metal-integration.patch

# 4. build and run (tt-metal's standard flow)
./build_metal.sh --build-programming-examples
./build/programming_examples/bposit_cancel_sfpu
```

(The device kernels are JIT-compiled by tt-metal from their source path, and quoted
includes resolve relative to the including file, so any layout that preserves the
relative paths above works — the layout of this directory itself does.)

Each example prints its own gate line (`BIT-EXACT`, `PASS`, or a `MISMATCH` with the
first differing limb) and exits non-zero on failure. Golden vectors are generated by the
`gen_*.py` scripts beside each example from `kernel/*.h` compiled on the host, so the
gate is against the scalar reference, never against the kernel itself.

## What is *not* here

The paper's evaluation also used an INT8-tensor-core layer matmul and a reduce-only
kernel that are not part of the exact-quire lineage; they are not in this set. The
gen_*.py generators need the arbitrary-precision oracle `bposit16_reference.py` from
https://github.com/anomly-labs/mosyne-bposit (clone it beside this repository or set
`BPOSIT16_REFERENCE_DIR`). The FPGA vector-MAC engine (§7.2) is proprietary RTL — the
exactness-critical cells extracted from it are formally proven in `../../formal/`.
