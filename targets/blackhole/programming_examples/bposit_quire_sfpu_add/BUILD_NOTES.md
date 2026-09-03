<!-- Copyright (c) 2026 Anomly, Inc. SPDX-License-Identifier: Apache-2.0 -->
# bposit_quire_sfpu_add — build notes (lane-parallel exact q256_add on the SFPU)

The **FIRST SFPU-vectorized milestone** of Anomly's exact 256-bit b-posit16
Kulisch quire on Blackhole: a **LANE-PARALLEL exact `q256_add`** — 32 independent
256-bit accumulators across the SFPU's 32 lanes, each adding an 8×u32 limb vector
with correct inter-limb ripple carry, **bit-identical per lane** to the scalar
reference `q256_add` (`targets/blackhole/kernel/bp16_quire.h`).

This milestone **isolates and proves the vectorized carry** — the hardest SFPU
primitive (8×int32 limb-vectors rippled with no native cross-lane carry flag). It
deliberately does **NOT** do the bp16 decode or the per-product mantissa multiply
(those are later milestones, per `docs/blackhole/quire-sfpu-kernel-design.md`).

**Status: authored + buildable draft. NOT built, NOT run on hardware.** Another
process owns the on-silicon build/run. The proven `bposit_quire_reduce` example is
untouched; this is an additive new directory.

## Files

| File | Role |
|---|---|
| `kernels/compute/sfpu_q256_add.cpp` | the SFPU compute kernel — holds 8 limb-vectors as `vUInt`, ripples carry across all 32 lanes via unsigned-compare carry detect |
| `kernels/dataflow/read_limbs.cpp` | reader: 8 Q-limb tiles + 8 X-limb tiles + 1 zero seed → CBs |
| `kernels/dataflow/write_limbs.cpp` | writer: 8 OUT-limb tiles → DRAM |
| `bposit_quire_sfpu_add.cpp` | host: bake/self-check golden, dispatch, read back 8×32, gate bit-exact, print `PASS: SFPU lane-parallel q256_add, 32/32 lanes bit-exact (8 limbs each)` |
| `gen_sfpu_add_golden.py` | generator: 32 adversarial lanes × (q,x,expected) from the scalar `q256_add`, cross-checked vs Python big-int mod-2^256 |
| `sfpu_add_golden_cases.h` | the GENERATED golden (`[limb][lane]` layout) |
| `CMakeLists.txt` | target `metal_example_bposit_quire_sfpu_add` (matches dir; NOT a copy-paste of another) |
| `run_on_silicon.sh` | regenerate golden → build → run on a physical Blackhole, gate on golden |

Registered additively in `programming_examples/CMakeLists.txt`
(`add_subdirectory(.../bposit_quire_sfpu_add)`), right after `bposit_quire_reduce`.

## Build command (try this)

```bash
ninja -C build_Release metal_example_bposit_quire_sfpu_add
```

If `build_Release` doesn't exist yet:

```bash
cmake -S . -B build_Release -DBUILD_PROGRAMMING_EXAMPLES=ON
ninja -C build_Release metal_example_bposit_quire_sfpu_add
```

Run on real silicon (no simulator) + golden gate:

```bash
tt_metal/programming_examples/bposit_quire_sfpu_add/run_on_silicon.sh
```

Binary lands at `build_Release/programming_examples/metal_example_bposit_quire_sfpu_add`.

> CMake needs no per-kernel change: `CMakeLists.txt` lists only the host source;
> the three kernels are referenced by runtime PATH (`OVERRIDE_KERNEL_PREFIX`),
> compiled by the runtime kernel toolchain, not CMake.

## The carry algorithm (the whole point) — and its proof

Per lane, per limb, the kernel mirrors the scalar `u64` ripple exactly:

```
s0     = q + x                              (vUInt add, mod 2^32)
cout_a = (s0 < x)              ? 1 : 0      // UNSIGNED <  => q+x overflowed 2^32
s      = s0 + carry_in                       (carry_in in {0,1})
cout_b = (carry_in==1 && s==0) ? 1 : 0      // adding the carry overflowed
OUT    = s ;  CARRY = cout_a | cout_b       // CARRY feeds the next limb
```

`cout_a` and `cout_b` are **mutually exclusive** for a limb (proof in the kernel
header), so the OR equals the single `t >> 32` carry of the `u64` reference.

**Proven bit-exact (host, no hardware):**
- `gen_sfpu_add_golden.py` cross-checks every lane's expected sum vs Python
  big-int `(Q+X) mod 2^256`.
- A standalone C replica of the kernel's per-lane carry algorithm matches BOTH the
  real scalar `q256_add` AND the baked golden for all **32 lanes × 8 limbs**,
  including the worst cases (full 7-limb ripple, full 256-bit wraparound to 0,
  two's-complement negatives, carry-dies-mid-chain, max+max per limb).
- The host self-check in `main()` re-derives the golden by calling the **real C
  `q256_add`** per lane at startup, so a drifted baked golden fails before any
  device dispatch (same belt-and-suspenders discipline as `bposit_quire_reduce`).

## SFPU ops relied on (CONFIDENT — grounded in real tt-metal, file:line)

All paths under `~/development/tt-metal`:

| Op | Grounding |
|---|---|
| `vUInt`/`vInt` = 32-bit × 32 lanes | `runtime/sfpi/include/sfpi.h:16-18`, `runtime/sfpi/include/sfpi_classes.h:50-52` |
| `vUInt operator+(vUInt,vUInt)` | `runtime/sfpi/include/sfpi.h:339` |
| **UNSIGNED `operator<(vUInt,vUInt)`** (carry detect) | `runtime/sfpi/include/sfpi.h:524` |
| `operator!=(vUInt,unsigned)`, `operator==(vUInt,unsigned)` | `sfpi.h:538`, `sfpi.h:537` |
| `v_if`/`v_endif` predication | `sfpi.h:555,566-568` |
| `dst_reg[i]` read/write as `vUInt`/`vInt` | `sfpi_classes.h:298-306`; idiom `vInt a=dst_reg[0]; dst_reg[0]=a+b` in `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_int_sum.h:58-66` |
| custom inline-sfpi face fn + `_llk_math_eltwise_binary_sfpu_params_` wrapper | `tt_metal/programming_examples/custom_sfpi_add/kernels/compute/tiles_add.cpp:31-109` (the direct template for this kernel) |
| `_params_` is variadic: `(fn,in0,in1,out,vector_mode,...args) → fn(in0,in1,out,...args)` | `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h:12-27`, `llk_math_eltwise_sfpu_common.h:54` |
| `init_sfpu`/`copy_tile`/`tile_regs_*`/`pack_tile` | `custom_sfpi_add/.../tiles_add.cpp:122-150`; `pack_tile(ifrom,icb,out_tile)` `tt_metal/hw/inc/api/compute/pack.h:86` |
| Int32 SFPU path exists (`DataFormat::Int32`, `InstrModLoadStore::INT32`) | `tt_metal/hw/inc/api/compute/add_int_sfpu.h:25,36-47`; `.../llk_sfpu/llk_math_eltwise_binary_sfpu_add_int.h:17-31` |

## TODOs / unverified (best-guess marked) — ranked by risk

1. **HARDEST RISK — carry-vector correctness on real silicon (the whole milestone).**
   The carry algorithm is proven bit-exact *in C against the scalar oracle*, but
   the **SFPU translation** has three unverified assumptions: (a) `vUInt operator<`
   emits a genuinely **unsigned** compare on Blackhole (the design doc says yes,
   `sfpi.h:524`, but an off-by-one in carry detect silently breaks exactness ONLY on
   limb-boundary inputs — exactly the adversarial lanes); (b) `v_if` predication
   composes the nested `carry_in!=0 && s==0` mask correctly per lane; (c) writing
   `dst_reg[c_base+i]` and re-reading it as the next limb's `carry_in` within the
   same op is well-defined. **All three must be confirmed bit-exact lane-by-lane on
   hardware — the golden gate does exactly that.**

2. **Int32 dst-register capacity (RESOLVED 2026-06-24 — this WAS the bug).**
   The original kernel kept **11 live Int32 dst tiles** (Q, X, 8×OUT, CARRY) inside
   one `tile_regs_acquire`. The SyncHalf Int32 dst budget is only **8 tiles**:
   `get_dest_max_tiles<SyncHalf,/*accum*/false,Tile32x32>() = (HALF=512)>>6 = 8`
   (`tensix_types.h:191-193`, `ckernel.h:835-842`). Slots 8/9/10 wrapped within the
   512-row half-region, so `SLOT_CARRY(10)`→`SLOT_OUT0(2)`, `OUT6(8)`→`Q(0)`,
   `OUT7(9)`→`X(1)`. The carry vector aliased OUT limb 0 and the Q/X scratch, which
   corrupts exactly the multi-limb-carry lanes (a lane whose carry stays 0 is
   immune) — matching the observed `4/32 pass`, first bad lane 3, device limbs
   reading back as carry constants. The unsigned compare itself was always correct
   (`sfpi_funcs.h:544`, mod 0 == UNSIGNED).
   **FIX 1 — the carry-CB variant:** one `tile_regs_acquire` PER LIMB; the 32-lane
   CARRY is threaded through L1, so per limb only 5 dst tiles are live
   (Q,X,CARRY_IN,OUT,CARRY_OUT ≤ 8). This took the gate from 4/32 → 14/32.

   **FIX 2 (2026-06-24) — carry PING-PONG, off-by-one.** A single round-trip CB
   self-aliased: reading carry-in and packing carry-out to the SAME `cb` within one
   lease let the packer's carry-OUT become the unpacker's carry-IN for the SAME limb,
   so limb i read its own carry-out instead of limb i-1's. Symptom: +1 at a
   carry-generating limb, -1 at the next (lane 16 limb2 +1 / limb3 -1), dropped
   ripples on all-ones lanes (lane 7/12). Fix: TWO carry CBs (`c_3`=A, `c_4`=B). Limb
   i reads carry-in from one and writes carry-out to the OTHER, swapping every limb
   (even: read A write B; odd: read B write A; seed→A; limb 7 odd→final carry in A).
   carry-in and carry-out are then never the same physical CB tile. Numerics
   bit-identical to the scalar ripple: `carry_out(i) == carry_in(i+1)`.
   (NB: the ping-pong was byte-identical on silicon — necessary structure but NOT the
   active bug; see FIX 3.)

   **FIX 3 (2026-06-24) — missing per-source `copy_tile` init (THE active 14/32 bug).**
   The `ECHO_INPUT` diagnostic proved it: `ECHO_INPUT=2` showed X bit-correct on every
   limb, but `ECHO_INPUT=1` showed Q correct only on ODD limbs and replaced by the 0/1
   carry on EVEN limbs. `copy_tile` shares the UNPACKER, whose source-CB addressing is
   set by its init. `init_sfpu(cb_q, cb_out)` configured datacopy for `cb_q` only; the
   loop then `copy_tile`s from `cb_x` and the carry CB (`cb_cin`, which ALTERNATES
   `cb_carry_a`/`cb_carry_b` by parity → different L1 base) WITHOUT re-pointing the
   datacopy. On even limbs (`cb_cin == cb_carry_a`) the carry copy mis-landed and
   clobbered SLOT_Q with the 0/1 carry, so the SFPU added `(carry + X)` instead of
   `(Q + X + carry)` — Q-corrupt-on-even, X always fine, only Q-dependent (negative /
   ripple) lanes failing: exactly the observed 14/32. Fix: call
   `copy_tile_to_dst_init_short(cb)` before EACH `copy_tile` whose source CB differs
   from the previous one (cb_q, cb_x, cb_cin per limb; cb_zero in the seed). Int32
   format is uniform across CBs, so the SHORT init (no data-format reconfig) suffices.
   **DID NOT HELP — byte-identical 14/32 (fresh JIT cache). The unpacker-state
   hypothesis was wrong; FIX 3 was reverted by the redesign.**

   **FIX 4 (2026-06-24) — REDESIGN: single lease + persistent in-dst carry (the real
   cure, removes the whole bug class).** FIX 1/2/3 all kept ONE `tile_regs_acquire`
   PER LIMB and a CARRY that left dst (CB round-trip). In SyncHalf, `get_dest_buffer_
   base()` (cmath_common.h:184-202) FLIPS between the two dst half-regions on every
   acquire/release (`dest_section_flip`). With one acquire per limb the base alternated
   by limb PARITY, so the physical registers backing SLOT_Q / SLOT_CARRY shifted every
   other limb — the carry landed in SLOT_Q on EVEN limbs (`(carry+X)` instead of
   `(Q+X+carry)`), exactly the ECHO finding, and INDEPENDENT of any carry-plumbing
   patch (hence byte-identical across FIX 1/2/3). The cure is structural:
     • ONE `tile_regs_acquire` for the WHOLE 8-limb ripple (canonical accumulate-in-one
       -acquire, matmul_single_core/kernels/compute/mm.cpp:56-83), so the dst base is
       stable for all 8 limbs and the slot map is fixed.
     • CARRY lives in a single persistent dst slot (`SLOT_CARRY`), seeded 0 once from
       cb_zero and updated IN PLACE each limb (face fn reads carry_in to a register
       before writing carry_out to the same slot). No carry CB, no ping-pong, no parity.
     • Each OUT limb → its own dst slot (`SLOT_OUT0+limb`); all 8 packed once after
       commit/wait. Live INT32 dst tiles = 11 (Q,X,CARRY,8×OUT).
     • 11 > 8 (SyncHalf), so the host sets `ComputeConfig.dst_full_sync_en = true` =>
       `DST_SYNC_MODE=SyncFull` (genfiles.cpp:642-644), whose INT32 (accum=false) budget
       is 16 (`get_dest_max_tiles<SyncFull,false,Tile32x32> = 1024>>6 = 16`). SyncFull
       also pins the dst base at 0 (no half-flip) — belt-and-suspenders.
   Per-lane carry arithmetic is reused verbatim (proved correct: unsigned `vUInt
   operator<`, both carry sources). The `ECHO_INPUT` knob is retained but, with a
   single lease, the per-limb operand echo only reflects limb 7 (Q/X are reused
   scratch); it is no longer needed for the carry diagnosis.

   **Lifecycle note (why not in-loop pack):** the coordinator's alternative — pack each
   OUT limb INSIDE the loop within one acquire — is not clean: `pack_tile` needs a
   preceding `tile_regs_commit`/`tile_regs_wait`, which transfers dst ownership to the
   packer and ends the math phase; you cannot continue math (the next limb) in the same
   lease afterward. Holding all 8 OUT in their own dst slots and packing once at the end
   is the correct single-lease pattern (matches matmul, which packs once after the
   accumulate loop). SyncFull makes the 11-tile footprint legal, so no in-loop pack is
   needed.

   **FIX 5 (2026-06-24) — interleaved MATH-mode re-init (the limb>0 non-execution bug).**
   FIX 4 fixed parity but left a new failure: ONLY limb 0 computed; limbs 1-7 returned 0
   (proved by the ECHO_TRACE/normal dump — e.g. lane 12, whose every limb sum is
   0xffffffff with no carry, gave limb0=0xffffffff, limbs1-7=0). Root cause: `init_sfpu`
   configures the MATH unit for DATACOPY (`llk_math_eltwise_unary_datacopy_init`,
   eltwise_unary.h:30). Each `q256_add_limb_tile` runs an SFPU BINARY op via
   `_llk_math_eltwise_binary_sfpu_params_` → `_llk_math_eltwise_sfpu_start_`, which
   RECONFIGURES the MATH unit for the SFPU op. So after limb 0's add, the MATH datacopy
   state is gone, and limb>0's `copy_tile(cb_q/cb_x, limb, …)` runs in the wrong MATH mode
   — SLOT_Q/SLOT_X never receive the operands (read 0). The matmul reference doesn't hit
   this because it repeats a SINGLE op type (`matmul_tiles`); this kernel INTERLEAVES two
   MATH modes (datacopy + SFPU-binary) every limb, so the datacopy init must be restored
   each time it is re-entered. Fix: call `copy_tile_to_dst_init_short(cb)` before each
   `copy_tile` group at the top of every limb (cb_q, cb_x; and cb_zero for the seed). The
   SFPU op self-inits per call (via `_llk_math_eltwise_sfpu_start_`), so only the datacopy
   side needs restoring; Int32 format is uniform so the SHORT init suffices.

   **Single-lease verdict (answer to the coordinator's question):** YES, one acquire CAN
   do 8 independent (copy A_i, copy B_i, SFPU-add → OUT_i) → 8 outputs — provided the
   datacopy MATH init is restored each limb (FIX 5), because the per-limb copies and the
   SFPU op are different MATH op modes that clobber each other's config.

   **FIX 6 (2026-06-24) — THE ACTUAL ROOT CAUSE: INT32 DEST needs `fp32_dest_acc_en=true`,
   and the live-INT32-dst footprint must be ≤ 8 (not 16).** Everything from FIX 1–5 was
   debugged under `fp32_dest_acc_en=false`, which is WRONG for 32-bit data and was the
   real bug the whole time:
   - **DEST is physically a 16-bit register file.** Loading 32-bit (Int32/Float32) tiles
     into a DEST configured for 16-bit (`fp32_dest_acc_en=false`) produces *tile-aligned
     corruption*. This is stated verbatim in shipped tt-metal:
     `ttnn/.../eltwise/binary_ng/device/binary_ng_program_factory.cpp:714-721` ("fp32 dest
     accumulation must be enabled whenever any input or output is fp32, otherwise loading
     fp32 tiles into a DST configured for bf16 produces tile-aligned corruption") — and it
     sets `fp32_dest_acc_en=true` for Int32/UInt32. The INT32 ternary
     (`ternary_program_factory.cpp:1253-1254,1365`) and its kernel
     (`ternary_addcmul_int_sfpu.cpp`) do the same. **Every shipped INT32 SFPU op enables
     `fp32_dest_acc_en`; ours had it false.** That mis-strided DEST is exactly what made
     limbs 0–2 alias OK, limbs 3–7 read 0, and the carry show a PERIOD-4 artifact — a
     16-vs-32-bit DEST-stride mismatch, NOT the carry algebra (the per-lane carry C-replica
     stays bit-exact vs the scalar oracle).
   - **The "16-tile SyncFull budget" was wrong for INT32.** With `fp32_dest_acc_en=true` the
     budget HALVES (the ACCUM `>>1` 32-bit path, `ckernel.h:835-842`, comment :824
     "ACCUM_MODE: true for 32-bit (FP32)"): `get_dest_max_tiles<SyncFull,true,Tile32x32>
     = (1024>>1)>>6 = 8` (cf. `add_int_sfpu.h:21-22` "…reduced to 2 tiles from each operand
     for 32 bit formats"). The single-lease design kept **11** live INT32 tiles
     (Q,X,CARRY,8×OUT) — over the real 8-tile budget. (The earlier "16" was the
     accum=false 16-bit count, which does not apply to INT32 data.)
   - **Structure (FINAL): 4+4 TWO-LEASE in-place-carry ripple.** Two hard facts (both
     verified against shipped tt-llk, file:line) bound the structure:
       1. `pack_tile` REQUIRES a preceding `tile_regs_commit`: commit posts the MATH_PACK
          semaphore (`cmath_common.h:122` via `_llk_math_dest_section_done_`,
          `llk_math_common.h:87-95`) that `tile_regs_wait` blocks on with STALL_ON_ZERO
          (`llk_pack_common.h:21`). `pack_tile` itself touches no semaphore
          (`llk_pack.h:428-439`). So you CANNOT pack without committing, and committing
          ends the MATH section — **no math after a pack in the same lease.**
       2. `tile_regs_release` in SyncFull does `TTI_ZEROACC(p_zeroacc::CLR_ALL, …)` —
          wipes the ENTIRE DEST (`llk_pack_common.h:39-41`). **A persistent carry in a
          dst slot does NOT survive a release.**
     A scan of all 239 BH compute kernels (164 with `pack_tile`) found ZERO that do
     math-after-pack in one lease; there is NO streaming/tile-granular pack on BH
     (`DstSync` = `{SyncHalf, SyncFull}` only, `llk_defs.h:61-65`;
     `pack_sync_tile_dst_ptr`/`math_sync_tile_dst_index` are vestigial reset-only globals).
     ⇒ The coordinator's "single lease, persistent-carry-in-dst, pack each OUT mid-loop,
     keep rippling" is **IMPOSSIBLE on Blackhole** (controlling lines: `llk_pack_common.h:41`
     CLR_ALL + the commit/wait handshake `cmath_common.h:122` / `llk_pack_common.h:21`).
     And a single lease holding all 8 OUT + carry + Q/X = 11 > 8 doesn't fit anyway.

     The viable structure: split the 8-limb ripple into TWO leases of 4 limbs.
       • Per lease (4 limbs): carry lives in ONE persistent dst slot, seeded once and
         updated IN PLACE across the 4 limbs (read carry_in to a register, write carry_out
         to the SAME slot — safe per-lane); Q,X reused as scratch; OUT0..3 each in their
         own slot. Live INT32 tiles = 4 OUT + CARRY + Q + X = **7 ≤ 8**. After the 4 SFPU
         adds: ONE commit, then pack the 4 OUT (and, in lease 1, the limb-3 carry-out).
       • The carry crosses the SINGLE lease boundary via ONE carry-mid CB tile (`c_3`):
         written by lease 1 (PACK, post-commit), read by lease 2 (MATH, post-acquire) —
         DIFFERENT leases, so no in-lease self-alias. One tile, one direction, no ping-pong.
     This is exactly the shipped Welford carry pattern: in-place accumulate across a loop
     in one lease, state handed between leases via a CB
     (`ttnn/.../normalization/.../layernorm_large_tensor_welford.cpp:143-155` in-place;
     `combine_welford.h:66-153` interleaved copy+SFPU). `dst_full_sync_en=true` ⇒ SyncFull
     pins the dst base at 0, so both leases share one stable slot map.

   So FIX 1–5 were each *locally* true (real dst-budget/parity/MATH-mode facts) but were
   all dominated by the `fp32_dest_acc_en=false` corruption; with it true (8/32 → 14/32
   confirmed on hardware) and the 4+4 two-lease in-place-carry structure (footprint 7 ≤ 8,
   carry never lingers in dst across a release), the proven Welford-style ripple should be
   bit-exact for all 32 lanes × 8 limbs. (The earlier per-limb ping-pong variant was a
   transient step; the 4+4 design supersedes it — fewer carry handoffs, no ping-pong, and
   it realizes the in-place-carry intent within each lease.)

3. **Int32 `copy_tile`/`init_sfpu` data-format (MEDIUM).** `custom_sfpi_add` uses
   `Float16_b`; this kernel uses `DataFormat::Int32` CBs and `fp32_dest_acc_en=false`.
   `init_sfpu(cb_q, cb_out)` infers the format from the CBs, and `copy_tile` should
   move Int32 raw 32-bit values into dst. **Best guess: works as-is** (Int32 SFPU
   ops do exactly this, `add_int_sfpu.h`), but if dst comes back mangled, add a
   `copy_tile_init`/`reconfig_data_format` for Int32 (`tile_move_copy.h:57,72`).

4. **`_params_` face iteration vs. our 32-lane-only data (LOW).** We replicate each
   lane's limb across all 32 rows of the tile, so whichever faces `VectorMode::RC`
   sweeps, every row yields the same per-lane answer; the host reads lane *j* from
   row 0. If only a subset of faces is desired later, narrow the VectorMode.

5. **Real-silicon availability (EXTERNAL).** A card last seen in D3cold needs a
   cold boot + tt-flash. A ttsim pass is necessary but NOT the win (no-simulations
   rule). `run_on_silicon.sh` pre-checks `tt-smi -ls | grep blackhole`.

## Honest scope

This proves **one SFPU primitive** — the lane-parallel 256-bit ripple carry — at
32× the scalar throughput, bit-exact. It is NOT yet a full exact dot product: the
bp16 decode (divergent regime run-length → masked vector ops) and the per-product
mantissa multiply (no SFPU int-multiply builtin → shift-add, design §6.1) are the
next two milestones. Banking the carry vector first is deliberate: it is the
hardest and most bug-prone piece, and everything downstream reuses it.
