#!/usr/bin/env bash
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
# run_on_silicon.sh — run a REAL information-theoretic ML primitive (Layer-1
# cross-entropy H(p,q) = -Sum_i p_i*log2(q_i), the standard classification training
# loss) on a PHYSICAL Tenstorrent Blackhole baby core and validate bit-exact vs the
# oracle golden (golden/cross_entropy.json).
#
#   Each term p_i*log2(q_i) is an EXACT bp16 product, accumulated in the EXACT
#   256-bit Kulisch quire and negated; the single rounding is the final
#   quire->bposit32 readout (H = 2.625 bits -> 0x45400000). HONEST SPLIT: the
#   per-element log2(q_i) is HOST-precomputed (bposit16_log2; the 65536-entry LUT
#   does not fit the baby core's local DATA region), while the load-bearing
#   exact-quire SUM runs 100% on-device.
#
# Mirrors bposit_quire_causet/run_on_silicon.sh: REAL silicon (no
# TT_METAL_SIMULATOR), ARCH_NAME=blackhole, SLOW dispatch. A ttsim pass is
# necessary but NOT the win (no-simulations project rule). Prereqs: tt-kmd +
# firmware + tt-smi healthy, tt-metal built for blackhole. The card on
# A card that has been power-cycled needs a cold boot + tt-flash first — confirm healthy before running.
set -u
TTM="${TT_METAL_HOME:?set TT_METAL_HOME}"
ARCH="${ARCH_NAME:-blackhole}"
BIN="$TTM/build_Release/programming_examples/metal_example_bposit_quire_cross_entropy"

# 0. device must be visible
if command -v tt-smi >/dev/null 2>&1; then
  tt-smi -ls 2>/dev/null | grep -iq blackhole || { echo "FAIL: tt-smi does not see a Blackhole"; exit 3; }
else
  echo "WARN: tt-smi not installed — cannot pre-verify the device"; fi

# 0b. regenerate the golden header from the oracle + golden/cross_entropy.json if
#     the generator is newer (keeps the baked header honest; the host re-verifies anyway).
GEN="$TTM/tt_metal/programming_examples/bposit_quire_cross_entropy/gen_cross_entropy_golden.py"
HDR="$TTM/tt_metal/programming_examples/bposit_quire_cross_entropy/cross_entropy_golden_cases.h"
if [ -f "$GEN" ] && { [ ! -f "$HDR" ] || [ "$GEN" -nt "$HDR" ]; }; then
  echo "=== regenerating cross-entropy golden from oracle + golden/cross_entropy.json ==="
  python3 "$GEN" > "$HDR" || { echo "FAIL: golden regen"; exit 2; }
fi

# 1. build the example for the real arch if missing.
# The source lives in-tree at tt_metal/programming_examples/bposit_quire_cross_entropy/
# and is registered in that dir's parent CMakeLists.txt, so a plain ninja build works.
if [ ! -x "$BIN" ]; then
  grep -q bposit_quire_cross_entropy "$TTM/tt_metal/programming_examples/CMakeLists.txt" || {
    echo "FAIL: bposit_quire_cross_entropy not registered in programming_examples/CMakeLists.txt"; exit 2; }
  if [ ! -d "$TTM/build_Release" ]; then
    cmake -S "$TTM" -B "$TTM/build_Release" -DBUILD_PROGRAMMING_EXAMPLES=ON || {
      echo "FAIL: cmake configure (see docs/blackhole + on_silicon_bringup_checklist.md §4)"; exit 2; }
  fi
  ninja -C "$TTM/build_Release" metal_example_bposit_quire_cross_entropy || { echo "FAIL: build"; exit 2; }
fi

# 2. run on the physical device (NO simulator), slow dispatch, gate on golden.
#    The binary takes a case selector: "cross_entropy" | "all" (default cross_entropy).
MODE="${1:-cross_entropy}"
export TT_METAL_HOME="$TTM" ARCH_NAME="$ARCH" TT_METAL_SLOW_DISPATCH_MODE=1
unset TT_METAL_SIMULATOR
cd "$TTM"
echo "=== cross-entropy H(p,q) on physical $ARCH baby core (mode: $MODE) ==="
echo "    N=4 classes, H(p,q) = 2.625 bits expected (golden/cross_entropy.json)"
echo "    host-precomputed log2(q): 0xba00 0xba00 0xbc00 0xc000  (q = 1/8,1/8,1/4,1/2)"
echo "    golden negated quire: 0000000000000000000000a00200000000000000000000000000000000000000  (bp32 0x45400000)"
"$BIN" "$MODE" 2>&1 | grep -iE "PASS|FAIL|match golden|cross_entropy|quire|H\(p,q\)|meaning|loss|bp32|log2|split" | tail -40
