#!/usr/bin/env bash
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
# run_on_silicon.sh — run the LANE-PARALLEL exact 256-bit q256_add on the SFPU of a
# PHYSICAL Tenstorrent Blackhole and gate bit-exact vs the generated golden
# (sfpu_add_golden_cases.h, itself re-derived at startup from the real scalar
# q256_add). Mirrors bposit_quire_reduce/run_on_silicon.sh: REAL silicon (no
# TT_METAL_SIMULATOR), ARCH_NAME=blackhole, SLOW dispatch. A ttsim pass is
# necessary but NOT the win (no-simulations project rule).
#
# Prereqs: tt-kmd + firmware + tt-smi healthy, tt-metal built for blackhole. The
# A power-cycled card needs a cold boot + tt-flash first — confirm healthy first.
#
# This script does NOT build/run by default in CI; it is the on-silicon entry point.
set -u
TTM="${TT_METAL_HOME:-~/development/tt-metal}"
ARCH="${ARCH_NAME:-blackhole}"
BIN="$TTM/build_Release/programming_examples/metal_example_bposit_quire_sfpu_add"

# 0. device must be visible
if command -v tt-smi >/dev/null 2>&1; then
  tt-smi -ls 2>/dev/null | grep -iq blackhole || { echo "FAIL: tt-smi does not see a Blackhole"; exit 3; }
else
  echo "WARN: tt-smi not installed — cannot pre-verify the device"; fi

# 1. (re)generate the golden header from the scalar reference (deterministic).
if command -v python3 >/dev/null 2>&1; then
  python3 "$TTM/tt_metal/programming_examples/bposit_quire_sfpu_add/gen_sfpu_add_golden.py" \
    "$TTM/tt_metal/programming_examples/bposit_quire_sfpu_add/sfpu_add_golden_cases.h" \
    || { echo "FAIL: golden generation"; exit 2; }
fi

# 2. build the example for the real arch if missing.
if [ ! -x "$BIN" ]; then
  grep -q bposit_quire_sfpu_add "$TTM/tt_metal/programming_examples/CMakeLists.txt" || {
    echo "FAIL: bposit_quire_sfpu_add not registered in programming_examples/CMakeLists.txt"; exit 2; }
  if [ ! -d "$TTM/build_Release" ]; then
    cmake -S "$TTM" -B "$TTM/build_Release" -DBUILD_PROGRAMMING_EXAMPLES=ON || {
      echo "FAIL: cmake configure (see docs/blackhole + on_silicon_bringup_checklist.md §4)"; exit 2; }
  fi
  ninja -C "$TTM/build_Release" metal_example_bposit_quire_sfpu_add || { echo "FAIL: build"; exit 2; }
fi

# 3. run on the physical device (NO simulator), slow dispatch, gate on golden.
export TT_METAL_HOME="$TTM" ARCH_NAME="$ARCH" TT_METAL_SLOW_DISPATCH_MODE=1
unset TT_METAL_SIMULATOR
cd "$TTM"
echo "=== lane-parallel exact q256_add on the SFPU of physical $ARCH (32 lanes, 8 limbs) ==="
"$BIN" 2>&1 | grep -iE "PASS|FAIL|lane|limb|bit-exact" | tail -40
