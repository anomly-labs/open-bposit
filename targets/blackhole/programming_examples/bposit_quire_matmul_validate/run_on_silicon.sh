#!/usr/bin/env bash
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
# run_on_silicon.sh — RIGOROUS FULL-VALIDATION of the EXACT 256-bit b-posit16
# quire MATMUL on a PHYSICAL Tenstorrent Blackhole: SEEDS=8 distinct RANDOM
# matmuls C[16,16]=A[16,16].B[16,16], gating EVERY output of EVERY seed (2048
# outputs, 32768 exact products) bit-exact vs the oracle golden
# (quire_matmul_validate_golden_cases.h, from gen_matmul_validate_golden.py),
# plus a full 256-bit quire byte-gate per seed. NO sampling, NO tolerance.
#
# Mirrors bposit_quire_matmul_multicore/run_on_silicon.sh: REAL silicon (no
# TT_METAL_SIMULATOR), ARCH_NAME=blackhole, SLOW dispatch. A ttsim pass is
# necessary but NOT the win (no-simulations project rule). Prereqs: tt-kmd +
# firmware + tt-smi healthy, tt-metal built for blackhole. The card
# needs a cold boot + tt-flash first — confirm healthy before running.
set -u
TTM="${TT_METAL_HOME:?set TT_METAL_HOME}"
ARCH="${ARCH_NAME:-blackhole}"
BIN="$TTM/build_Release/programming_examples/metal_example_bposit_quire_matmul_validate"

# 0. device must be visible
if command -v tt-smi >/dev/null 2>&1; then
  tt-smi -ls 2>/dev/null | grep -iq blackhole || { echo "FAIL: tt-smi does not see a Blackhole"; exit 3; }
else
  echo "WARN: tt-smi not installed — cannot pre-verify the device"; fi

# 1. (re)generate the golden header from the canonical oracle, then build.
#    Regen is the slow part (pure-Python Fraction oracle over 32768 products,
#    ~minutes); it guarantees the baked golden matches the oracle. The host
#    program ALSO self-checks the baked golden against the reused C headers at
#    startup for ALL seeds before any device dispatch.
if command -v python3 >/dev/null 2>&1; then
  ( cd "$TTM/tt_metal/programming_examples/bposit_quire_matmul_validate" && \
    python3 gen_matmul_validate_golden.py > quire_matmul_validate_golden_cases.h ) || {
      echo "FAIL: golden regen (oracle bposit16_reference.py)"; exit 2; }
fi

# 2. build the example for the real arch if missing.
if [ ! -x "$BIN" ]; then
  grep -q bposit_quire_matmul_validate "$TTM/tt_metal/programming_examples/CMakeLists.txt" || {
    echo "FAIL: bposit_quire_matmul_validate not registered in programming_examples/CMakeLists.txt"; exit 2; }
  if [ ! -d "$TTM/build_Release" ]; then
    cmake -S "$TTM" -B "$TTM/build_Release" -DBUILD_PROGRAMMING_EXAMPLES=ON || {
      echo "FAIL: cmake configure (see docs/blackhole + on_silicon_bringup_checklist.md §4)"; exit 2; }
  fi
  ninja -C "$TTM/build_Release" metal_example_bposit_quire_matmul_validate || { echo "FAIL: build"; exit 2; }
fi

# 3. run on the physical device (NO simulator), slow dispatch, gate on golden.
export TT_METAL_HOME="$TTM" ARCH_NAME="$ARCH" TT_METAL_SLOW_DISPATCH_MODE=1
unset TT_METAL_SIMULATOR
cd "$TTM"
echo "=== FULL-VALIDATION exact 256-bit b-posit16 quire MATMUL on physical $ARCH ==="
echo "    8 random seeds x C[16,16]=A[16,16].B[16,16] = 2048 outputs, 32768 exact products"
echo "    every output gated bit-exact vs oracle + one full 256-bit quire byte-gate/seed"
"$BIN" 2>&1 | grep -iE "VALIDATE|PASS|FAIL|bit-exact|quire|seeds|cores used|rows/s" | tail -60
