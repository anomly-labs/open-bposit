#!/usr/bin/env bash
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
# run_on_silicon.sh — run the -for-TRAINING result on a PHYSICAL Tenstorrent
# Blackhole baby core: EXACT 256-bit b-posit16 quire GRADIENT ACCUMULATION vs a
# genuine IEEE fp32 running accumulator. A large early contribution inflates the
# running sum to magnitude M, so each subsequent tiny per-step gradient < ULP(M) is
# dropped by fp32 (the swamping / gradient-underflow problem); the exact quire keeps
# every contribution and is bit-exact to the oracle (quire_gradient_cases.h). Sweeps
# the step count T (64,128,192,256) to show the fp32-lost gradient fraction grow.
#
# Mirrors bposit_quire_vs_float/run_on_silicon.sh: REAL silicon (no
# TT_METAL_SIMULATOR), ARCH_NAME=blackhole, SLOW dispatch. A ttsim pass is necessary
# but NOT the win (no-simulations project rule). Prereqs: tt-kmd + firmware + tt-smi
# healthy, tt-metal built for blackhole. The card needs a cold boot +
# tt-flash first — confirm healthy before running.
set -u
TTM="${TT_METAL_HOME:?set TT_METAL_HOME}"
ARCH="${ARCH_NAME:-blackhole}"
BIN="$TTM/build_Release/programming_examples/metal_example_bposit_quire_gradient"

# 0. (re)generate the golden from the oracle so the baked header is never stale. The
#    host ALSO self-checks the baked golden against the reused C headers at startup,
#    but regenerating here keeps the source of truth honest.
if command -v python3 >/dev/null 2>&1; then
  python3 "$TTM/tt_metal/programming_examples/bposit_quire_gradient/gen_gradient_golden.py" \
    >/dev/null 2>&1 || echo "WARN: golden regen failed (using committed quire_gradient_cases.h)"
fi

# 1. device must be visible
if command -v tt-smi >/dev/null 2>&1; then
  tt-smi -ls 2>/dev/null | grep -iq blackhole || { echo "FAIL: tt-smi does not see a Blackhole"; exit 3; }
else
  echo "WARN: tt-smi not installed — cannot pre-verify the device"; fi

# 2. build the example for the real arch if missing.
if [ ! -x "$BIN" ]; then
  grep -q bposit_quire_gradient "$TTM/tt_metal/programming_examples/CMakeLists.txt" || {
    echo "FAIL: bposit_quire_gradient not registered in programming_examples/CMakeLists.txt"; exit 2; }
  if [ ! -d "$TTM/build_Release" ]; then
    cmake -S "$TTM" -B "$TTM/build_Release" -DBUILD_PROGRAMMING_EXAMPLES=ON || {
      echo "FAIL: cmake configure"; exit 2; }
  fi
  ninja -C "$TTM/build_Release" metal_example_bposit_quire_gradient || { echo "FAIL: build"; exit 2; }
fi

# 3. run on the physical device (NO simulator), slow dispatch, gate on golden.
#    Optional arg: a single config index (0..3); default runs the whole T sweep.
ONLY="${1:-}"
export TT_METAL_HOME="$TTM" ARCH_NAME="$ARCH" TT_METAL_SLOW_DISPATCH_MODE=1
unset TT_METAL_SIMULATOR
cd "$TTM"
echo "=== exact 256-bit b-posit16 quire GRADIENT ACCUMULATION vs IEEE fp32 on physical $ARCH baby core ==="
echo "    running sum M = 2^14 = 16384 ; tau = 2^-14 (each tiny grad) << ULP(M)=2^-9 ; coarse surviving g0 = 1/128"
echo "    T= 64 : exact_g=0.01171875 | quire 0x2600 EXACT | fp32 0.0078125 -> loses 33.3% of g"
echo "    T=128 : exact_g=0.01562500 | quire 0x2800 EXACT | fp32 0.0078125 -> loses 50.0% of g"
echo "    T=192 : exact_g=0.01953125 | quire 0x2900 EXACT | fp32 0.0078125 -> loses 60.0% of g"
echo "    T=256 : exact_g=0.02343750 | quire 0x2a00 EXACT | fp32 0.0078125 -> loses 66.7% of g"
"$BIN" $ONLY 2>&1 | grep -iE "PASS|FAIL|^T=|quire|fp32|exact|gradient" | tail -40
