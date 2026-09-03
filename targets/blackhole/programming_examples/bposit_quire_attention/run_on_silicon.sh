#!/usr/bin/env bash
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
# run_on_silicon.sh — run the attention-under-massive-activation-outliers
# result on a PHYSICAL Tenstorrent Blackhole baby core: the EXACT 256-bit
# b-posit16 quire dot vs a genuine IEEE fp32 naive dot, over one query and 3 keys
# (head_dim D=32). The quire dot must be bit-exact to the oracle ground truth
# (attention_golden_cases.h) while the fp32 dot loses the signal under the massive
# activation outliers and FLIPS the softmax argmax — on real silicon.
#
# Mirrors bposit_quire_vs_float/run_on_silicon.sh: REAL silicon (no
# TT_METAL_SIMULATOR), ARCH_NAME=blackhole, SLOW dispatch. A ttsim pass is
# necessary but NOT the win (no-simulations project rule). Prereqs: tt-kmd +
# firmware + tt-smi healthy, tt-metal built for blackhole. The card
# needs a cold boot + tt-flash first — confirm healthy before running.
set -u
TTM="${TT_METAL_HOME:?set TT_METAL_HOME}"
ARCH="${ARCH_NAME:-blackhole}"
BIN="$TTM/build_Release/programming_examples/metal_example_bposit_quire_attention"

# 0. (re)generate the golden from the oracle so the baked header is never stale.
#    The host ALSO self-checks the baked golden against the reused C headers at
#    startup, but regenerating here keeps the source of truth honest.
if command -v python3 >/dev/null 2>&1; then
  python3 "$TTM/tt_metal/programming_examples/bposit_quire_attention/gen_attention_golden.py" \
    >/dev/null 2>&1 || echo "WARN: golden regen failed (using committed attention_golden_cases.h)"
fi

# 1. device must be visible
if command -v tt-smi >/dev/null 2>&1; then
  tt-smi -ls 2>/dev/null | grep -iq blackhole || { echo "FAIL: tt-smi does not see a Blackhole"; exit 3; }
else
  echo "WARN: tt-smi not installed — cannot pre-verify the device"; fi

# 2. build the example for the real arch if missing.
if [ ! -x "$BIN" ]; then
  grep -q bposit_quire_attention "$TTM/tt_metal/programming_examples/CMakeLists.txt" || {
    echo "FAIL: bposit_quire_attention not registered in programming_examples/CMakeLists.txt"; exit 2; }
  if [ ! -d "$TTM/build_Release" ]; then
    cmake -S "$TTM" -B "$TTM/build_Release" -DBUILD_PROGRAMMING_EXAMPLES=ON || {
      echo "FAIL: cmake configure"; exit 2; }
  fi
  ninja -C "$TTM/build_Release" metal_example_bposit_quire_attention || { echo "FAIL: build"; exit 2; }
fi

# 3. run on the physical device (NO simulator), slow dispatch, gate on golden.
export TT_METAL_HOME="$TTM" ARCH_NAME="$ARCH" TT_METAL_SLOW_DISPATCH_MODE=1
unset TT_METAL_SIMULATOR
cd "$TTM"
echo "=== exact 256-bit b-posit16 quire attention vs IEEE fp32 on physical $ARCH baby core ==="
echo "    q.k_j with massive-activation outliers (q*k = +/-2^33) that cancel exactly in the quire"
echo "    expected: exact/quire scores [3.6875, 2.375, 2.625] -> argmax key0 (true winner)"
echo "              fp32 naive scores  [0.0625, 0.25,  1.0   ] -> argmax key2 (WRONG: signal lost)"
"$BIN" 2>&1 | grep -iE "PASS|FAIL|^s_|softmax|argmax|quire|fp32|exact|FLIP|divergence|cause" | tail -40
