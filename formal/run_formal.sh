#!/usr/bin/env bash
# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# run_formal.sh — the three Yosys SAT proofs for the exactness-critical arithmetic of the
# 256-bit exact quire (CoNGA'26 paper, "Formal Verification"). Vector testbenches check golden
# vectors; these PROVE each block correct for EVERY possible input. A proof passes iff yosys
# reports "no model found: SUCCESS!"; a failure prints a concrete counterexample.
#
#   1. csa_compressor      s + c == s0 + c0 + s1 + c1 (mod 2^256) for the 4:2 carry-save cell —
#                          the whole reduction tree and the redundant-quire accumulate are
#                          compositions of this cell, so they are exact by composition.
#   2. oc_signed_reduction tc == oc + sign per lane, so sum(tc) = sum(oc) + popcount(signs).
#   3. quire_addend        exact_quire256 forms the two's-complement addend correctly from a
#                          sign-magnitude product (the defect two earlier quires shipped with).
#
# Requires yosys on PATH (any recent release; run from anywhere).  Exit code 0 iff all three pass.
set -u
cd "$(dirname "$0")/.." || exit 2   # -> repository root

pass=0; fail=0
# name : top module : sources : "prove LHS RHS"
prove () {
    local name="$1" top="$2" srcs="$3" lhs="$4" rhs="$5"
    local out
    out="$(yosys -p "read_verilog $srcs; hierarchy -top $top; proc; flatten; sat -prove $lhs $rhs -verify" 2>&1)"
    if echo "$out" | grep -q "no model found: SUCCESS"; then
        printf '  PROVEN  %-22s (%s == %s, all inputs)\n' "$name" "$lhs" "$rhs"; pass=$((pass+1))
    else
        printf '  FAIL    %s\n' "$name"; echo "$out" | grep -iE "model|counter|error" | sed 's/^/      /' | head; fail=$((fail+1))
    fi
}

echo "== exact-quire datapath: formal (SAT) proofs =="
yosys -V 2>/dev/null | head -1
prove csa_compressor      formal_csa_compressor "formal/bposit_csa_cs42.v formal/formal_csa_compressor.v" resolved reference
prove oc_signed_reduction formal_oc             "formal/formal_oc_signed_reduction.v"                     tc       oc_plus_sign
prove quire_addend        formal_quire_addend   "formal/formal_quire_addend.v"                            addend   reference

echo "== summary: $pass proven, $fail failed =="
[ "$fail" -eq 0 ]
