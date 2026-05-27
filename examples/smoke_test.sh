#!/usr/bin/env bash
# smoke_test.sh — one-command check that open-bposit works end to end.
# Runs the rung sweep + mixed-precision demo (synthetic fallback when no HF model
# is cached) and asserts the reproducibility invariants hold, then a direct
# quire round-trip determinism check. Requires numpy (see examples/requirements.txt).
#
#   bash examples/smoke_test.sh        # uses python3
#   PYTHON=.venv/bin/python bash examples/smoke_test.sh
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
fail=0
note() { printf '%s\n' "$1"; }

note "[1/3] rung sweep (bp4 / aip5 / bp8)…"
out=$("$PY" rung_sweep.py 2>&1) || { echo "$out"; note "FAIL: rung_sweep.py exited nonzero"; fail=1; }
echo "$out" | grep -q "reproducible:✓" || { note "FAIL: rung_sweep missing 'reproducible:✓'"; fail=1; }
echo "$out" | grep -qE "^bp8 " || { note "FAIL: rung_sweep missing bp8 row"; fail=1; }

note "[2/3] mixed-precision (bp8 + aip5 in one exact quire)…"
out=$("$PY" mixed_precision_demo.py 2>&1) || { echo "$out"; note "FAIL: mixed_precision_demo.py exited nonzero"; fail=1; }
echo "$out" | grep -q "bit-reproducible across runs: YES" || { note "FAIL: mixed-precision not bit-reproducible"; fail=1; }

note "[3/3] quire matmul determinism (reference)…"
"$PY" - <<'PYEOF' || fail=1
import sys
from pathlib import Path
sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "reference") if "__file__" in dir() else Path("../reference").resolve()))
sys.path.insert(0, str(Path("../reference").resolve()))
import numpy as np
import bposit_quantize as q
W = np.random.default_rng(0).standard_normal((16, 32)) * 0.05
X = np.random.default_rng(1).standard_normal((4, 32)) * 0.5
a = q.quantize_w8a8(W, X)
b = q.quantize_w8a8(W, X)
assert np.array_equal(a, b), "quire W8A8 matmul is NOT deterministic"
print("  quire matmul deterministic ✓")
PYEOF
[ $? -eq 0 ] || fail=1

if [ "$fail" -eq 0 ]; then
  note ""; note "✅ open-bposit smoke test PASSED"
else
  note ""; note "❌ open-bposit smoke test FAILED"; exit 1
fi
