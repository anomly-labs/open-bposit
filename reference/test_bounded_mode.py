# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""test_bounded_mode.py — gates for the opt-in bounded-encode mode.

Run:  python3 reference/test_bounded_mode.py
The forge-oracle differential (gate 3) auto-skips if the mosyne tree is absent.
"""
import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bposit_ref as ref

FORGE = Path.home() / "development" / "mosyne" / "src" / "mosyne_forge" / "bp16_ref.py"

fails = 0

# gate 1: default mode is byte-for-byte the historical unbounded behavior
ref.set_bounded_encode(False)
assert ref.encode_bposit16(Fraction(2) ** 60) not in (0x7FFF,), "default must stay unbounded"
a60_unbounded = ref.encode_bposit16(Fraction(2) ** 60)
print("gate 1 PASS: default unbounded (2^60 encodes distinctly, code "
      f"0x{a60_unbounded:04x})")

# gate 2: golden-zone invariance — every value reachable within +/-48 encodes
# identically in both modes (bp16, sweep of all codes whose decode is in-envelope)
diff = 0
in_zone = 0
for p in range(1 << 16):
    d = ref.decode_bposit16(p)
    if d.is_special:
        continue
    v = ref.decoded_to_fraction(d)
    av = abs(v)
    if av == 0 or av > Fraction(2) ** 48 or av < Fraction(1, 2 ** 48):
        continue
    in_zone += 1
    ref.set_bounded_encode(False); e_u = ref.encode_bposit16(v)
    ref.set_bounded_encode(True);  e_b = ref.encode_bposit16(v)
    if e_u != e_b:
        diff += 1
ref.set_bounded_encode(False)
print(f"gate 2 {'PASS' if diff == 0 else 'FAIL'}: golden-zone invariance "
      f"({in_zone} in-envelope codes, {diff} mode-dependent)")
fails += diff != 0

# gate 3: bounded mode matches the forge/CUDA canonical oracle bit-for-bit
if FORGE.exists():
    spec = importlib.util.spec_from_file_location("forge_ref", FORGE)
    forge = importlib.util.module_from_spec(spec)
    sys.modules["forge_ref"] = forge
    spec.loader.exec_module(forge)
    ref.set_bounded_encode(True)
    mism = 0
    cases = [Fraction(2) ** e for e in range(-60, 61)]
    cases += [Fraction(3, 2) * Fraction(2) ** e for e in range(-52, 53)]
    cases += [-c for c in cases]
    for v in cases:
        if ref.encode_bposit16(v) != forge.encode_bposit16(v):
            mism += 1
    ref.set_bounded_encode(False)
    print(f"gate 3 {'PASS' if mism == 0 else 'FAIL'}: bounded == forge oracle "
          f"({len(cases)} values incl. out-of-envelope, {mism} mismatches)")
    fails += mism != 0
else:
    print("gate 3 SKIP: forge oracle not present")

print("ALL GATES PASS" if fails == 0 else f"{fails} GATE(S) FAILED")
sys.exit(1 if fails else 0)
