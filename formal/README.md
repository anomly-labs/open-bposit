# Formal (SAT) proofs of the exactness-critical arithmetic

Three Yosys SAT proofs covering the arithmetic that an exact 256-bit quire cannot get
wrong (CoNGA'26 paper, *Formal Verification*). Golden-vector testbenches check the
vectors they were given; each of these proves its block for **every possible input**.

| proof | property proven | why it matters |
|---|---|---|
| `formal_csa_compressor.v` | `s + c == s0 + c0 + s1 + c1 (mod 2^256)` for the 4:2 carry-save cell `bposit_csa_cs42.v` | the multi-lane reduction tree and the redundant-quire accumulate are compositions of this one cell, so they are exact by composition |
| `formal_oc_signed_reduction.v` | `tc == oc + sign` per lane — the two's-complement of a sign-magnitude product equals its one's-complement plus its sign bit | lets the sign correction be folded into a single popcount instead of a per-lane +1 chain, without changing the value |
| `formal_quire_addend.v` | the exact-quire datapath forms the two's-complement addend correctly from a sign-magnitude product | the defect two earlier quire revisions shipped with; caught by this proof |

The proof harnesses are self-contained: the CSA cell is extracted verbatim into
`bposit_csa_cs42.v`, and the other two carry the datapath fragment under proof inline.
The full vector-MAC engine these cells come from is not published.

## Run

```bash
./formal/run_formal.sh        # needs yosys (0.33 or later) on PATH; ~10 s total
```

A proof passes iff Yosys reports `no model found: SUCCESS!`; a failure prints a concrete
counterexample. Transcript from this repository's layout (Yosys 0.33, 2026-09-03):

```
== exact-quire datapath: formal (SAT) proofs ==
Yosys 0.33 (git sha1 2584903a060)
  PROVEN  csa_compressor         (resolved == reference, all inputs)
  PROVEN  oc_signed_reduction    (tc == oc_plus_sign, all inputs)
  PROVEN  quire_addend           (addend == reference, all inputs)
== summary: 3 proven, 0 failed ==
```

Yosys without root: `apt-get download yosys && dpkg-deb -x yosys_*.deb ./yosys-root`
then `PATH=./yosys-root/usr/bin:$PATH ./formal/run_formal.sh`.
