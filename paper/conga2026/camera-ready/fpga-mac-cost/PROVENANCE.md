<!-- Copyright (c) 2026 Anomly, Inc. SPDX-License-Identifier: Apache-2.0 -->
# §7.3 exact vs correctly-rounded b-posit16 MAC on XCU200 — report provenance

The paper's §7.3 states, post-route on `xcu200-fsgd2104-2-e` (Vivado 2022.2):
exact MAC 4,179 LUTs, 294 flip-flops, 216 CARRY8, 1 DSP, 0 BRAM at 341.6 MHz and
16 mW dynamic; rounded MAC 9,673 LUTs, 48 flip-flops, 782 CARRY8, 1 DSP, 0 BRAM at
32.6 MHz and 38 mW dynamic; an fp16 fused MAC on the same device at 7 mW.

The original July 2026 post-route reports were lost in a workstation cleanup on
2026-10-01. The synthesis was re-run from source on 2026-10-17 with the same tool
version, out-of-context, `opt_design`/`place_design`/`phys_opt_design`/`route_design`,
each design constrained at the period the paper reports (2.927 ns exact, 30.67 ns
rounded). The MAC RTL itself is not published (see the repository README on scope);
the Vivado reports are, with only the two header lines naming the build host removed.

| file | design | RTL revision |
|---|---|---|
| `bposit16_mac_*.rpt`, `bposit16_mac_summary.txt` | exact MAC | the RTL as of the paper's submission (2026-09-11), before a later timing respin |
| `bposit16_mac_rounded_*.rpt`, `_summary.txt` | rounded MAC | unchanged since submission |

What the reports show against the paper:

| quantity | paper | regenerated (submission-era RTL) |
|---|---|---|
| exact LUTs / FF / CARRY8 / DSP / BRAM | 4,179 / 294 / 216 / 1 / 0 | 4,184 / 294 / 216 / 1 / 0 |
| rounded LUTs / FF / CARRY8 / DSP / BRAM | 9,673 / 48 / 782 / 1 / 0 | 9,458 / 48 / 782 / 1 / 0 |
| exact Fmax | 341.6 MHz | closes the 2.927 ns constraint with +0.574 ns slack (425.0 MHz achieved) |
| rounded Fmax | 32.6 MHz | 32.9 MHz |

So the area and carry-chain figures reproduce to within 5 LUTs (0.12 %) on the exact
arm and 2.2 % on the rounded arm. The paper's 341.6 MHz is 1/2.927 ns, the constraint
the design was proven to meet; the achieved critical path is shorter, so the stated
10.5× frequency ratio is conservative (12.9× achieved).

The power pair (16 mW / 38 mW, and the 7 mW fp16 reference) is the paper's August
2026 iso-frequency estimate at Vivado's "Medium" confidence; it was not re-run here.
The `*_power.rpt` files in this directory clock each design at its own Fmax and are
therefore not an iso-frequency comparison; they are included for completeness and
must not be read against the paper's power figures.

The current RTL (after a 2026-09-16 Kogge-Stone respin of the 256-bit accumulate)
synthesises to 5,840 LUTs and 184 CARRY8 at 374.8 MHz: it trades carry chains for
LUTs to buy speed, as a prefix adder does. The paper's numbers are the pre-respin
design and should be read as such.
