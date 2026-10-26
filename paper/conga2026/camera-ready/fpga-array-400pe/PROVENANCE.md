<!-- Copyright (c) 2026 Anomly, Inc. SPDX-License-Identifier: Apache-2.0 -->
# §7.2 full-array utilization and power of the 400-PE U200 build

**What the paper says (§7.2, camera-ready 2026-10-24):** the 400-PE kernel
occupies 406,048 LUTs (34.4 % of the XCU200), 366,229 flip-flops, 22,706
CARRY8 and 803 DSP48E2 slices and no block RAM, spread over the three SLRs
(195,753 / 70,001 / 140,294 LUTs); Vivado's vectorless post-route estimate puts
the dynamic power of the user partition holding it at 10.9 W (confidence
"Low"). The reports of the 625-PE build were not retained.

**Where the numbers come from.** The Vitis 2022.2 link run of the paper's
400-PE variant (`anomly_gemm_batch20s3`, 20×20 PEs, mid-multiplier pipeline
stage, 140 MHz requested; routed 2026-07-08, worst negative slack −0.103 ns as
§7.2 states) was retained on the build machine together with its routed
checkpoint. On 2026-10-24 Vivado 2022.2 opened that checkpoint and ran
`report.tcl` (this directory):

| file | command | what it reports |
|---|---|---|
| `util_kernel.rpt` | `report_utilization -cells level0_i/ulp/anomly_gemm_batch20s3_1` | the kernel alone: the §7.2 LUT / FF / CARRY8 / DSP / BRAM figures and the per-SLR split |
| `util_ulp.rpt` | `report_utilization -cells level0_i/ulp` | the user partition (kernel + DFX interconnect): 432,984 LUTs, 411,943 FFs, 37 BRAM tiles, 803 DSPs |
| `util_full.rpt` | `report_utilization` | the whole device including the static shell: 576,695 LUTs (48.8 %), 608,731 FFs, 810 DSPs, 242.5 BRAM tiles, 20 URAMs |
| `util_hier.rpt` | `report_utilization -hierarchical -hierarchical_depth 4` | the same, by hierarchy |
| `power.rpt` | `report_power` | 24.469 W total: 21.558 W dynamic, 2.912 W static; overall confidence **Low** (no switching activity supplied; implementation state and clocks "High", internal-node activity "Medium", I/O "Low") |
| `power_hier.rpt` | `report_power -hierarchical_depth 3` | the user partition `ulp` at 10.912 W dynamic, which is the §7.2 figure |

The checkpoint and the Vitis run directory (580 MB) are not published; the
array RTL is not published either (as for §7.3, its exactness-critical cells are
the proven `formal/` blocks). The 625-PE 90 MHz build that §7.2 reports at
25.95 / 47.80 GMAC/s was built in June 2026 and its implementation reports did
not survive a disk cleanup on 2026-10-01; a 625-PE stage-3 run at 100 MHz
(timing-failed, not in the paper) is retained and routes at 793,596 LUTs and
1,260 DSPs for context only.
