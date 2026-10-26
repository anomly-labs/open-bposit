open_checkpoint /home/u200build_server/fwdkbuild/chip-design/fpga/u200/runs/vpp_b20s3140pr_tmp/anomly_gemm_batch20s3/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed.dcp
set K level0_i/ulp/anomly_gemm_batch20s3_1
puts "KERNEL_CELL: [get_cells $K]"
catch {report_utilization -file /home/u200build_server/conga_util_1024/util_full.rpt} m; puts "util_full: $m"
catch {report_utilization -cells [get_cells $K] -file /home/u200build_server/conga_util_1024/util_kernel.rpt} m; puts "util_kernel: $m"
catch {report_utilization -cells [get_cells level0_i/ulp] -file /home/u200build_server/conga_util_1024/util_ulp.rpt} m; puts "util_ulp: $m"
catch {report_utilization -hierarchical -hierarchical_depth 4 -file /home/u200build_server/conga_util_1024/util_hier.rpt} m; puts "util_hier: $m"
catch {report_timing_summary -no_detailed_paths -max_paths 1 -file /home/u200build_server/conga_util_1024/timing.rpt} m; puts "timing: $m"
catch {report_power -file /home/u200build_server/conga_util_1024/power.rpt} m; puts "power: $m"
catch {report_power -hierarchical_depth 3 -file /home/u200build_server/conga_util_1024/power_hier.rpt} m; puts "power_hier: $m"
puts "REPORTS_DONE"
