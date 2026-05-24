#!/usr/bin/env bash
# run_synth.sh — one-command sky130 synthesis + area estimate for bposit8_dot.
# Designed to run as-is inside the VSD cloud labs (vsd-rtl or vsd-openlane
# GitHub Codespace), where yosys and the sky130 PDK are preinstalled.
#
#   cd targets/coreet/synth && ./run_synth.sh
#
# It locates the sky130 high-density typical-corner liberty, runs synth_check.ys,
# and leaves the area/cell report in synth_stat.txt.
set -euo pipefail
cd "$(dirname "$0")"

# Common locations across the VSD labs / OpenLane volumes.
_HD="libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
CANDIDATES=(
  "${LIB:-}"
  "${PDK_ROOT:-}/sky130A/$_HD"
  "/home/vscode/.ciel/sky130A/$_HD"     # vsd-openlane Codespace (ciel)
  "$HOME/.ciel/sky130A/$_HD"
  "$HOME/.volare/sky130A/$_HD"          # older volare layout
  "/usr/local/share/pdk/sky130A/$_HD"
)
LIB=""
for c in "${CANDIDATES[@]}"; do
  [ -n "$c" ] && [ -f "$c" ] && { LIB="$c"; break; }
done
if [ -z "$LIB" ]; then
  echo "error: could not find sky130_fd_sc_hd liberty." >&2
  echo "       set PDK_ROOT (e.g. export PDK_ROOT=\$HOME/.volare) or LIB=<path>.lib" >&2
  exit 1
fi
echo "[run_synth] using liberty: $LIB"

yosys -ql synth.log -DLIB="$LIB" synth_check.ys
echo
echo "==================== area / cell report ===================="
cat synth_stat.txt
echo "============================================================"
echo "full report: synth.log   |   for P&R + STA (Fmax): see README.md (OpenLane)"
