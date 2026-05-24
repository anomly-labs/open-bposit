# Synthesis & timing — real silicon numbers (sky130, cloud, free)

The cosim suite (`make verify-full`) proves the RTL is *functionally* bit-exact
against the reference oracle. This directory closes the other half: getting real
**area, gate count, and Fmax** by pushing the W8A8 cell (`bposit8_dot`) through an
open synthesis + place-and-route flow — with no local install, using the free
[VSD cloud labs](https://github.com/vsdip) that run in a GitHub Codespace.

> **Read this caveat first (it's the honest framing).** These numbers use the
> **sky130 130 nm** open PDK, which is **not** CORE-ET's process node. So the
> absolute area/Fmax are an **illustrative ballpark, not a CORE-ET spec** — what
> they establish is that the block **synthesizes cleanly to real standard cells**
> and roughly how big and how fast it is. That's the credibility step ("here's a
> synthesizable block with real numbers"), not a tape-out datasheet.

`bposit8_dot` = `bposit8_qmac` (exact bp8→quire256 MAC) + `bposit_encode`
(round-once quire→bposit32). Clock port `clk_i`, reset `rst_ni`.

## Quick: area + gate count (yosys only — vsd-rtl lab)

Open the [`vsdip/vsd-rtl`](https://github.com/vsdip/vsd-rtl) Codespace (yosys +
sky130 preinstalled), then:

```bash
cd targets/coreet/synth
./run_synth.sh          # finds the sky130hd liberty, runs synth_check.ys
cat synth_stat.txt      # cell count + area (µm²)
```

`run_synth.sh` auto-locates the `sky130_fd_sc_hd` typical-corner liberty (or set
`PDK_ROOT` / `LIB=<path>.lib`). Output: `synth_stat.txt` (area/cells), `synth.log`
(full run). `abc` is run with a ~10 ns target, so the log also gives a first
critical-path estimate.

## Full: place-and-route + STA Fmax (vsd-openlane lab)

Open the [`vsdip/vsd-openlane`](https://github.com/vsdip/vsd-openlane) Codespace
(OpenLane + sky130). Drop this design in and run the flow:

```bash
# inside the OpenLane container
mkdir -p designs/bposit8_dot/src
cp targets/coreet/bposit8_qmac.sv targets/coreet/bposit_encode.sv \
   targets/coreet/bposit8_dot.sv  designs/bposit8_dot/src/
cp targets/coreet/synth/config.json designs/bposit8_dot/
./flow.tcl -design bposit8_dot
```

The final `runs/*/reports/` has the post-route **STA** (slack → Fmax), area, and
power. To find Fmax: lower `CLOCK_PERIOD` in `config.json` until STA reports a
setup violation; the last passing period is your Fmax estimate.

**If yosys rejects the SystemVerilog** (some OpenLane builds parse plain Verilog
only): enable the slang/surelog plugin (`SYNTH_DEFINES`/`USE_SLANG` in newer
OpenLane), or just use the pure-yosys `synth_check.ys` above for the area/gate
estimate — `read_verilog -sv` there handles these files directly.

## Files

| file | what |
|------|------|
| `synth_check.ys` | pure-yosys synth + sky130 map + area/gate `stat` (no P&R) |
| `run_synth.sh`   | locates the sky130hd liberty and runs `synth_check.ys` |
| `config.json`    | OpenLane (classic, sky130hd) config for full P&R + STA |
