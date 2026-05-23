# CORE-ET b-posit RTL — reproducible verification (iverilog). Vectors come from
# the shared reference (../../reference); they land here and the TBs consume them.
REF = ../../reference
IV  = iverilog -g2012
RTL = bposit_qmac.sv bposit_encode.sv bposit_dot.sv

vectors:
	python3 $(REF)/gen_rtl_vectors.py
	python3 $(REF)/gen_encoder_vectors.py
	python3 $(REF)/gen_alu_tables.py
	python3 $(REF)/gen_bp8_exhaustive.py
qmac: vectors ; $(IV) -o /tmp/qmac.vvp bposit_qmac.sv bposit_qmac_tb.sv && vvp /tmp/qmac.vvp
dot: vectors ; $(IV) -o /tmp/dot.vvp $(RTL) bposit_dot_tb.sv && vvp /tmp/dot.vvp
dot8: vectors ; $(IV) -o /tmp/dot8.vvp bposit8_qmac.sv bposit_encode.sv bposit8_dot.sv bposit8_dot_tb.sv && vvp /tmp/dot8.vvp
encode: vectors ; $(IV) -o /tmp/enc.vvp bposit_encode.sv bposit_encode_tb.sv && vvp /tmp/enc.vvp
alu: vectors ; $(IV) -o /tmp/alu.vvp bposit_lowp_alu.sv bposit_lowp_alu_tb.sv && vvp /tmp/alu.vvp
bp8x: vectors ; $(IV) -o /tmp/bp8x.vvp bposit8_qmac.sv bposit_encode.sv bposit8_dot.sv bposit8_exhaustive_tb.sv && vvp /tmp/bp8x.vvp
lint: ; $(IV) -Wall -Wno-timescale -o /dev/null $(RTL) bposit_dot_tb.sv && echo "lint clean"
verify: qmac dot dot8 encode alu lint
verify-full: verify bp8x
clean: ; rm -f *.hex *_dims.svh dims.svh *_alu.svh *.vvp ; rm -rf __pycache__
