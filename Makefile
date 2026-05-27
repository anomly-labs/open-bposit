# open-bposit — top-level convenience targets.
# Unifies the two entry points the README documents:
#   make smoke   — fast end-to-end check (rung sweep + mixed-precision + quire
#                  determinism); needs python3 + numpy (see examples/requirements.txt).
#   make verify  — exhaustive RTL conformance cosim vs the reference oracle;
#                  needs iverilog + python3 (delegates to targets/coreet).
#   make all     — smoke then verify.
# No build artifacts are produced at the top level; both targets shell out.

PYTHON ?= python3

.PHONY: all smoke verify clean help

help:
	@echo "open-bposit targets:"
	@echo "  make smoke   fast end-to-end check (python3 + numpy)"
	@echo "  make verify  exhaustive RTL conformance cosim (iverilog + python3)"
	@echo "  make all     smoke then verify"
	@echo "  make clean   remove cosim scratch (delegates to targets/coreet)"

smoke:
	PYTHON=$(PYTHON) bash examples/smoke_test.sh

verify:
	$(MAKE) -C targets/coreet verify-full

all: smoke verify

clean:
	$(MAKE) -C targets/coreet clean
