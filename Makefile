# Top-level convenience targets. The paper has its own Makefile in paper/.
.PHONY: help sync oct-deps

help:
	@echo "Targets:"
	@echo "  sync      - uv sync (install the main project)"
	@echo "  oct-deps  - install OpenCharacterTraining deps (character, openrlhf)"
	@echo "              that uv sync cannot; run after 'make sync' if you need"
	@echo "              the OCT training pipeline. See scripts/setup/install_oct_deps.sh."

sync:
	uv sync

# character + openrlhf use SSH git submodules that uv cannot resolve; this
# target installs them via 'uv pip install --no-deps'. See the script header.
oct-deps:
	./scripts/setup/install_oct_deps.sh
