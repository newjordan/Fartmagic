.PHONY: install-dev test verify-stack

install-dev:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest

verify-stack:
	bash scripts/pod_stack_guard.sh verify
