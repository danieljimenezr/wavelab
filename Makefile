.PHONY: help sync test test-net bench lint run hydrate clean provision contain-test
UV := uv

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync:          ## Install the environment (CPython 3.13 via uv)
	$(UV) sync --group dev

test:          ## Full suite, exactly what CI gates on. The M0 gate lives here.
	$(UV) run pytest

test-net:      ## The handful that really call Binance. Fails when someone else's API is down.
	$(UV) run pytest -m net

bench:         ## Actually measures the timings and records them in docs/BENCHMARKS.md
	$(UV) run pytest tests/bench --benchmark-only -q || true

lint:
	$(UV) run ruff check src tests

run:           ## Feeds + engine + web on 127.0.0.1:8000  (M2)
	$(UV) run uvicorn wavelab.server.app:app --host 127.0.0.1 --port 8000

hydrate:       ## Download the history from data.binance.vision  (M1)
	$(UV) run python -m wavelab.store.hydrate

provision:     ## VPS containment: loopback image, slice, units, quotas  (M0b)
	@echo "Requires ssh with sudo on the node. See scripts/provision_vps.sh"

contain-test:  ## Proves wavelab cannot fill the host disk  (M0b)
	@echo "See scripts/contain_test.sh - it runs ON the node"

clean:
	rm -rf .pytest_cache .hypothesis .ruff_cache .benchmarks
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
