.PHONY: help sync test bench lint run hydrate clean provision contain-test
UV := uv

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync:          ## Instala el entorno (CPython 3.13 vía uv)
	$(UV) sync --group dev

test:          ## Suite completa. La puerta de M0 vive aquí.
	$(UV) run pytest

bench:         ## Mide de verdad los tiempos y los anota en docs/BENCHMARKS.md
	$(UV) run pytest tests/bench --benchmark-only -q || true

lint:
	$(UV) run ruff check src tests

run:           ## Feeds + motor + web en 127.0.0.1:8000  (M2)
	$(UV) run uvicorn wavelab.server.app:app --host 127.0.0.1 --port 8000

hydrate:       ## Descarga el histórico desde data.binance.vision  (M1)
	$(UV) run python -m wavelab.store.hydrate

provision:     ## Contención del VPS: imagen loopback, slice, units, cuotas  (M0b)
	@echo "Requiere ssh con sudo en el nodo. Ver scripts/provision_vps.sh"

contain-test:  ## Demuestra que wavelab no puede llenar el disco del host  (M0b)
	@echo "Ver scripts/contain_test.sh — se ejecuta EN el nodo"

clean:
	rm -rf .pytest_cache .hypothesis .ruff_cache .benchmarks
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
