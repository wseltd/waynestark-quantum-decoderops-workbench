# Developer entry points for the Quantum DecoderOps Workbench.
# Every tool invocation resolves through $(VENV)/bin/... so the Makefile works
# without a sourced venv (the protected .venv is the canonical environment).

SHELL := /bin/bash

VENV ?= .venv
RUN_ID ?=
OUTPUT_DIR ?=
MSG ?=

.PHONY: help lint format type-check test test-integration test-real-artefacts test-runtime-capability test-all docker-build report-render db-migrate db-migrate-revision clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2}'

lint:  ## Run ruff lint checks over app and tests
	$(VENV)/bin/ruff check app tests

format:  ## Apply ruff formatting to app and tests
	$(VENV)/bin/ruff format app tests

type-check:  ## Run mypy over the app package
	$(VENV)/bin/mypy app

test:  ## Run fast unit tests (excludes integration, real_artefacts, runtime_capability, gpu)
	$(VENV)/bin/pytest -m "not integration and not real_artefacts and not runtime_capability and not gpu"

test-integration:  ## Run deterministic integration tests
	$(VENV)/bin/pytest -m integration

test-real-artefacts:  ## Run integration tests that exercise the real NVIDIA Ising checkpoints
	$(VENV)/bin/pytest -m real_artefacts

test-runtime-capability:  ## Run Tier 3 runtime capability probes (TensorRT, cudaq, cuQuantum, ORT)
	$(VENV)/bin/pytest -m runtime_capability

test-all:  ## Run the full pytest suite (all markers)
	$(VENV)/bin/pytest

docker-build:  ## Build the workbench Docker image
	docker build -f docker/Dockerfile -t decoderops:dev .

report-render:  ## Render reports for a run (RUN_ID=... OUTPUT_DIR=...)
	$(VENV)/bin/python -m app.reports.render --run-id $(RUN_ID) --output-dir $(OUTPUT_DIR)

db-migrate:  ## Apply Alembic migrations up to head
	$(VENV)/bin/alembic upgrade head

db-migrate-revision:  ## Create a new Alembic revision (MSG="description")
	$(VENV)/bin/alembic revision --autogenerate -m '$(MSG)'

clean:  ## Remove build and cache artefacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
