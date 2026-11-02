PYTHON ?= python3.12
VENV := .venv
RUN := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
ARTIFACT_VERSION ?= v2
PROCESSED_SNAPSHOT ?= data/processed/snapshots/$(ARTIFACT_VERSION)
SOURCE_MANIFEST ?= reports/source_manifest.$(ARTIFACT_VERSION).json
DEVELOPMENT_REPORTS ?= reports/experiments/$(ARTIFACT_VERSION)
LOCKED_REPORTS ?= reports/locked/$(ARTIFACT_VERSION)
ARTIFACT_OUTPUT ?= artifacts/$(ARTIFACT_VERSION)

.PHONY: install lint format-check test quality data-plan data-download data-build data-audit experiment-prepare experiment-select experiment-confirm locked-rank locked-report package-release verify-artifact artifact-gate api app docker-build compose-check

install:
	$(PYTHON) -m venv $(VENV)
	$(RUN) -m pip install --requirement requirements.lock
	$(RUN) -m pip install --no-deps --no-build-isolation --editable .
	$(RUN) -m pip check

lint:
	$(RUFF) check .

format-check:
	$(RUFF) format --check .

test:
	PYTHONWARNINGS=error $(RUN) -m pytest -W error --cov=job_market_intelligence --cov=api --cov=app --cov-report=term-missing --cov-fail-under=70

quality: lint format-check test

data-plan:
	$(RUN) scripts/download_data.py --plan

data-download:
	$(RUN) scripts/download_data.py

data-build:
	$(RUN) scripts/build_warehouse.py

data-audit:
	$(RUN) scripts/audit_data.py

experiment-prepare:
	$(RUN) scripts/benchmark.py prepare \
		--snapshot $(PROCESSED_SNAPSHOT) \
		--output $(DEVELOPMENT_REPORTS) \
		--source-manifest $(SOURCE_MANIFEST)

experiment-select:
	$(RUN) scripts/benchmark.py select \
		--snapshot $(PROCESSED_SNAPSHOT) \
		--output $(DEVELOPMENT_REPORTS) \
		--source-manifest $(SOURCE_MANIFEST)

experiment-confirm:
	$(RUN) scripts/benchmark.py confirm \
		--snapshot $(PROCESSED_SNAPSHOT) \
		--output $(DEVELOPMENT_REPORTS) \
		--source-manifest $(SOURCE_MANIFEST)

locked-rank:
	$(RUN) scripts/evaluate.py begin-and-rank \
		--snapshot $(PROCESSED_SNAPSHOT) \
		--development $(DEVELOPMENT_REPORTS) \
		--output $(LOCKED_REPORTS) \
		--source-manifest $(SOURCE_MANIFEST)

locked-report:
	$(RUN) scripts/evaluate.py report-only --output $(LOCKED_REPORTS)

package-release:
	$(RUN) scripts/train.py \
		--snapshot $(PROCESSED_SNAPSHOT) \
		--development $(DEVELOPMENT_REPORTS) \
		--locked $(LOCKED_REPORTS) \
		--source-manifest $(SOURCE_MANIFEST) \
		--output $(ARTIFACT_OUTPUT)

verify-artifact:
	$(RUN) -m job_market_intelligence.cli --artifact-dir $(ARTIFACT_OUTPUT) verify-artifact

artifact-gate:
	@if test -f $(ARTIFACT_OUTPUT)/manifest.json; then \
		$(MAKE) verify-artifact; \
	else \
		$(RUN) -m job_market_intelligence.cli --artifact-dir $(ARTIFACT_OUTPUT) verify-artifact >/dev/null 2>&1; \
		test $$? -eq 3; \
	fi

api:
	JOB_INTELLIGENCE_ARTIFACT_DIR=$(ARTIFACT_OUTPUT) $(VENV)/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000

app:
	JOB_INTELLIGENCE_ARTIFACT_DIR=$(ARTIFACT_OUTPUT) $(VENV)/bin/streamlit run app/main.py --server.headless=true

docker-build:
	docker build --target runtime --tag ai-job-market-intelligence:local .

compose-check:
	docker compose config --quiet
