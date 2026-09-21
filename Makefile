.DEFAULT_GOAL := help
.PHONY: help setup proto lint fmt typecheck test test-integration check web-install web-lint \n        web-build tts-engine worker-story worker-tts up up-gpu down logs clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install Python workspace, web deps, git hooks, and gRPC stubs
	uv sync --all-packages --all-groups
	uv run python infra/scripts/gen_proto.py
	uv run pre-commit install
	cd web && npm install

proto: ## Regenerate gRPC stubs from proto/tts/v1/tts.proto
	uv run python infra/scripts/gen_proto.py

lint: ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Apply ruff fixes and formatting
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Mypy (strict)
	uv run mypy packages services

test: ## Unit tests (no Docker, no network)
	uv run pytest tests/unit -m "not integration and not e2e" --timeout=120

test-integration: ## End-to-end tests against a running compose stack
	uv run pytest tests/integration -m integration --timeout=900

check: lint typecheck test web-lint web-build ## Everything CI runs

web-install: ## Install frontend dependencies
	cd web && npm install

web-lint: ## Lint + typecheck the frontend
	cd web && npm run lint && npm run typecheck

web-build: ## Production build of the frontend
	cd web && npm run build

tts-engine: ## Run the TTS engine (stub backend unless TTS_BACKEND=xtts)
	uv run --project services/tts_engine python -m tts_engine

tts-engine-native: ## Run real XTTS from the GPU venv, for the compose stack to call
	@echo "Point the worker at this first:"
	@echo "  docker compose -f infra/docker-compose.yml stop tts-engine"
	@echo "  TTS_ENGINE_ADDRESS=host.docker.internal:50051 \\"
	@echo "    docker compose --env-file .env -f infra/docker-compose.yml up -d tts-worker"
	COQUI_TOS_AGREED=1 TTS_BACKEND=xtts TTS_ENGINE_HOST=0.0.0.0 \
		.venv-tts-gpu/Scripts/python.exe -m tts_engine

worker-tts: ## Run the TTS worker against the local broker
	uv run --package story2audio-tts-worker celery -A tts_worker.app:celery_app worker \n		--queues tts --concurrency 1 --loglevel info

worker-story: ## Run the story worker against the local broker
	uv run --package story2audio-story-worker celery -A story_worker.app:celery_app worker \n		--queues story --concurrency 2 --loglevel info

up: ## Bring up the local stack (stub TTS; no GPU needed)
	docker compose -f infra/docker-compose.yml up --build

up-gpu: ## Bring up the local stack with real XTTS on a local GPU
	docker compose -f infra/docker-compose.yml -f infra/docker-compose.gpu.yml up --build

down: ## Tear down the local stack
	docker compose -f infra/docker-compose.yml down -v

logs: ## Tail stack logs
	docker compose -f infra/docker-compose.yml logs -f

clean: ## Remove caches and build output
	rm -rf .mypy_cache .ruff_cache .pytest_cache web/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
