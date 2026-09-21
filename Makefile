.DEFAULT_GOAL := help
.PHONY: help setup lint fmt typecheck test check web-install web-lint web-build worker-story up down logs clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install Python workspace, web deps, and git hooks
	uv sync --all-packages --all-groups
	uv run pre-commit install
	cd web && npm install

lint: ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Apply ruff fixes and formatting
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Mypy (strict)
	uv run mypy packages services

test: ## Unit tests
	uv run pytest tests/unit -m "not integration and not e2e" --timeout=120

check: lint typecheck test web-lint web-build ## Everything CI runs

web-install: ## Install frontend dependencies
	cd web && npm install

web-lint: ## Lint + typecheck the frontend
	cd web && npm run lint && npm run typecheck

web-build: ## Production build of the frontend
	cd web && npm run build

worker-story: ## Run the story worker against the local broker
	uv run --package story2audio-story-worker celery -A story_worker.app:celery_app worker \n		--queues story --concurrency 2 --loglevel info

up: ## Bring up the local stack (Phase 5)
	docker compose -f infra/docker-compose.yml up --build

down: ## Tear down the local stack
	docker compose -f infra/docker-compose.yml down -v

logs: ## Tail stack logs
	docker compose -f infra/docker-compose.yml logs -f

clean: ## Remove caches and build output
	rm -rf .mypy_cache .ruff_cache .pytest_cache web/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
