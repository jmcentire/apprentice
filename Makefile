.PHONY: install dev test test-quick lint lint-fix serve init clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,lint]"

test:
	python3 -m pytest tests/ -v

test-quick:
	python3 -m pytest tests/ -x -q

lint:
	python3 -m ruff check src/ tests/

lint-fix:
	python3 -m ruff check --fix src/ tests/

serve:
	apprentice serve

init:
	apprentice init

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/
