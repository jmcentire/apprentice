.PHONY: install dev test test-quick lint clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	python3 -m pytest tests/ -v

test-quick:
	python3 -m pytest tests/ -x -q

lint:
	python3 -m py_compile src/apprentice/__init__.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/
