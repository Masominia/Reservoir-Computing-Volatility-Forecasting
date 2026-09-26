.PHONY: install test lint run data all

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

test:
	pytest

run:
	python -m reservoirvol run

data:
	python scripts/fetch_data.py

all: lint test run
