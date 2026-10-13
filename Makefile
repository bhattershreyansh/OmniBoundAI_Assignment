.PHONY: run test ingest help

help:
	@echo "Available commands:"
	@echo "  make run      - Run OmniBound FastAPI server"
	@echo "  make test     - Run pytest unit tests (offline, no API key needed)"
	@echo "  make ingest   - Load and ingest corpus data for tenant 'acme'"

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

ingest:
	python scripts/load_corpus.py
