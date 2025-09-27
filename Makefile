.PHONY: venv ingest build index query

venv:
	bash create_venv.sh

ingest:
	python src/ingest_finance.py --csv $(DATA_PATH)

build:
	python src/build_index.py --jsonl ./data/ingested.jsonl --index-dir $(INDEX_DIR)

query:
	python src/query_rag.py --query "$(q)" --index-dir $(INDEX_DIR)
