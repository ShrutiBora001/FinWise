
"""
Ingestion script: reads CSV or JSONL (expects columns/fields: id, text/content, date, source, ticker(optional))
Outputs a cleaned JSONL used by build_index.py
"""
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import uuid
import argparse
import json
import logging

from utils import env, ensure_dir
from preprocess import clean_text, deduplicate_records

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def ingest_csv(path):
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        doc_id = str(row.get("id") or uuid.uuid4())
        text = str(row.get("text") or row.get("content") or "")
        if not text.strip():
            logging.warning(f"Skipping empty row with id={doc_id}")
            continue
        metadata = {
            "date": row.get("date"),
            "source": row.get("source"),
            "ticker": row.get("ticker"),
        }
        records.append({"id": doc_id, "text": text, "metadata": metadata})
    return records


def ingest_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                doc_id = str(row.get("id") or uuid.uuid4())
                text = str(row.get("text") or row.get("content") or "")
                if not text.strip():
                    continue
                metadata = {
                    "date": row.get("date"),
                    "source": row.get("source"),
                    "ticker": row.get("ticker"),
                }
                records.append({"id": doc_id, "text": text, "metadata": metadata})
            except json.JSONDecodeError:
                logging.error(f"Skipping invalid JSON line: {line}")
    return records


def ingest(path, out_path, clean=True):
    path = Path(path)
    if path.suffix == ".csv":
        records = ingest_csv(path)
    elif path.suffix in [".jsonl", ".json"]:
        records = ingest_jsonl(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if clean:
        logging.info("Cleaning and deduplicating records...")
        for r in records:
            r["text"] = clean_text(r["text"])
        records = deduplicate_records(records)

    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logging.info(f"Wrote {len(records)} docs to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=env("DATA_PATH", "./data/finance_sample.csv"))
    parser.add_argument("--out", default="./data/ingested.jsonl")
    parser.add_argument("--no-clean", action="store_true", help="Skip cleaning/deduplication")
    args = parser.parse_args()

    ingest(args.input, args.out, clean=not args.no_clean)

