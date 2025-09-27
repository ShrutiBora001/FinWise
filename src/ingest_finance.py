# src/ingest_finance.py
"""
Simple ingester: reads CSV (expects columns: id, text, date, source, ticker(optional))
Outputs a cleaned CSV or a JSONL used by build_index.py
"""
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import uuid
import argparse
from utils import env, ensure_dir
import json

def ingest_csv(path, out_path):
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        doc_id = str(row.get("id") or uuid.uuid4())
        text = str(row.get("text") or row.get("content") or "")
        if not text.strip():
            continue
        metadata = {
            "date": row.get("date"),
            "source": row.get("source"),
            "ticker": row.get("ticker")
        }
        records.append({"id": doc_id, "text": text, "metadata": metadata})
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} docs to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=env("DATA_PATH", "./data/finance_sample.csv"))
    parser.add_argument("--out", default="./data/ingested.jsonl")
    args = parser.parse_args()
    ingest_csv(args.csv, args.out)