# src/build_index.py
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
from utils import env, ensure_dir, init_metadata_db, store_metadata
from pathlib import Path
from tqdm import tqdm

def build(jsonl_path, index_dir, model_name="all-MiniLM-L6-v2", db_path="./data/metadata.db"):
    jsonl_path = Path(jsonl_path)
    index_dir = Path(index_dir)
    ensure_dir(index_dir)
    model = SentenceTransformer(model_name)

    embeddings = []
    ids = []
    conn = init_metadata_db(db_path)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            doc = json.loads(line)
            text = doc["text"]
            emb = model.encode(text, normalize_embeddings=True)
            embeddings.append(emb)
            ids.append(doc["id"])
            store_metadata(conn, doc["id"], doc["metadata"].get("source"), doc["metadata"])

    if len(embeddings) == 0:
        raise SystemExit("No documents found to index.")

    emb_matrix = np.vstack(embeddings).astype("float32")
    dim = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)  # use inner-product since embeddings are normalized -> cosine
    index.add(emb_matrix)
    faiss.write_index(index, str(index_dir / "faiss.index"))

    # persist ids order
    with open(index_dir / "ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f)

    print(f"Index built: {len(ids)} vectors, dim={dim}, saved to {index_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="./data/ingested.jsonl")
    parser.add_argument("--index-dir", default=env("INDEX_DIR", "./data/index"))
    parser.add_argument("--model", default=env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    parser.add_argument("--db", default=env("METADATA_DB", "./data/metadata.db"))
    args = parser.parse_args()
    build(args.jsonl, args.index_dir, args.model, args.db)
