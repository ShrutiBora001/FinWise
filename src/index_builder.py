
"""
Index builder: reads ingested JSONL and stores in vector DB (FAISS/Chroma/Pinecone).
"""
import argparse
import json
from pathlib import Path
import logging

from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings

from utils import env

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def build_index(input_path, index_dir, backend="faiss"):
    input_path = Path(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    texts = [r["text"] for r in records]
    metadatas = [r["metadata"] for r in records]

    embeddings = OpenAIEmbeddings(model=env("EMBED_MODEL", "text-embedding-3-small"))

    if backend == "faiss":
        db = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    elif backend == "chroma":
        db = Chroma.from_texts(texts, embedding=embeddings, metadatas=metadatas, persist_directory=index_dir)
        db.persist()
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    logging.info(f"Index built with {len(records)} documents using backend={backend}")
    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./data/ingested.jsonl")
    parser.add_argument("--index-dir", default="./data/index")
    parser.add_argument("--backend", default="faiss", choices=["faiss", "chroma"])
    args = parser.parse_args()

    build_index(args.input, args.index_dir, backend=args.backend)
