# src/build_index.py
import json
from pathlib import Path
from tqdm import tqdm
import argparse

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from utils import env, ensure_dir, init_metadata_db, store_metadata

def build(jsonl_path, index_dir, model_name="all-MiniLM-L6-v2", db_path="./data/metadata.db"):
    """
    Build a FAISS vectorstore index from a JSONL file with text documents.
    Saves both FAISS index and metadata for retrieval.
    """
    jsonl_path = Path(jsonl_path)
    index_dir = Path(index_dir)
    ensure_dir(index_dir)

    # Load embeddings model
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

    # Load documents and store metadata
    docs = []
    conn = init_metadata_db(db_path)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading documents"):
            doc_json = json.loads(line)
            doc_id = doc_json["id"]
            text = doc_json["text"]
            metadata = doc_json.get("metadata", {})
            docs.append(Document(page_content=text, metadata=metadata))
            store_metadata(conn, doc_id, metadata.get("source"), metadata)

    if len(docs) == 0:
        raise SystemExit("No documents found to index.")

    # Build FAISS vectorstore using langchain_community
    vectorstore = FAISS.from_documents(docs, embeddings_model)

    # Save the vectorstore locally (FAISS index + docstore)
    vectorstore.save_local(str(index_dir))

    print(f"Index built successfully: {len(docs)} documents, saved to {index_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from JSONL documents")
    parser.add_argument("--jsonl", default="./data/ingested.jsonl", help="Path to input JSONL file")
    parser.add_argument("--index-dir", default=env("INDEX_DIR", "./data/index"), help="Directory to save FAISS index")
    parser.add_argument("--model", default=env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"), help="Embedding model name")
    parser.add_argument("--db", default=env("METADATA_DB", "./data/metadata.db"), help="Path to metadata DB")
    args = parser.parse_args()

    build(args.jsonl, args.index_dir, args.model, args.db)
