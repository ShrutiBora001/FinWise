
"""
FastAPI layer for FinRAG system.
Endpoints:
- /query: ask a question
- /ingest: ingest new data file (CSV or JSONL)
"""
from fastapi import FastAPI, UploadFile, Form
import shutil
from pathlib import Path

from ingest_finance import ingest
from query_pipeline import run_query
from logger import get_logger

app = FastAPI(title="FinRAG API")
logger = get_logger(__name__)


@app.post("/query")
async def query_endpoint(query: str, backend: str = "faiss", llm: str = "gpt-4o-mini", k: int = 5):
    try:
        result = run_query(query, "./data/index", backend, llm, k)
        return {"query": query, "answer": result["result"], "sources": [d.metadata for d in result["source_documents"]]}
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return {"error": str(e)}


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile, out: str = "./data/ingested.jsonl"):
    try:
        temp_path = Path(f"./data/uploads/{file.filename}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ingest(str(temp_path), out)
        return {"status": "success", "output": out}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return {"error": str(e)}


@app.get("/health")
async def health_check():
    return {"status": "ok"}
