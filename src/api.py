
"""
FastAPI layer for FinRAG system.
Endpoints:
- /query: ask a question
- /ingest: ingest new data file (CSV or JSONL)
"""
from fastapi import FastAPI, UploadFile, Form
import shutil
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from ingest_finance import ingest
from query_pipeline import run_query
from logger import get_logger

app = FastAPI(title="FinRAG API")
logger = get_logger(__name__)

from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    backend: str = "faiss"
    llm: str = "distilgpt2"
    k: int = 5
import math

def clean_metadata(md: dict):
    """Remove NaN/None values to make JSON-safe metadata."""
    cleaned = {}
    for k, v in md.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        if v is None:
            continue
        cleaned[k] = v
    return cleaned

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        result = run_query(request.query, "./data/index", request.backend, request.llm, request.k)

        sources = []
        for d in result.get("source_documents", []):
            if isinstance(d, dict):
                sources.append(clean_metadata(d.get("metadata", {})))
            else:  # LangChain Document
                sources.append(clean_metadata(d.metadata))

        return {
            "query": request.query,
            "answer": result["result"],
            "sources": sources,
        }
    except Exception as e:
        logger.exception("Query failed")
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
