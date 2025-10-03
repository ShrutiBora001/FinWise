import pytest
from fastapi.testclient import TestClient
from src.api import app
from pathlib import Path
import shutil

client = TestClient(app)

DATA_DIR = Path("./data")
UPLOADS_DIR = DATA_DIR / "uploads"
INGESTED_FILE = DATA_DIR / "ingested.jsonl"
INDEX_DIR = DATA_DIR / "index"

# Utility to clean previous files for a fresh test
def clean_test_files():
    if INGESTED_FILE.exists():
        INGESTED_FILE.unlink()
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    if UPLOADS_DIR.exists():
        shutil.rmtree(UPLOADS_DIR)
    # Clean cache file
    cache_file = DATA_DIR / "cache.json"
    if cache_file.exists():
        cache_file.unlink()

@pytest.fixture(scope="module", autouse=True)
def setup_module():
    clean_test_files()
    yield
    clean_test_files()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ingest_and_query():
    # Step 1: Ingest file
    with open(DATA_DIR / "aapl_news.csv", "rb") as f:
        response = client.post(
            "/ingest",
            files={"file": ("aapl_news.csv", f)},
            data={"out": str(INGESTED_FILE), "index_dir": str(INDEX_DIR)}
        )
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["status"] == "success"
    assert Path(json_resp["output"]).exists()
    assert Path(json_resp["index_dir"]).exists()

    # Step 2: Query
    query_payload = {
        "query": "What is Apple's latest product launch?",
        "backend": "faiss",
        "llm": "distilgpt2",
        "k": 5
    }
    response = client.post("/query", json=query_payload)
    assert response.status_code == 200
    json_resp = response.json()
    assert "answer" in json_resp
    assert "sources" in json_resp
    assert isinstance(json_resp["sources"], list)
