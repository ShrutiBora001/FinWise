import pytest
import sys
from pathlib import Path
import time
import shutil

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from query_pipeline import run_query

# Test paths and parameters
INDEX_DIR = "./data/index"
QUERY = "What is Apple's latest product launch?"
BACKEND = "faiss"
LLM_MODEL = "distilgpt2"
K = 5
CACHE_FILE = Path("./data/cache.json")

@pytest.fixture(autouse=True)
def clean_cache():
    """Clean cache before each test"""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    yield
    # Cleanup after test
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()

def test_cache_miss_then_hit():
    """Test that first query is cache miss, second is cache hit"""
    # First query should be slower (cache miss)
    start1 = time.time()
    result1 = run_query(QUERY, INDEX_DIR, BACKEND, LLM_MODEL, K, use_cache=True)
    time1 = time.time() - start1

    # Verify result structure
    assert "result" in result1
    assert "source_documents" in result1
    assert isinstance(result1["source_documents"], list)
    for doc in result1["source_documents"]:
        assert "metadata" in doc
        assert "page_content" in doc

    # Second query should be faster (cache hit)
    start2 = time.time()
    result2 = run_query(QUERY, INDEX_DIR, BACKEND, LLM_MODEL, K, use_cache=True)
    time2 = time.time() - start2

    # Verify cached result matches
    assert result1["result"] == result2["result"]
    assert len(result1["source_documents"]) == len(result2["source_documents"])

    # Cache hit should be significantly faster
    assert time2 < time1, f"Cache hit ({time2:.2f}s) should be faster than miss ({time1:.2f}s)"
    print(f"✅ Cache miss: {time1:.2f}s, Cache hit: {time2:.2f}s")

def test_cache_disabled():
    """Test that cache can be disabled"""
    result1 = run_query(QUERY, INDEX_DIR, BACKEND, LLM_MODEL, K, use_cache=False)
    result2 = run_query(QUERY, INDEX_DIR, BACKEND, LLM_MODEL, K, use_cache=False)

    # Results should still match even without cache
    assert result1["result"] == result2["result"]

def test_different_queries_different_cache():
    """Test that different queries get different cache entries"""
    query1 = "What is Apple's latest product?"
    query2 = "When was Apple founded?"

    result1 = run_query(query1, INDEX_DIR, BACKEND, LLM_MODEL, K, use_cache=True)
    result2 = run_query(query2, INDEX_DIR, BACKEND, LLM_MODEL, K, use_cache=True)

    # Different queries should produce different results
    assert result1["result"] != result2["result"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
