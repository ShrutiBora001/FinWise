import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from query_pipeline import run_query

import time
from query_pipeline import run_query

# Adjust these paths and parameters as needed
INDEX_DIR = "./data/index"
QUERY = "What is Apple's latest product launch?"
BACKEND = "faiss"
LLM_MODEL = "distilgpt2"
K = 5

def test_query_cache():
    print("Running first query (should be a cache miss)...")
    start_time = time.time()
    result1 = run_query(QUERY, INDEX_DIR, BACKEND, LLM_MODEL, K, use_cache=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f}s")
    print(f"Answer: {result1['result'][:100]}...")  # print first 100 chars
    print(f"Sources: {[doc['metadata'] for doc in result1['source_documents']]}")
    print("\n")

    print("Running second query (should be a cache hit)...")
    start_time = time.time()
    result2 = run_query(QUERY, INDEX_DIR, BACKEND, LLM_MODEL, K, use_cache=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f}s")
    print(f"Answer: {result2['result'][:100]}...")
    print(f"Sources: {[doc['metadata'] for doc in result2['source_documents']]}")

    # Check if cached result matches the first result
    assert result1['result'] == result2['result'], "Cached result does not match original result!"
    print("\n✅ Cache test passed!")

if __name__ == "__main__":
    test_query_cache()
