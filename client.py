#!/usr/bin/env python3
"""
Python client for FinRAG API - easier to use than curl!

Usage:
    python client.py "What is Apple's latest product launch?"
    python client.py --query "Tell me about Apple" --k 10
    python client.py --ingest data/my_data.csv
"""
import argparse
import requests
import json
from pathlib import Path
from typing import Optional


API_URL = "http://localhost:8000"


def query(
    question: str,
    backend: str = "faiss",
    llm: str = "distilgpt2",
    k: int = 5,
    api_url: str = API_URL
) -> dict:
    """
    Query the FinRAG system.

    Args:
        question: The question to ask
        backend: Vector store backend ('faiss' or 'chroma')
        llm: LLM model to use
        k: Number of documents to retrieve
        api_url: Base URL of the API

    Returns:
        dict with 'answer' and 'sources' keys
    """
    payload = {
        "query": question,
        "backend": backend,
        "llm": llm,
        "k": k
    }

    response = requests.post(f"{api_url}/query", json=payload)
    response.raise_for_status()
    return response.json()


def ingest(
    file_path: str,
    out: str = "./data/ingested.jsonl",
    index_dir: str = "./data/index",
    api_url: str = API_URL
) -> dict:
    """
    Ingest a file into the FinRAG system.

    Args:
        file_path: Path to CSV or JSONL file
        out: Output path for ingested data
        index_dir: Directory to store vector index
        api_url: Base URL of the API

    Returns:
        dict with status and output paths
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f)}
        data = {"out": out, "index_dir": index_dir}
        response = requests.post(f"{api_url}/ingest", files=files, data=data)
        response.raise_for_status()
        return response.json()


def health_check(api_url: str = API_URL) -> dict:
    """Check if the API is healthy."""
    response = requests.get(f"{api_url}/health")
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(
        description="FinRAG Python Client - Query financial documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query the system
  python client.py "What is Apple's latest product?"
  python client.py --query "Tell me about revenue" --k 10

  # Ingest a file
  python client.py --ingest data/aapl_news.csv

  # Health check
  python client.py --health
        """
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Question to ask (positional argument)"
    )
    parser.add_argument(
        "--query", "-q",
        dest="query_arg",
        help="Question to ask (flag argument)"
    )
    parser.add_argument(
        "--ingest", "-i",
        help="Path to file to ingest (CSV or JSONL)"
    )
    parser.add_argument(
        "--backend", "-b",
        default="faiss",
        choices=["faiss", "chroma"],
        help="Vector store backend (default: faiss)"
    )
    parser.add_argument(
        "--llm", "-l",
        default="distilgpt2",
        help="LLM model to use (default: distilgpt2)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--out", "-o",
        default="./data/ingested.jsonl",
        help="Output path for ingested data"
    )
    parser.add_argument(
        "--index-dir",
        default="./data/index",
        help="Index directory"
    )
    parser.add_argument(
        "--url",
        default=API_URL,
        help=f"API URL (default: {API_URL})"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check API health"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON response"
    )

    args = parser.parse_args()

    try:
        # Health check
        if args.health:
            result = health_check(args.url)
            print(f"✅ API is healthy: {result}")
            return

        # Ingest
        if args.ingest:
            print(f"📥 Ingesting file: {args.ingest}")
            result = ingest(args.ingest, args.out, args.index_dir, args.url)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"✅ Ingestion complete!")
                print(f"   Output: {result.get('output')}")
                print(f"   Index: {result.get('index_dir')}")
            return

        # Query
        question = args.query or args.query_arg
        if not question:
            parser.print_help()
            return

        print(f"🔍 Querying: {question}")
        result = query(question, args.backend, args.llm, args.k, args.url)

        if args.json:
            print(json.dumps(result, indent=2))
        elif "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"\n💡 Answer:\n{result['answer']}\n")
            if result.get('sources'):
                print(f"📚 Sources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"   {i}. {source}")

    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API at {args.url}")
        print("   Make sure the API is running with: docker compose up api")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if e.response is not None:
            print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
