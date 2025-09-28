
"""
Evaluation script: checks retrieval quality on sample Q&A.
"""
import json
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def load_vectorstore(index_dir, backend="faiss"):
    embeddings = OpenAIEmbeddings()
    if backend == "faiss":
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    elif backend == "chroma":
        return Chroma(persist_directory=index_dir, embedding_function=embeddings)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def evaluate(index_dir, backend="faiss"):
    # Example evaluation dataset
    eval_data = [
        {"query": "What did Apple announce recently?", "expected": "Apple"},
        {"query": "Latest news about Microsoft", "expected": "Microsoft"},
    ]

    db = load_vectorstore(index_dir, backend)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    correct = 0
    for item in eval_data:
        docs = retriever.get_relevant_documents(item["query"])
        if any(item["expected"].lower() in d.page_content.lower() for d in docs):
            correct += 1

    accuracy = correct / len(eval_data)
    logging.info(f"Evaluation accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    evaluate("./data/index", backend="faiss")
