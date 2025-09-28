"""
Query pipeline: adds caching layer on top of retrieval + local Hugging Face LLM answering.
"""
import argparse
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.vectorstores import FAISS  # note: not langchain_community
from langchain.embeddings import HuggingFaceEmbeddings


from utils import env
from logger import get_logger
from caching import Cache

logger = get_logger(__name__)


def load_vectorstore(index_dir, backend="faiss"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if backend == "faiss":
        vectorstore = FAISS.load_local(
            index_dir, 
            embeddings, 
            allow_dangerous_deserialization=True  # safe since you built it
        )
        return vectorstore
    elif backend == "chroma":
        from langchain_community.vectorstores import Chroma
        return Chroma(persist_directory=index_dir, embedding_function=embeddings)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def load_local_llm(model_name="distilgpt2", task="text-generation"):
    pipe = pipeline(task=task, model=model_name, device=-1)  # -1 = CPU
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def make_query_pipeline(index_dir, backend="faiss", llm_model="distilgpt2", k=5):
    logger.info(f"Loading vectorstore from {index_dir} using {backend}...")
    db = load_vectorstore(index_dir, backend)
    retriever = db.as_retriever(search_kwargs={"k": k})

    logger.info(f"Setting up local Hugging Face model {llm_model} for answer generation...")
    llm = load_local_llm(model_name=llm_model)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa

def serialize_result(result):
    # Convert Document objects to dicts
    if "source_documents" in result:
        docs = result["source_documents"]
        result["source_documents"] = [
            {"page_content": d.page_content, "metadata": d.metadata} for d in docs
        ]
    return result

def run_query(query, index_dir, backend="faiss", llm_model="distilgpt2", k=5, use_cache=True):
    cache = Cache()
    
    # Try retrieving from cache
    if use_cache:
        cached = cache.get(query)
        if cached:
            logger.info(f"Cache hit for query: {query}")
            return cached

    # Run retrieval + LLM pipeline
    qa = make_query_pipeline(index_dir, backend, llm_model, k)
    result = qa.invoke(query)

    # Convert Documents to dicts for JSON serialization
    if use_cache:
        result_to_cache = result.copy()
        if "source_documents" in result_to_cache:
            result_to_cache["source_documents"] = [
                {"page_content": doc.page_content, "metadata": doc.metadata} 
                for doc in result_to_cache["source_documents"]
            ]
        cache.set(query, result_to_cache)

    logger.info(f"Query: {query}")
    logger.info(f"Answer: {result['result']}")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--index-dir", default="./data/index")
    parser.add_argument("--backend", default="faiss", choices=["faiss", "chroma"])
    parser.add_argument("--llm", default="distilgpt2")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    run_query(args.query, args.index_dir, args.backend, args.llm, args.k, use_cache=not args.no_cache)
