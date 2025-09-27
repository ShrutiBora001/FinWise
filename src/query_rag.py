# src/query_rag.py
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
from utils import env, init_metadata_db, load_metadata
from pathlib import Path
import os
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


# optional text generation: OpenAI or HF local
USE_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))

if USE_OPENAI:
    import openai

def load_index(index_dir):
    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / "faiss.index"))
    with open(index_dir / "ids.json", "r", encoding="utf-8") as f:
        ids = json.load(f)
    return index, ids

def retrieve(query, index, ids, embed_model, topk=5):
    q_emb = embed_model.encode(query, normalize_embeddings=True).astype("float32")
    D, I = index.search(np.array([q_emb]), topk)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        doc_id = ids[int(idx)]
        results.append((doc_id, float(score)))
    return results

def generate_answer_openai(prompt, openai_api_key, max_tokens=300):
    openai.api_key = openai_api_key
    # Keep this simple: use chat completion if available
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if True else "gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # fallback to text completion
        resp = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=max_tokens)
        return resp["choices"][0]["text"].strip()

def generate_answer_local(prompt, hf_model_name="facebook/opt-350m", max_new_tokens=256):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # device=-1 -> CPU
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"]

def build_prompt(query, retrieved_docs):
    header = "You are a helpful finance assistant. Use the following extracted documents to answer the question. Cite the source id in square brackets.\n\n"
    docs_text = ""
    for doc_id, score, snippet in retrieved_docs:
        docs_text += f"[{doc_id}] (score={score:.3f}): {snippet}\n\n"
    prompt = header + "Documents:\n" + docs_text + "\nQuestion:\n" + query + "\nAnswer concisely:"
    return prompt

def get_snippet(doc_id, conn):
    meta = load_metadata(conn, doc_id) or {}
    # In this Phase 1 we don't store full text in the DB; for now return metadata summary.
    return meta.get("summary") or json.dumps(meta)[:500]

def rag_query(query, index_dir, embed_model_name, topk=5):
    index, ids = load_index(index_dir)
    embed_model = SentenceTransformer(embed_model_name)
    hits = retrieve(query, index, ids, embed_model, topk=topk)
    conn = init_metadata_db(env("METADATA_DB", "./data/metadata.db"))
    retrieved_docs = []
    for doc_id, score in hits:
        snippet = get_snippet(doc_id, conn)
        retrieved_docs.append((doc_id, score, snippet))
    prompt = build_prompt(query, retrieved_docs)
    if USE_OPENAI and os.environ.get("OPENAI_API_KEY"):
        answer = generate_answer_openai(prompt, os.environ.get("OPENAI_API_KEY"))
    else:
        answer = generate_answer_local(prompt, os.environ.get("HF_MODEL", "facebook/opt-350m"))
    return {"query": query, "answer": answer, "retrieved": retrieved_docs}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--index-dir", default=env("INDEX_DIR", "./data/index"))
    parser.add_argument("--model", default=env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    res = rag_query(args.query, args.index_dir, args.model, args.topk)
    print("=== ANSWER ===")
    print(res["answer"])
    print("\n=== RETRIEVED ===")
    for doc_id, score, snippet in res["retrieved"]:
        print(f"{doc_id} (score={score}): {snippet}")
