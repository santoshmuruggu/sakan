"""
index.py — builds the search indexes over the chunks ingest.py produced.

Builds two separate search indexes:

  1. A VECTOR index (ChromaDB) — finds chunks by MEANING, using
     embeddings. Good for questions phrased differently than the law's
     own wording (e.g. "kicked out" instead of "eviction").
  2. A KEYWORD index (BM25) — finds chunks by exact word match, like a
     smarter Ctrl+F. Good for legal terms embeddings can blur together,
     e.g. "Ejari" or an exact article number.

retrieve.py combines results from both for hybrid search. ChromaDB's
default embedding model runs locally (no API key needed); an LLM API key
only becomes necessary at the answer-generation step.
"""

import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

PROCESSED_DIR = Path("data/processed")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "sakan_articles"


def load_chunks() -> list[dict]:
    return json.loads((PROCESSED_DIR / "chunks.json").read_text())


def chunk_label(chunk: dict) -> str:
    """A short human-readable label prepended to each chunk's text before
    embedding/indexing, so the embedding model and BM25 both 'see' which
    article this is, not just bare legal prose with no context. Also used
    as the citation format the LLM is told to use in generate.py, so this
    is the ONE place that format is defined."""
    law = chunk["law"]
    if law:
        # Decree entries already read as "Decree 43/2013" — don't turn
        # that into the double-prefixed "Law Decree 43/2013".
        prefix = law if law.startswith("Decree") else f"Law {law}"
        return f"{prefix}, Article {chunk['article_no']}"
    return f"Tenancy Guide — {chunk['section']}"


def chunk_document(chunk: dict) -> str:
    return f"{chunk_label(chunk)}: {chunk['text']}"


def chroma_metadata(chunk: dict) -> dict:
    """ChromaDB metadata values must be str/int/float/bool — never None —
    so prose chunks (no law/article_no) and law articles (no section) get
    placeholder values instead."""
    return {
        "law": chunk["law"] or "",
        "article_no": chunk["article_no"] if chunk["article_no"] is not None else -1,
        "section": chunk["section"] or "",
        "source_pdf": chunk["source_pdf"],
        "is_current": chunk["is_current"],
    }


def build_vector_index(chunks: list[dict]) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Drop and rebuild every time index.py runs, so re-running after an
    # ingest.py change never leaves stale chunks sitting in the collection.
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        ids=[str(c["chunk_id"]) for c in chunks],
        documents=[chunk_document(c) for c in chunks],
        metadatas=[chroma_metadata(c) for c in chunks],
    )
    return collection


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    tokenized_corpus = [tokenize(chunk_document(c)) for c in chunks]
    return BM25Okapi(tokenized_corpus)


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {PROCESSED_DIR / 'chunks.json'}")

    print("\nBuilding vector index (ChromaDB, local embedding model)...")
    collection = build_vector_index(chunks)
    print(f"  -> {collection.count()} chunks embedded, stored in {CHROMA_DIR}/")

    print("Building keyword index (BM25)...")
    bm25 = build_bm25_index(chunks)
    print("  -> built in memory (cheap enough at 78 chunks to rebuild on every run)")

    # Smoke test: a real eval question (f01). Both indexes should surface
    # the RERA/Ejari registration article near the top.
    test_query = "Does a lease contract need to be registered anywhere to be enforceable?"
    print(f"\nSmoke test query: {test_query!r}\n")

    vector_hits = collection.query(query_texts=[test_query], n_results=3)
    print("Vector search — top 3:")
    for doc_id, meta, dist in zip(
        vector_hits["ids"][0], vector_hits["metadatas"][0], vector_hits["distances"][0]
    ):
        print(f"  chunk {doc_id} (distance={dist:.3f}): law={meta['law']} "
              f"article={meta['article_no']} current={meta['is_current']}")

    scores = bm25.get_scores(tokenize(test_query))
    top_bm25 = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:3]
    print("\nBM25 search — top 3:")
    for i in top_bm25:
        c = chunks[i]
        print(f"  chunk {c['chunk_id']} (score={scores[i]:.3f}): law={c['law']} "
              f"article={c['article_no']} current={c['is_current']}")


if __name__ == "__main__":
    main()
