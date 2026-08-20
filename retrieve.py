"""
retrieve.py — combines the two indexes from index.py into ONE ranked list
of relevant article chunks per question ("hybrid search").

WHY COMBINE TWO SEARCH METHODS INSTEAD OF PICKING ONE:
  - Vector search finds chunks by MEANING, so a question phrased
    differently from the law's own wording ("kicked out") still finds the
    right article ("eviction").
  - BM25 keyword search finds chunks by EXACT WORD MATCH, so a precise
    legal term (like "Ejari") or an article number doesn't get blurred
    together with similar-sounding but different text the way embeddings
    sometimes do.
Reciprocal Rank Fusion (RRF) merges both ranked lists into one: each chunk
earns 1/(K + rank) points from every list it appears in, so a chunk that
shows up in BOTH lists (even outside the very top of either) can outrank
something that's #1 in just one.

WHY WE FILTER OUT SUPERSEDED ARTICLES HERE, NOT EARLIER: a smoke test
showed vector search's #1 hit for a rent-registration question was the
OUTDATED pre-2008 wording of Article 4, not the current amended one —
both texts are legitimately "about Article 4", so nothing in the ranking
math itself knows one of them is overruled. hybrid_search drops
is_current=False chunks by default so Sakan can't accidentally cite
outdated law.
"""

import re

import chromadb

from index import (
    CHROMA_DIR,
    COLLECTION_NAME,
    build_bm25_index,
    build_vector_index,
    chunk_label,
    load_chunks,
    tokenize,
)

RRF_K = 60

# Legal questions often name an exact article ("Under Article 9, does...").
# Ranking alone can miss it: an amended article's CURRENT text can be
# genuinely dissimilar in meaning to a question about what it USED to say
# (found via eval question e01 — the current Article 9 no longer mentions
# the "two years" rule the old one had, so nothing about vector/keyword
# similarity points to it). When a query names an article explicitly, that
# article's current chunk is always included, regardless of how it ranks.
ARTICLE_MENTION = re.compile(r"\bArticle\s*\(?(\d+)\)?", re.IGNORECASE)


def reciprocal_rank_fusion(*ranked_id_lists: list[str], k_constant: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_constant + rank + 1)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)


class Retriever:
    """Loads both indexes once so repeated hybrid_search() calls are cheap
    — important since app.py calls this on every question asked.

    Outdated (superseded) articles are excluded from BOTH indexes' search
    space entirely, not filtered out after ranking. Found the hard way:
    the old and new versions of an amended article share almost the same
    label and often similar wording, so they directly compete for the same
    top-N ranking slots. A "fetch top N, then drop is_current=False"
    approach can lose the current version completely if the outdated twin
    happens to rank higher — e.g. a query about the old Article 9 "two-year
    rule" is naturally MORE similar to the outdated text (which is what
    that rule was), so it out-ranked the current Article 9 by a wide margin
    and pushed it out of the top 20 entirely. Excluding outdated chunks
    from the pool up front means the current version never has to compete
    with its own outdated twin for a ranking slot."""

    def __init__(self):
        all_chunks = load_chunks()
        self.chunks = [c for c in all_chunks if c["is_current"]]
        self.chunks_by_id = {str(c["chunk_id"]): c for c in self.chunks}
        # "Article N" in a question, with no law specified, means Law
        # 26/2007 — the base tenancy law everything else amends.
        self.article_26_2007 = {
            c["article_no"]: c for c in self.chunks if c["law"] == "26/2007"
        }
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        existing = {c.name for c in client.list_collections()}
        if COLLECTION_NAME in existing:
            self.collection = client.get_collection(COLLECTION_NAME)
        else:
            # chroma_db/ is a regenerable build artifact, gitignored on
            # purpose — a fresh deploy (e.g. Streamlit Community Cloud)
            # only runs `streamlit run app.py`, not ingest.py/index.py, so
            # the index has to be able to build itself on first launch
            # rather than assuming it's already there.
            self.collection = build_vector_index(all_chunks)
        self.bm25 = build_bm25_index(self.chunks)

    def hybrid_search(self, query: str, k: int = 6) -> list[dict]:
        vector_ranked = self.collection.query(
            query_texts=[query], n_results=k, where={"is_current": True}
        )["ids"][0]

        bm25_scores = self.bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(self.chunks)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ranked = [str(self.chunks[i]["chunk_id"]) for i in ranked_indices[:k]]

        fused_ids = reciprocal_rank_fusion(vector_ranked, bm25_ranked)
        results = [self.chunks_by_id[chunk_id] for chunk_id in fused_ids[:k]]

        match = ARTICLE_MENTION.search(query)
        if match:
            named_chunk = self.article_26_2007.get(int(match.group(1)))
            if named_chunk and named_chunk not in results:
                results = [named_chunk] + results[:k - 1]
        return results


def main():
    retriever = Retriever()
    demo_questions = [
        "Does a lease contract need to be registered anywhere to be enforceable?",
        "What is Ejari?",
        "By what percentage can rent increase if it's 25% below the average rental value?",
        "How much notice for eviction if the landlord wants to sell the property?",
    ]
    for question in demo_questions:
        print(f"\nQ: {question}")
        for chunk in retriever.hybrid_search(question, k=3):
            print(f"  [{chunk_label(chunk)}] {chunk['text'][:120]}...")


if __name__ == "__main__":
    main()
