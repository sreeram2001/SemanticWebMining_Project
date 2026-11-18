import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rerank as rr
import retrieval as rt

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def select_resumes(df, per_category=3, seed=573, ids=None):
    if ids is not None:
        wanted = {str(i) for i in ids}
        return df[df["ID"].astype(str).isin(wanted)]
    parts = [
        g.sample(n=min(per_category, len(g)), random_state=seed)
        for _, g in df.groupby("Category")
    ]
    return pd.concat(parts) if parts else df.head(0)


def run_rankers(queries, bm25, dense, tfidf, encode_query, k=100):
    if not (bm25.ids == dense.ids == tfidf.ids):
        raise ValueError("BM25, dense and TF-IDF indexes must share the same job order")
    runs = {m: {} for m in ("bm25", "dense", "hybrid", "rrf")}
    rids = list(queries)
    qvecs = encode_query([queries[r] for r in rids])
    for rid, qvec in zip(rids, qvecs):
        text = queries[rid]
        runs["bm25"][rid] = bm25.search(text, k)
        runs["dense"][rid] = dense.search(qvec, k)
        runs["hybrid"][rid] = rt.weighted_hybrid(
            tfidf.similarities(text), dense.similarities(qvec), dense.ids, k=k
        )
        runs["rrf"][rid] = rt.rrf([runs["bm25"][rid], runs["dense"][rid]], top=k)
    return runs


def build_dense_index(ids, texts, cache_path, model_name=DEFAULT_MODEL):
    if cache_path and os.path.exists(cache_path):
        idx = rt.DenseIndex.load(cache_path)
        if idx.ids == list(ids):
            return idx
        print("Embedding cache does not match the current postings; recomputing.")
    idx = rt.DenseIndex(ids, rt.encode_texts(texts, model_name))
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        idx.save(cache_path)
    return idx


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the baseline rankers for each resume.")
    p.add_argument("--processed", default="DATA/processed")
    p.add_argument("--out", default="DATA/runs/baselines.json")
    p.add_argument("--cache", default="DATA/embeddings/postings.npz")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--k", type=int, default=100, help="results kept per method")
    p.add_argument("--per-category", type=int, default=3)
    p.add_argument("--resume-ids", help="text file with one resume id per line")
    p.add_argument(
        "--limit-postings", type=int, help="use only the first N postings (smoke test)"
    )
    p.add_argument(
        "--rerank", action="store_true", help="cross-encode the top RRF results"
    )
    p.add_argument("--rerank-top", type=int, default=rr.DEFAULT_TOP_N)
    p.add_argument("--seed", type=int, default=573)
    a = p.parse_args(argv)

    posts, resumes = rt.load_corpus(a.processed)
    if a.limit_postings:
        posts = posts.head(a.limit_postings)
    ids = posts["job_id"].astype(str).tolist()
    texts = [rt.posting_text(r) for r in posts.to_dict("records")]
    print(f"{len(ids):,} postings")

    ids_filter = None
    if a.resume_ids:
        with open(a.resume_ids) as f:
            ids_filter = [line.strip() for line in f if line.strip()]
    chosen = select_resumes(resumes, a.per_category, a.seed, ids_filter)
    queries = {str(r["ID"]): r["Resume_clean"] for r in chosen.to_dict("records")}
    print(f"{len(queries)} resumes")

    bm25 = rt.BM25Index(ids, texts)
    tfidf = rt.TfidfIndex(ids, texts)
    dense = build_dense_index(ids, texts, a.cache, a.model)
    runs = run_rankers(
        queries, bm25, dense, tfidf, lambda t: rt.encode_texts(t, a.model), a.k
    )

    result = {
        "config": {
            "model": a.model,
            "k": a.k,
            "seed": a.seed,
            "rrf_k": rt.DEFAULT_RRF_K,
            "n_postings": len(ids),
            "n_resumes": len(queries),
        },
        "runs": runs,
    }
    if a.rerank:
        job_texts = dict(zip(ids, texts))
        reranker = rr.CrossEncoderReranker()
        result["rerank"] = rr.rerank_run(
            runs["rrf"], queries, job_texts, reranker, a.rerank_top
        )
        result["config"]["rerank_top"] = a.rerank_top
        result["config"]["cross_encoder"] = reranker.model_name

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
