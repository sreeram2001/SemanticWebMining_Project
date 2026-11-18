import math
import os
import re
from collections import Counter, defaultdict

DEFAULT_RRF_K = 60

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was were will with you your we our they their".split()
)


def tokenize(text):
    if not text:
        return []
    toks = (t.rstrip(".-") for t in _TOKEN.findall(str(text).lower()))
    return [t for t in toks if t and t not in STOPWORDS]


class BM25Index:
    def __init__(self, ids, texts, k1=1.5, b=0.75):
        ids, texts = list(ids), list(texts)
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        self.ids, self.k1, self.b = ids, k1, b
        self.postings = defaultdict(list)
        self.doc_len = []
        for i, text in enumerate(texts):
            toks = tokenize(text)
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        n = len(ids)
        self.avg_len = (sum(self.doc_len) / n) if n else 0.0
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        }

    def scores(self, query):
        out = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = 1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1)
                out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return out

    def search(self, query, k=100):
        ranked = sorted(self.scores(query).items(), key=lambda x: (-x[1], x[0]))
        return [(self.ids[i], s) for i, s in ranked[:k]]


def encode_texts(
    texts,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=64,
    model=None,
):
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype("float32")


class DenseIndex:
    def __init__(self, ids, embeddings):
        import numpy as np

        self.np = np
        self.ids = list(ids)
        self.emb = np.asarray(embeddings, dtype="float32")
        if len(self.ids) != len(self.emb):
            raise ValueError("ids and embeddings must have the same length")

    def similarities(self, query_vec):
        return self.emb @ self.np.asarray(query_vec, dtype="float32")

    def search(self, query_vec, k=100):
        sims = self.similarities(query_vec)
        k = min(k, len(sims))
        if k == 0:
            return []
        top = self.np.argpartition(-sims, k - 1)[:k]
        top = top[self.np.lexsort((top, -sims[top]))]
        return [(self.ids[i], float(sims[i])) for i in top]

    def save(self, path):
        self.np.savez_compressed(
            path, ids=self.np.array(self.ids, dtype=object), emb=self.emb
        )

    @classmethod
    def load(cls, path):
        import numpy as np

        data = np.load(path, allow_pickle=True)
        return cls(data["ids"].tolist(), data["emb"])


class TfidfIndex:
    def __init__(self, ids, texts, **vectorizer_args):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.ids = list(ids)
        args = dict(stop_words="english", sublinear_tf=True, max_features=200_000)
        args.update(vectorizer_args)
        self.vectorizer = TfidfVectorizer(**args)
        self.matrix = self.vectorizer.fit_transform(texts)

    def similarities(self, query):
        q = self.vectorizer.transform([query])
        return (self.matrix @ q.T).toarray().ravel()


def weighted_hybrid(tfidf_sims, dense_sims, ids, w_tfidf=0.5, k=100):
    if len(tfidf_sims) != len(dense_sims) or len(ids) != len(dense_sims):
        raise ValueError("score vectors and ids must have the same length")
    combined = [
        w_tfidf * float(t) + (1 - w_tfidf) * float(d)
        for t, d in zip(tfidf_sims, dense_sims)
    ]
    order = sorted(range(len(ids)), key=lambda i: (-combined[i], i))[:k]
    return [(ids[i], combined[i]) for i in order]


def rrf(rankings, k=DEFAULT_RRF_K, top=None):
    fused, best, first = defaultdict(float), {}, {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            doc = item[0] if isinstance(item, (tuple, list)) else item
            fused[doc] += 1.0 / (k + rank)
            best[doc] = min(best.get(doc, rank), rank)
            first.setdefault(doc, len(first))
    ordered = sorted(fused, key=lambda d: (-fused[d], best[d], first[d]))
    return [(d, fused[d]) for d in ordered[:top]]


def precision_at_k(ranked_ids, relevance, k=10, threshold=1):
    if k <= 0:
        raise ValueError("k must be positive")
    hits = sum(1 for d in ranked_ids[:k] if relevance.get(d, 0) >= threshold)
    return hits / k


def dcg_at_k(grades, k):
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg_at_k(ranked_ids, relevance, k=10):
    ideal = sorted(relevance.values(), reverse=True)
    best = dcg_at_k(ideal, k)
    if best == 0:
        return 0.0
    return dcg_at_k([relevance.get(d, 0) for d in ranked_ids], k) / best


def evaluate(run, qrels, k=10):
    scored = [r for r in run if r in qrels]
    if not scored:
        return {"n": 0, f"P@{k}": 0.0, f"nDCG@{k}": 0.0}
    return {
        "n": len(scored),
        f"P@{k}": sum(precision_at_k(run[r], qrels[r], k) for r in scored)
        / len(scored),
        f"nDCG@{k}": sum(ndcg_at_k(run[r], qrels[r], k) for r in scored) / len(scored),
    }


def posting_text(row):
    title = str(row.get("title") or "")
    return f"{title}. {title}. {row.get('description') or ''}"


def load_corpus(processed_dir="DATA/processed"):
    import pandas as pd

    posts = pd.read_csv(os.path.join(processed_dir, "postings_cleaned.csv"))
    resumes = pd.read_csv(os.path.join(processed_dir, "resumes_cleaned.csv"))
    return posts, resumes


PRIOR_SKILLS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prior_skills.json")


def load_prior_skills(path=PRIOR_SKILLS_PATH):
    import json
    with open(path, encoding="utf-8") as f:
        skills = json.load(f)["skills"]
    return sorted(set(skills), key=len, reverse=True)


def prior_extract_skills(text, skills):
    low = text.lower()
    return sorted({s for s in skills if re.search(r"\b" + re.escape(s) + r"\b", low)})


def prior_query(text, skills):
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    return " ".join(prior_extract_skills(clean, skills)) + " " + clean


def prior_job_text(row):
    return f"{row.get('title') or ''} {row.get('description') or ''}"


def prior_tfidf(ids, texts):
    return TfidfIndex(ids, texts, sublinear_tf=False, max_features=None)
