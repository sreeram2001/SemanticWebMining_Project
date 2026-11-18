"""Step 6: retrieval baselines (BM25, dense, TF-IDF, hybrid, RRF); scores(query) -> floats."""

import hashlib
import math
import os
import re
from collections import Counter, defaultdict

_TOKEN = re.compile(r"[a-z0-9+#]+(?:\.[a-z0-9]+)*")
STOP = set("""a an and are as at be by for from has have in is it its of on or our that the their this to
we will with you your they them who what which while all any can may must should would also more most
other such than then there these those into out over about including within across per""".split())


def tokenize(text):
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in STOP]


class BM25:
    name = "bm25"

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        toks = [tokenize(d) for d in docs]
        self.n = len(docs)
        self.lens = [len(t) for t in toks]
        self.avgdl = (sum(self.lens) / self.n) if self.n else 0.0
        self.postings = defaultdict(list)
        for i, t in enumerate(toks):
            for w, f in Counter(t).items():
                self.postings[w].append((i, f))
        self.idf = {w: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.postings.items()}

    def scores(self, query):
        out = [0.0] * self.n
        for w, qf in Counter(tokenize(query)).items():
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, f in self.postings[w]:
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avgdl or 1))
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


class TfidfCosine:
    name = "tfidf"

    def __init__(self, docs):
        toks = [tokenize(d) for d in docs]
        self.n = len(docs)
        df = Counter(w for t in toks for w in set(t))
        self.idf = {w: math.log((1 + self.n) / (1 + c)) + 1 for w, c in df.items()}
        self.postings = defaultdict(list)
        for i, t in enumerate(toks):
            vec = self._vec(t)
            for w, v in vec.items():
                self.postings[w].append((i, v))

    def _vec(self, toks):
        tf = Counter(toks)
        v = {w: (1 + math.log(c)) * self.idf[w] for w, c in tf.items() if w in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def scores(self, query):
        out = [0.0] * self.n
        for w, qv in self._vec(tokenize(query)).items():
            for i, dv in self.postings[w]:
                out[i] += qv * dv
        return out


class Dense:
    name = "dense"

    def __init__(self, docs, model_name=None, batch_size=64, strict=False, ids=None, cache=None):
        self.backend = None
        self._model = None
        if model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
                self.emb = self._cached(cache, model_name, ids, docs)
                if self.emb is None:
                    import numpy as np                   # float32 to match the cache
                    self.emb = np.asarray(self._model.encode(list(docs), normalize_embeddings=True,
                                                             batch_size=batch_size), dtype="float32")
                    self._save(cache, model_name, ids, docs)
                self.backend = model_name
            except Exception as exc:
                if strict:
                    raise RuntimeError(f"dense retrieval model '{model_name}' failed to load: {exc}") from exc
                print(f"[retrieval] sentence-transformers unavailable ({type(exc).__name__}); dense uses a stand-in")
        if self._model is None:
            try:
                from sklearn.decomposition import TruncatedSVD
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.preprocessing import normalize
                self._vec = TfidfVectorizer(sublinear_tf=True, stop_words="english")
                m = self._vec.fit_transform(docs)
                k = max(2, min(256, m.shape[0] - 1, m.shape[1] - 1))
                self._svd = TruncatedSVD(n_components=k, random_state=0)
                self.emb = normalize(self._svd.fit_transform(m))
                self._normalize = normalize
                self.backend = f"tfidf-lsa-{k} (stand-in)"
            except ImportError:
                self._fallback = TfidfCosine(docs)
                self.backend = "tfidf (stand-in)"

    @staticmethod
    def _key(model_name, ids, docs):
        h = hashlib.sha256(model_name.encode())
        for i, d in zip(ids or range(len(docs)), docs):
            h.update(f"{i}\x00{d}\x01".encode("utf-8", "ignore"))
        return h.hexdigest()[:16]

    def _cached(self, path, model_name, ids, docs):
        """Cached posting embeddings if model and corpus match."""
        if not path or not os.path.exists(path):
            return None
        try:
            import numpy as np
            data = np.load(path, allow_pickle=False)
            if str(data["key"]) == self._key(model_name, ids, docs) and len(data["emb"]) == len(docs):
                return data["emb"]
        except (OSError, ValueError, KeyError):
            pass
        print("[retrieval] embedding cache does not match the postings; re-embedding")
        return None

    def _save(self, path, model_name, ids, docs):
        if not path:
            return
        import numpy as np
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, emb=np.asarray(self.emb, dtype="float32"), key=self._key(model_name, ids, docs))
        os.replace(tmp, path)

    def scores(self, query):
        if self._model is not None:
            q = self._model.encode([query], normalize_embeddings=True)[0]
            return [float(x) for x in self.emb @ q]
        if hasattr(self, "_svd"):
            q = self._normalize(self._svd.transform(self._vec.transform([query])))[0]
            return [float(x) for x in self.emb @ q]
        return self._fallback.scores(query)


class WeightedHybrid:
    """Equal-weight TF-IDF + SBERT (Proposal 3)."""
    name = "weighted_hybrid"

    def __init__(self, tfidf, dense, weight=0.5):
        self.tfidf, self.dense, self.weight = tfidf, dense, weight

    def scores(self, query):
        a, b = self.tfidf.scores(query), self.dense.scores(query)
        return [self.weight * x + (1 - self.weight) * y for x, y in zip(a, b)]


def order(scores, ids):
    """ids by descending score (stable)."""
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-scores[i], i))]


def top_k(scores, ids, k):
    """Top k via argpartition."""
    if k >= len(ids):
        return order(scores, ids)
    try:
        import numpy as np
    except ImportError:
        return order(scores, ids)[:k]
    s = np.asarray(scores, dtype=float)
    kth = np.partition(-s, k - 1)[k - 1]             # keep ties at k
    cand = np.flatnonzero(-s <= kth)
    cand = cand[np.lexsort((cand, -s[cand]))][:k]
    return [ids[i] for i in cand]


def posting_text(title, description):
    """Posting text: title twice, then description."""
    title = (title or "").strip()
    return f"{title}. {title}. {description or ''}" if title else (description or "")


def rrf(rankings, k=60):
    """RRF(j) = sum 1 / (k + rank)."""
    out = defaultdict(float)
    for ranked in rankings:
        for rank, doc_id in enumerate(ranked, start=1):
            out[doc_id] += 1.0 / (k + rank)
    return dict(out)
