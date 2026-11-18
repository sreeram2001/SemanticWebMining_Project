"""Step 7: S = beta1*CE + beta2*G - beta3*P.

CE  cross-encoder score, min-max normalized.
G   skill coverage (1 exact, lambda_sub/lambda_cat for related).
P   degree/years conflicts; unknown is not penalized.
"""

import math
from collections import Counter, OrderedDict

from core.config import resolve
from linking.entity_linking import NIL, Gazetteer, TextSearch
from kg.graph import DEGREE_RANK
from rank.retrieval import BM25, Dense, TfidfCosine, WeightedHybrid, order, posting_text, rrf, tokenize, top_k
from extract.rule_extractor import years_from_dates

METHODS = ["bm25", "dense", "weighted_hybrid", "rrf", "full", "full_no_graph"]
BASELINES = ["bm25", "dense", "weighted_hybrid", "rrf"]


# ------------------------------------------------------------------ rerankers

def minmax(xs):
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    return [0.0] * len(xs) if hi - lo < 1e-12 else [(x - lo) / (hi - lo) for x in xs]


class LexicalReranker:
    """Cross-encoder stand-in: 1-2 gram TF-IDF cosine."""
    name = "lexical (stand-in)"

    def __init__(self, corpus):
        grams = [self._grams(t) for t in corpus]
        df = Counter(g for c in grams for g in set(c))
        n = len(corpus)
        self.idf = {g: math.log((1 + n) / (1 + d)) + 1 for g, d in df.items()}

    @staticmethod
    def _grams(text):
        t = tokenize(text)
        return t + [f"{a} {b}" for a, b in zip(t, t[1:])]

    def _vec(self, text):
        tf = Counter(self._grams(text))
        v = {g: (1 + math.log(c)) * self.idf.get(g, 1.0) for g, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norm for g, x in v.items()}

    def score(self, query, docs):
        q = self._vec(query)
        return [sum(q.get(g, 0.0) * v for g, v in self._vec(d).items()) for d in docs]


class CrossEncoderReranker:
    def __init__(self, model_name):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=512)
        self.name = model_name

    def score(self, query, docs):
        return [float(x) for x in self.model.predict([(query, d) for d in docs])] if docs else []


def make_reranker(model_name, corpus, strict=False):
    if model_name:
        try:
            return CrossEncoderReranker(model_name)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"cross-encoder '{model_name}' failed to load: {exc}") from exc
            print(f"[ranking] cross-encoder unavailable ({type(exc).__name__}); using the lexical stand-in")
    return LexicalReranker(corpus)


# ------------------------------------------------------------------ ranker

class Ranker:
    """Ranks postings for a resume with any method in METHODS (weight-independent work cached)."""

    def __init__(self, graph, cfg, use_models=True, strict=False, first_stage_cache=256):
        self.g = graph
        self.index = graph.index
        self.cfg = cfg
        self.job_ids = graph.doc_ids("posting")
        if not self.job_ids:
            raise ValueError("the graph has no job postings; run extract and build first")
        self._pos = {j: i for i, j in enumerate(self.job_ids)}
        # title-weighted text for retrieval
        self._job_text = {j: posting_text(graph.meta.get(j, {}).get("title"), graph.texts[j]) for j in self.job_ids}
        texts = [self._job_text[j] for j in self.job_ids]
        r = cfg["retrieval"]
        self.bm25 = BM25(texts)
        self.tfidf = TfidfCosine(texts)
        self.dense = Dense(texts, r.get("dense_model") if use_models else None, strict=strict,
                           ids=self.job_ids, cache=resolve(r.get("emb_cache")))
        self.hybrid = WeightedHybrid(self.tfidf, self.dense, r.get("hybrid_weight", 0.5))
        corpus = texts + [graph.texts[d] for d in graph.doc_ids("resume")]
        self.reranker = make_reranker(cfg["rerank"].get("model") if use_models else None, corpus, strict)
        self._first = OrderedDict()                 # LRU cache of first-stage scores
        self._first_max = first_stage_cache
        self._ce, self._search, self._years, self._base = {}, {}, {}, {}
        self.miss_log = {}

    @property
    def gazetteer(self):
        """Lazy skill-name finder for explanations."""
        if getattr(self, "_gazetteer", None) is None:
            self._gazetteer = Gazetteer(self.index)
        return self._gazetteer

    @property
    def backends(self):
        return {"dense": self.dense.backend, "reranker": self.reranker.name,
                "linker": self.g.linker.backend, "query": self.cfg["retrieval"]["query"]}

    def _scoring(self, scoring):
        sc = {**self.cfg["scoring"], **(scoring or {})}
        if not 0 < sc["lambda_cat"] < sc["lambda_sub"] < 1:
            raise ValueError(f"need 0 < lambda_cat < lambda_sub < 1, got {sc['lambda_cat']}, {sc['lambda_sub']}")
        return sc

    # -------------------------------------------------------------- first stage
    def query(self, resume_id):
        """Retrieval query: linked skills, else resume text."""
        if self.cfg["retrieval"].get("query", "skills") == "skills":
            names = [self.index.display_name(s) for s in self.g.linked_skills(resume_id)]
            names += [f["name"] for f in self.g.facts.get(resume_id, {}).get("skills", []) if f["skill_id"] == NIL]
            if names:
                return " ; ".join(dict.fromkeys(names))
        return self.g.texts[resume_id]

    def first_stage(self, resume_id):
        if resume_id in self._first:
            self._first.move_to_end(resume_id)
        else:
            q = self.query(resume_id)
            bm, de = self.bm25.scores(q), self.dense.scores(q)
            tf = self.tfidf.scores(q)
            w = self.hybrid.weight
            self._first[resume_id] = {"bm25": bm, "dense": de,
                                      "weighted_hybrid": [w * a + (1 - w) * b for a, b in zip(tf, de)]}
            if len(self._first) > self._first_max:
                self._first.popitem(last=False)
        return self._first[resume_id]

    def ce_scores(self, resume_id, job_ids):
        missing = [j for j in job_ids if (resume_id, j) not in self._ce]
        if missing:
            vals = self.reranker.score(self.g.texts[resume_id], [self._job_text[j] for j in missing])
            self._ce.update({(resume_id, j): v for j, v in zip(missing, vals)})
        return [self._ce[(resume_id, j)] for j in job_ids]

    # -------------------------------------------------------------- ranking
    def rank(self, resume_id, method="full", scoring=None):
        """All jobs for a resume, best first."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if resume_id not in self.g.texts:
            raise KeyError(f"resume {resume_id} is not in the graph")
        sc = self._scoring(scoring)
        fs = self.first_stage(resume_id)
        pos = self._pos
        if method in ("bm25", "dense", "weighted_hybrid"):
            s = fs[method]
            return self._numbered([{"job_id": j, "score": s[pos[j]]} for j in order(s, self.job_ids)])

        k1 = self.cfg["retrieval"]["first_stage_k"]
        fused = rrf([top_k(fs["bm25"], self.job_ids, k1), top_k(fs["dense"], self.job_ids, k1)],
                    k=self.cfg["retrieval"]["rrf_k"])
        fused_order = sorted(fused, key=lambda j: (-fused[j], pos[j]))
        fused_order += [j for j in self.job_ids if j not in fused]
        if method == "rrf":
            return self._numbered([{"job_id": j, "score": fused.get(j, 0.0)} for j in fused_order])

        if method == "full_no_graph":
            sc["beta_graph"] = 0.0
        top = fused_order[: self.cfg["rerank"]["top_n"]]
        ce = minmax(self.ce_scores(resume_id, top))
        rows = []
        for i, (j, c) in enumerate(zip(top, ce)):
            G, P = self.graph_score(resume_id, j, sc)
            rows.append({"job_id": j, "score": sc["beta_ce"] * c + sc["beta_graph"] * G - sc["beta_penalty"] * P,
                         "ce": c, "G": G, "P": P, "_i": i})
        rows.sort(key=lambda x: (-x["score"], x.pop("_i")))
        floor = min((x["score"] for x in rows), default=0.0) - 1.0
        rows += [{"job_id": j, "score": floor - i * 1e-6} for i, j in enumerate(fused_order[len(top):])]
        return self._numbered(rows)

    @staticmethod
    def _numbered(rows):
        for i, row in enumerate(rows, 1):
            row["rank"] = i
        return rows

    # -------------------------------------------------------------- graph score
    def _text_search(self, resume_id):
        if resume_id not in self._search:
            self._search[resume_id] = TextSearch(self.g.texts[resume_id], self.index)
        return self._search[resume_id]

    def resume_years(self, resume_id):
        """Years from dates, else summed durations, else None."""
        if resume_id not in self._years:
            years = years_from_dates(self.g.texts[resume_id], self.cfg["scoring"].get("current_date", "2021-12"))
            if years is None:
                months = [f.get("duration_months") for f in self.g.facts.get(resume_id, {}).get("experience", [])]
                months = [m for m in months if isinstance(m, int) and not isinstance(m, bool)]
                years = round(sum(months) / 12, 2) if months else None
            self._years[resume_id] = years
        return self._years[resume_id]

    def _match_base(self, resume_id, job_id):
        """Cached per-requirement status for a pair."""
        key = (resume_id, job_id)
        base = self._base.get(key)
        if base is not None:
            return base
        have = self.g.linked_skills(resume_id)
        fuzzy = self.cfg["scoring"].get("fuzzy_threshold", 0.9)
        reqs = []
        for req in self.g.skill_requirements(job_id):
            sid = req["skill_id"]
            item = {"skill_id": sid, "mention": req["name"], "importance": req["importance"],
                    "requirement_span": _span(req), "relation": "none", "evidence": None}
            if sid == NIL:
                item.update(status="unresolved", name=req["name"])
            else:
                item["name"] = self.index.display_name(sid)
                if sid in have:
                    item.update(status="met", relation="same", evidence=_span(have[sid]))
                else:
                    hit = self._text_search(resume_id).find(sid, fuzzy)
                    if hit:
                        item.update(status="recovered", relation="same",
                                    evidence={"text": hit["text"], "start": hit["start"], "end": hit["end"],
                                              "how": hit["how"]})
                        self.miss_log.setdefault((resume_id, sid), {
                            "resume_id": resume_id, "skill_id": sid, "skill": item["name"],
                            "found": hit["text"], "how": hit["how"]})
                    else:
                        best = None
                        for tid in have:
                            rel = self.index.relation(sid, tid)
                            if rel == "subcategory":
                                best = (rel, tid)
                                break
                            if rel == "category" and best is None:
                                best = (rel, tid)
                        if best:
                            item.update(status="related", relation=best[0],
                                        related_skill=self.index.display_name(best[1]), evidence=_span(have[best[1]]))
                        else:
                            item["status"] = "not_found"
            reqs.append(item)
        base = {"skills": reqs, "hard": self._hard(resume_id, job_id)}
        self._base[key] = base
        return base

    def graph_score(self, resume_id, job_id, sc):
        """(G, P) for one pair."""
        base = self._match_base(resume_id, job_id)
        lam = {"same": 1.0, "subcategory": sc["lambda_sub"], "category": sc["lambda_cat"], "none": 0.0}
        num = den = 0.0
        for x in base["skills"]:
            if x["status"] == "unresolved":
                continue
            w = sc["w_required"] if x["importance"] == "required" else sc["w_preferred"]
            den += w
            num += w * lam[x["relation"]]
        req = [h for h in base["hard"] if h["importance"] == "required"]
        return (num / den if den else 0.0), (sum(h["shortfall"] for h in req) / len(req) if req else 0.0)

    def match(self, resume_id, job_id, scoring=None):
        """Per-requirement comparison for S and explanations."""
        sc = self._scoring(scoring)
        base = self._match_base(resume_id, job_id)
        lam = {"same": 1.0, "subcategory": sc["lambda_sub"], "category": sc["lambda_cat"], "none": 0.0}
        skills = []
        for x in base["skills"]:
            w = sc["w_required"] if x["importance"] == "required" else sc["w_preferred"]
            skills.append({**x, "weight": w, "credit": 0.0 if x["status"] == "unresolved" else lam[x["relation"]]})
        G, P = self.graph_score(resume_id, job_id, sc)
        return {"resume_id": resume_id, "job_id": job_id, "skills": skills, "hard": base["hard"], "G": G, "P": P,
                "insufficient_evidence": not any(x["status"] != "unresolved" for x in skills)}

    def _hard(self, resume_id, job_id):
        checks = []
        degs = [d for d in self.g.degrees(job_id) if d.get("level") in DEGREE_RANK]
        if degs:
            required = [d for d in degs if d["importance"] == "required"] or degs
            need = min(required, key=lambda d: DEGREE_RANK[d["level"]])
            mine = [d for d in self.g.degrees(resume_id) if d.get("level") in DEGREE_RANK]
            c = {"kind": "degree", "importance": need["importance"], "required": need["level"],
                 "requirement_span": _span(need), "shortfall": 0.0}
            if not mine:
                c.update(status="unknown", have=None)
            else:
                best = max(mine, key=lambda d: DEGREE_RANK[d["level"]])
                ok = DEGREE_RANK[best["level"]] >= DEGREE_RANK[need["level"]]
                c.update(status="met" if ok else "conflict", have=best["level"], evidence_span=_span(best),
                         shortfall=0.0 if ok else 1.0)
            checks.append(c)
        req = self.g.min_years(job_id)
        if req:
            years = self.resume_years(resume_id)
            need = float(req["min_years"])
            c = {"kind": "experience", "importance": req["importance"], "required": need,
                 "requirement_span": _span(req), "shortfall": 0.0}
            if years is None:
                c.update(status="unknown", have=None)
            else:
                gap = max(0.0, need - years) / need if need > 0 else 0.0
                c.update(status="met" if gap == 0 else "conflict", have=years, shortfall=round(min(1.0, gap), 4))
            checks.append(c)
        return checks


def _span(fact):
    return {"text": fact["source_span"], "start": fact["start"], "end": fact["end"]}
