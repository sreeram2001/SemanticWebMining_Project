"""Step 4b: BLINK-style skill linking to Lightcast IDs or NIL.

alias -> candidates (char 3-grams, bi-encoder) -> LLM / cross-encoder / threshold.
"""

import csv
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from core.config import read_json_or, write_json_atomic
from linking.skill_linking import build_name_index, by_id, normalize_name, strip_qualifier

NIL = "NIL"

_TOKEN = re.compile(r"\.?\w[\w+#]*(?:[./\-&']\w[\w+#]*)*[+#]*")
_WORD = re.compile(r"[a-z0-9+#]+")
_STOP = {"a", "an", "and", "the", "of", "for", "in", "on", "to", "with", "or", "as", "at", "by", "is",
         "are", "be", "we", "you", "our", "your", "experience", "skills", "using", "use", "including"}
MAX_NGRAM = 6


def compact(text):
    """'Postgre SQL' -> 'postgresql', 'React.js' -> 'reactjs'; keeps + and #."""
    return re.sub(r"[\s.\-_/]+", "", normalize_name(text))


def tokens_with_offsets(text):
    return [(m.start(), m.end()) for m in _TOKEN.finditer(text or "")]


def _grams(text, n=3):
    s = f" {text} "
    return Counter(s[i:i + n] for i in range(max(1, len(s) - n + 1)))


# ------------------------------------------------------------------ index

class SkillIndex:
    """Lightcast skills and their surface forms."""

    def __init__(self, skills, aliases=None):
        self.skills = by_id(skills)
        self.exact = {k: list(v) for k, v in build_name_index(skills).items()}
        self.surface = defaultdict(set)          # skill_id -> normalized surface forms
        for s in skills:
            self.surface[s["id"]].add(normalize_name(s["name"]))
            bare = normalize_name(strip_qualifier(s["name"]))
            if bare:
                self.surface[s["id"]].add(bare)

        self.aliases_used = 0
        for alias, target in (aliases or {}).items():
            if alias.startswith("_"):
                continue
            ids = self.exact.get(normalize_name(target)) or []
            if len(ids) != 1:
                continue                         # unknown or ambiguous target
            key = normalize_name(alias)
            if ids[0] not in self.exact.setdefault(key, []):
                self.exact[key].insert(0, ids[0])   # explicit alias wins
            self.surface[ids[0]].add(key)
            self.aliases_used += 1

        self.compact = defaultdict(list)
        for key, ids in self.exact.items():
            for sid in ids:
                if sid not in self.compact[compact(key)]:
                    self.compact[compact(key)].append(sid)

        # char 3-gram TF-IDF inverted index
        self.forms = [(form, sid) for sid, forms in self.surface.items() for form in sorted(forms)]
        grams = [_grams(f) for f, _ in self.forms]
        df = Counter(g for c in grams for g in c)
        n = len(self.forms)
        self.idf = {g: math.log((n + 1) / (d + 1)) + 1 for g, d in df.items()}
        self.inverted = defaultdict(list)
        for i, c in enumerate(grams):
            w = {g: tf * self.idf[g] for g, tf in c.items()}
            norm = math.sqrt(sum(v * v for v in w.values())) or 1.0
            for g, v in w.items():
                self.inverted[g].append((i, v / norm))

    @classmethod
    def load(cls, skills_path, aliases_path=None):
        from linking.skill_linking import load_skills
        aliases = None
        if aliases_path and os.path.exists(aliases_path):
            with open(aliases_path, encoding="utf-8") as f:
                aliases = json.load(f)
        return cls(load_skills(skills_path), aliases)

    def lookup(self, mention):
        """IDs for an exact/punctuation-free hit, else []."""
        key = normalize_name(mention)
        ids = self.exact.get(key) or self.exact.get(normalize_name(strip_qualifier(mention)))
        return list(ids or self.compact.get(compact(mention), []))

    def char_candidates(self, mention, top_k):
        # unseen 3-grams get max idf
        unseen = math.log(len(self.forms) + 1) + 1
        q = {g: tf * self.idf.get(g, unseen) for g, tf in _grams(normalize_name(mention)).items()}
        norm = math.sqrt(sum(v * v for v in q.values())) or 1.0
        scores = defaultdict(float)
        for g, v in q.items():
            for i, w in self.inverted.get(g, ()):
                scores[i] += v / norm * w
        best = {}
        for i, s in scores.items():
            sid = self.forms[i][1]
            if s > best.get(sid, 0.0):
                best[sid] = s
        return sorted(best.items(), key=lambda x: (-x[1], x[0]))[:top_k]

    def name(self, skill_id):
        s = self.skills.get(skill_id)
        return s["name"] if s else skill_id

    def display_name(self, skill_id):
        return strip_qualifier(self.name(skill_id))

    def relation(self, a, b):
        """'same' | 'subcategory' | 'category' | 'none' (Lightcast hierarchy)."""
        if a == b:
            return "same"
        sa, sb = self.skills.get(a), self.skills.get(b)
        if not sa or not sb:
            return "none"
        if sa.get("subcategory") and sa.get("subcategory") == sb.get("subcategory"):
            return "subcategory"
        if sa.get("category") and sa.get("category") == sb.get("category"):
            return "category"
        return "none"


# ------------------------------------------------------------------ linker

class SkillLinker:
    """Mention -> Lightcast ID or NIL.

    Without an alias hit: >= llm_accept accept, < llm_reject NIL, else LLM
    (or cross-encoder / min_score). Cached per mention; strict=True disables fallbacks.
    """

    def __init__(self, index, top_k=5, min_score=0.65, margin=0.0, dense_model=None,
                 cross_encoder=None, llm_client=None, llm_model=None, alias_only=False,
                 llm_accept=0.9, llm_reject=0.3, cache_path=None, emb_cache=None, strict=False):
        self.index = index
        self.top_k = top_k
        self.min_score = min_score
        self.margin = margin
        self.alias_only = alias_only
        self.llm_client, self.llm_model = llm_client, llm_model
        self.llm_accept, self.llm_reject = llm_accept, llm_reject
        self.cache_path = cache_path
        self.llm_calls = 0
        self._cache = {}
        self._llm_cache = read_json_or(cache_path, {}, "linking LLM cache")
        self._unsaved = 0
        self._dense = self._ce = None
        if dense_model and not alias_only:
            try:
                from sentence_transformers import SentenceTransformer
                self._dense = SentenceTransformer(dense_model)
                self._ids = list(index.skills)
                self._emb = self._skill_embeddings(dense_model, emb_cache)
            except Exception as exc:
                if strict:
                    raise RuntimeError(f"dense linker model '{dense_model}' failed to load: {exc}") from exc
                print(f"[linking] dense retrieval off ({type(exc).__name__}); using character n-grams")
                self._dense = None
        if cross_encoder and not alias_only:
            try:
                from sentence_transformers import CrossEncoder
                self._ce = CrossEncoder(cross_encoder)
            except Exception as exc:
                if strict:
                    raise RuntimeError(f"linking cross-encoder '{cross_encoder}' failed to load: {exc}") from exc
                print(f"[linking] cross-encoder off ({type(exc).__name__})")

    def _skill_embeddings(self, model_name, cache):
        """Embed all skills once; reuse a matching .npy cache."""
        import numpy as np
        if cache:
            meta = cache + ".json"
            key = {"model": model_name, "n": len(self._ids), "first": self._ids[:3], "last": self._ids[-3:]}
            if os.path.exists(cache) and read_json_or(meta, None, "embedding cache key") == key:
                try:
                    emb = np.load(cache)
                    if emb.shape[0] == len(self._ids):
                        return emb
                except (OSError, ValueError):
                    print("[linking] embedding cache unreadable; re-embedding")
        docs = [f"{s['name']}: {s.get('description') or ''}" for s in self.index.skills.values()]
        emb = self._dense.encode(docs, normalize_embeddings=True, batch_size=256)
        if cache:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            tmp = cache + ".tmp.npy"
            np.save(tmp, emb)
            os.replace(tmp, cache)
            write_json_atomic(cache + ".json", key)
        return emb

    @property
    def backend(self):
        if self.alias_only:
            return "alias-only"
        parts = ["char3gram"] + (["bi-encoder"] if self._dense else [])
        parts.append(f"llm:{self.llm_model}" if self.llm_client else "cross-encoder" if self._ce else "threshold")
        return "+".join(parts)

    def set_params(self, **params):
        """Update thresholds; clears decisions, keeps LLM answers."""
        for k, v in params.items():
            setattr(self, k, v)
        self._cache = {}

    def save_cache(self):
        if self.cache_path and self._llm_cache:
            write_json_atomic(self.cache_path, self._llm_cache, indent=0)
            self._unsaved = 0

    def link(self, mention, context=""):
        ids = self.index.lookup(mention)
        if len(ids) == 1:
            return self._result(mention, ids[0], 1.0, "alias", [(ids[0], 1.0)])
        if len(ids) > 1:
            sid = self._disambiguate(ids, context)
            return self._result(mention, sid, 1.0, "alias_ambiguous", [(i, 1.0) for i in ids])
        if self.alias_only:
            return self._result(mention, NIL, 0.0, "nil", [])

        # one decision per unique mention
        cache_key = normalize_name(mention)
        if cache_key in self._cache:
            return dict(self._cache[cache_key], mention=mention)

        scores = dict(self.index.char_candidates(mention, self.top_k * 2))
        if self._dense is not None:
            q = self._dense.encode([f"{mention} | {context}"], normalize_embeddings=True)[0]
            sims = self._emb @ q
            for i in sims.argsort()[::-1][: self.top_k]:
                sid = self._ids[int(i)]
                scores[sid] = max(scores.get(sid, 0.0), float(sims[int(i)]))
        cands = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[: self.top_k]
        best = cands[0][1] if cands else 0.0
        if not cands:
            out = self._result(mention, NIL, 0.0, "nil", [])
        elif self.llm_client is not None and best >= self.llm_accept:
            out = self._result(mention, cands[0][0], best, "confident", cands)
        elif self.llm_client is not None and best < self.llm_reject:
            out = self._result(mention, NIL, best, "nil", cands)
        elif self.llm_client is not None:
            out = self._llm_decide(mention, context, cands)
        elif self._ce is not None:
            out = self._ce_decide(mention, context, cands)
        else:
            second = cands[1][1] if len(cands) > 1 else 0.0
            ok = best >= self.min_score and best - second >= self.margin
            out = self._result(mention, cands[0][0] if ok else NIL, best, "char" if ok else "nil", cands)
        self._cache[cache_key] = out
        return out

    def _result(self, mention, sid, score, method, cands):
        return {"mention": mention, "skill_id": sid,
                "name": self.index.name(sid) if sid != NIL else None,
                "score": round(float(score), 4), "method": method,
                "candidates": [(c, round(float(s), 4)) for c, s in cands]}

    def _disambiguate(self, ids, context):
        """Disambiguate same-name skills by context overlap."""
        names = set()
        for sid in ids:
            names |= set(_WORD.findall(normalize_name(strip_qualifier(self.index.name(sid)))))
        ctx = set(_WORD.findall(normalize_name(context))) - _STOP - names
        def overlap(sid):
            s = self.index.skills[sid]
            text = " ".join(str(s.get(k) or "") for k in ("description", "category", "subcategory", "name"))
            return len(ctx & (set(_WORD.findall(normalize_name(text))) - _STOP))
        # ties: taxonomy order
        return max(ids, key=lambda sid: (overlap(sid), -ids.index(sid)))

    def _ce_decide(self, mention, context, cands):
        pairs = [(f"{mention} | {context}", f"{self.index.name(s)}: {self.index.skills[s].get('description') or ''}")
                 for s, _ in cands]
        logits = [float(x) for x in self._ce.predict(pairs)]
        i = max(range(len(cands)), key=lambda j: logits[j])
        ok = logits[i] > 0.0
        return self._result(mention, cands[i][0] if ok else NIL, logits[i], "cross-encoder" if ok else "nil", cands)

    def _ask(self, prompt, retries=3, backoff=2.0):
        """One chat call, retried on 429/5xx/timeouts."""
        from extract.extraction import _is_transient
        for attempt in range(retries + 1):
            try:
                self.llm_calls += 1
                return self.llm_client.chat.completions.create(
                    model=self.llm_model, temperature=0.0, max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}])
            except Exception as exc:
                if attempt == retries or not _is_transient(exc):
                    raise
                time.sleep(backoff * 2 ** attempt)

    def _llm_decide(self, mention, context, cands):
        from extract.extraction import parse_json
        key = normalize_name(mention)
        sid = self._llm_cache.get(key)
        if sid is None:
            options = "\n".join(f"{s}: {self.index.name(s)} - {(self.index.skills[s].get('description') or '')[:160]}"
                                for s, _ in cands)
            prompt = (f"Skill mention: {mention}\nContext: {context}\nCandidates:\n{options}\n\n"
                      'Return JSON {"skill_id": "<one id from the list>"} if one candidate is the same skill '
                      'as the mention, or {"skill_id": "NIL"} if none is. A related skill is not the same skill. '
                      "Return JSON only.")
            try:
                reply = self._ask(prompt)
                sid = str(parse_json(reply.choices[0].message.content).get("skill_id", NIL))
                self._llm_cache[key] = sid
                self._unsaved += 1
                if self._unsaved >= 25:          # save every 25 answers
                    self.save_cache()
            except Exception as exc:
                print(f"[linking] LLM decision failed for '{mention}' ({type(exc).__name__}); using the threshold")
                best_id, best = cands[0]
                sid = best_id if best >= self.min_score else NIL
        if sid != NIL and sid not in dict(cands):
            sid = NIL                                   # answer not a candidate
        return self._result(mention, sid, dict(cands).get(sid, 0.0), "llm" if sid != NIL else "nil", cands)


# ------------------------------------------------------------------ finding skills in text

def _windows(text, max_n=MAX_NGRAM):
    """{normalized n-gram: (start, end)}, first occurrence."""
    toks = tokens_with_offsets(text)
    out = {}
    for i in range(len(toks)):
        for n in range(1, max_n + 1):
            if i + n > len(toks):
                break
            start, end = toks[i][0], toks[i + n - 1][1]
            out.setdefault(normalize_name(text[start:end]), (start, end, n))
    return out


class Gazetteer:
    """Find skill names/aliases in text, longest first."""

    def __init__(self, index):
        self.index = index
        self.keys = {k: ids for k, ids in index.exact.items() if k}

    def find(self, text):
        toks = tokens_with_offsets(text)
        hits = []
        for i in range(len(toks)):
            for n in range(min(MAX_NGRAM, len(toks) - i), 0, -1):
                start, end = toks[i][0], toks[i + n - 1][1]
                surface = text[start:end]
                ids = self.keys.get(normalize_name(surface))
                if not ids:
                    continue
                if len(normalize_name(surface)) <= 2 and not _short_ok(surface, ids, self.index):
                    continue
                hits.append((start, end, ids))
                break
        hits.sort(key=lambda h: (-(h[1] - h[0]), h[0]))
        taken, out = [], []
        for start, end, ids in hits:
            if any(start < e and s < end for s, e in taken):
                continue
            taken.append((start, end))
            out.append((start, end, ids))
        return sorted(out)


def _short_ok(surface, ids, index):
    """Short names ('R', 'Go') must match case exactly."""
    if not any(ch.isupper() for ch in surface):
        return False
    return any(surface == strip_qualifier(index.name(sid)) or surface.isupper() for sid in ids)


class TextSearch:
    """Find a missed skill in raw text: exact, else fuzzy (ratio >= t)."""

    def __init__(self, text, index):
        self.text = text or ""
        self.index = index
        self.windows = _windows(self.text)
        self._keys = list(self.windows)
        self._tri = None                            # lazy trigram index
        self._memo = {}

    def find(self, skill_id, fuzzy_threshold=0.9):
        """Exact hit, else best fuzzy hit, else None (memoized)."""
        key = (skill_id, fuzzy_threshold)
        if key not in self._memo:
            self._memo[key] = self._find(skill_id, fuzzy_threshold)
        return self._memo[key]

    def _find(self, skill_id, fuzzy_threshold):
        forms = sorted(self.index.surface.get(skill_id, ()), key=len, reverse=True)
        for form in forms:
            hit = self.windows.get(form)
            if hit and (len(form) > 2 or _short_ok(self.text[hit[0]:hit[1]], [skill_id], self.index)):
                return {"start": hit[0], "end": hit[1], "text": self.text[hit[0]:hit[1]], "how": "exact"}
        if fuzzy_threshold is None or fuzzy_threshold >= 1:
            return None
        for form in forms:
            if len(form) < 5:
                continue
            best = None
            sm = SequenceMatcher(None, "", form)       # form is seq2 (cached); windows go in seq1
            for key in self._fuzzy_candidates(form, fuzzy_threshold):
                sm.set_seq1(key)
                if sm.real_quick_ratio() < fuzzy_threshold or sm.quick_ratio() < fuzzy_threshold:
                    continue
                r = sm.ratio()
                if r >= fuzzy_threshold and (best is None or r > best[0]):
                    best = (r, key)
            if best:
                s, e, _ = self.windows[best[1]]
                return {"start": s, "end": e, "text": self.text[s:e], "how": f"fuzzy ({best[0]:.2f})"}
        return None

    def _fuzzy_candidates(self, form, threshold):
        """Windows that can reach `threshold` (q-gram bound, lossless)."""
        if self._tri is None:                       # (token count, trigram) -> [(window index, count)]
            self._tri = defaultdict(list)
            for i, key in enumerate(self._keys):
                n_tok = self.windows[key][2]
                for g, c in _grams(key).items():
                    self._tri[(n_tok, g)].append((i, c))
        n = len(_TOKEN.findall(form)) or 1
        tol = max(2, len(form) // 5)
        shared = defaultdict(int)
        for g, c in _grams(form).items():
            for i, ci in self._tri.get((n, g), ()):
                shared[i] += ci if ci < c else c
        out = []
        for i in sorted(shared):                    # text order for stable ties
            key = self._keys[i]
            if abs(len(key) - len(form)) > tol:
                continue
            d_max = int((len(key) + len(form)) * (1 - threshold))
            if shared[i] >= max(len(key), len(form)) - 3 * d_max:
                out.append(key)
        return out


# ------------------------------------------------------------------ evaluation helpers

def context_of(text, start, end, width=160):
    """Line (or window) around a span, as context."""
    ls = text.rfind("\n", 0, start) + 1
    le = text.find("\n", end)
    le = len(text) if le < 0 else le
    if le - ls > width * 2:
        ls, le = max(0, start - width), min(len(text), end + width)
    return re.sub(r"\s+", " ", text[ls:le]).strip()


def load_linking_gold(path, split=None):
    """Labeled rows, optionally only one split half."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("gold_skill_id") or "").strip()]
    if split and rows and "split" in rows[0]:
        rows = [r for r in rows if r["split"] == split]
    return rows


def tune_min_score(index, rows, grid=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9), **linker_args):
    """Tune min_score on dev labels -> (best, [(score, acc)])."""
    results = []
    for t in grid:
        res = evaluate_linking(SkillLinker(index, min_score=t, **linker_args), rows)
        results.append((t, res["accuracy@1"] if res else 0.0))
    best = max(results, key=lambda x: (x[1], -abs(x[0] - 0.65)))
    return best[0], results


def evaluate_linking(linker, rows):
    """Accuracy@1 and NIL accuracy over labeled mentions."""
    preds = [(linker.link(r["mention"], r.get("context", ""))["skill_id"], r["gold_skill_id"].strip()) for r in rows]
    if not preds:
        return None
    acc = sum(p == g for p, g in preds) / len(preds)
    nil = [(p, g) for p, g in preds if g == NIL]
    linked = [(p, g) for p, g in preds if g != NIL]
    return {"n": len(preds), "accuracy@1": acc,
            "nil_accuracy": sum(p == NIL for p, _ in nil) / len(nil) if nil else None,
            "linked_accuracy": sum(p == g for p, g in linked) / len(linked) if linked else None,
            "per_mention": [int(p == g) for p, g in preds]}


def linking_rows(records, texts, linker, n=300, seed=573):
    """Unique mentions to label with candidates; rows alternate dev/test."""
    mentions = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        text = texts.get(rec["doc_id"], "")
        for field in ("skills", "required_skills"):
            for f in rec["facts"].get(field) or []:
                mentions.append((rec["kind"], rec["doc_id"], f["name"], context_of(text, f["start"], f["end"])))
    rng = random.Random(seed)
    rng.shuffle(mentions)
    seen, rows = set(), []
    for kind, doc_id, mention, ctx in mentions:
        if normalize_name(mention) in seen:
            continue
        seen.add(normalize_name(mention))
        res = linker.link(mention, ctx)
        rows.append({"mention_id": f"M{len(rows) + 1:04d}", "kind": kind, "doc_id": doc_id,
                     "mention": mention, "context": ctx,
                     "candidates": " | ".join(f"{c}={linker.index.name(c)}" for c, _ in res["candidates"][:5]),
                     "split": "dev" if len(rows) % 2 == 0 else "test",
                     "gold_skill_id": "", "annotator": ""})
        if len(rows) >= n:
            break
    return rows
