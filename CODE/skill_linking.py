"""Skill linking: map extracted skill mentions to Lightcast Open Skills IDs.

Each mention goes through these stages until one succeeds:

  1. exact    - the mention is a Lightcast name, with or without its bracketed
                qualifier ("Python" finds "Python (Programming Language)")
  2. alias    - the alias table rewrites a known variant (JS, Postgres, k8s)
  3. fuzzy    - a near-identical spelling (rapidfuzz, or difflib without it)
  4. contains - the mention contains a whole Lightcast name; the longest one
                wins ("network architecture design" -> "Network Architecture").
                A single-word name only counts if it is a specialized skill, so
                generic words like "design" or "security" do not match.
  5. embed    - a bi-encoder retrieves the closest Lightcast skills for the
                mention in its sentence (first stage of the BLINK-style linker)
  6. NIL      - nothing fits

With a decider (an LLM, proposal Sec. III-B) the uncertain cases are settled:
mentions left NIL that have candidates, and links flagged needs_review. The
decider picks one candidate or NIL. Without one, embedding matches are only
suggestions.

Also here: batch linking of the extraction output (run_linking), CSV exports
for the Neo4j graph (export_graph), and the raw-text check used before a job
skill is called missing from a resume (SkillTextFinder, check_missing).

The linked fields match the Skill node in CODE/ontology.json: skill_id, name,
category, subcategory and source (lightcast | extracted).
"""

import csv
import json
import os
import re
import time
from collections import Counter, defaultdict

TOKEN_URL = "https://auth.emsicloud.com/connect/token"
# Current documented host first; the older host is tried if it fails.
SKILLS_URLS = (
    "https://api.lightcast.io/skills/versions/{version}/skills",
    "https://emsiservices.com/skills/versions/{version}/skills",
)
SCOPE = "emsi_open"

# Asked for first; if the server rejects the extra fields, the download is
# retried with the default fields, which do not include the hierarchy.
FULL_FIELDS = "id,name,type,category,subcategory,description,infoUrl"

DEFAULT_CACHE = "DATA/lightcast/skills.json"
DEFAULT_ALIASES = "CODE/skill_aliases.json"


# ================================================================ step 1: download

def fetch_token(client_id, client_secret, scope=SCOPE, session=None):
    """Request an OAuth access token (valid for one hour)."""
    session = session or _requests()
    resp = session.post(
        TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
            "scope": scope,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def download_skills(token, version="latest", session=None, retries=3, backoff=2.0):
    """Download every skill in one request.

    Returns (raw_skills, fields_used, url). Falls back to the default fields if
    the server does not accept the full field list, and to the older host if
    the current one cannot be reached or does not know the path.
    """
    session = session or _requests()
    headers = {"Authorization": f"Bearer {token}"}
    last_error = None
    for template in SKILLS_URLS:
        url = template.format(version=version)
        try:
            raw, fields = _download_from(session, url, headers, retries, backoff)
            return raw, fields, url
        except _HostProblem as exc:
            last_error = exc
    raise RuntimeError(f"skills download failed on every host: {last_error}")


class _HostProblem(Exception):
    pass


def _download_from(session, url, headers, retries, backoff):
    for fields in (FULL_FIELDS, None):
        params = {"fields": fields} if fields else {}
        for attempt in range(retries + 1):
            try:
                resp = session.get(url, headers=headers, params=params, timeout=120)
            except Exception as exc:  # DNS, connection refused, timeout
                if "Connection" in type(exc).__name__ or "Timeout" in type(exc).__name__:
                    raise _HostProblem(f"{url}: {type(exc).__name__}") from exc
                raise
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == retries:
                    raise _HostProblem(f"{url}: HTTP {resp.status_code}")
                time.sleep(backoff * 2 ** attempt)
                continue
            break
        if resp.status_code == 404:
            raise _HostProblem(f"{url}: HTTP 404")
        if resp.status_code in (400, 422) and fields:
            continue
        resp.raise_for_status()  # 401/403 are credential problems: stop and say so
        return resp.json()["data"], fields or "default"
    raise _HostProblem(f"{url}: rejected both field lists")


def _requests():
    import requests  # imported here so the tests run without it
    return requests.Session()


# ---------------------------------------------------------------- records

def _name_of(value):
    """Lightcast returns type/category/subcategory as {"id", "name"} objects."""
    if isinstance(value, dict):
        return value.get("name")
    return value


def simplify(raw):
    """Keep the fields the linker and the graph need."""
    return {
        "id": raw["id"],
        "name": raw["name"],
        "type": _name_of(raw.get("type")),
        "category": _name_of(raw.get("category")),
        "subcategory": _name_of(raw.get("subcategory")),
        "description": raw.get("description"),
    }


def save_skills(skills, path=DEFAULT_CACHE, version="latest", fields="default"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "source": "Lightcast Open Skills",
        "version": version,
        "fields": fields,
        "fetched": time.strftime("%Y-%m-%d"),
        "count": len(skills),
        "skills": skills,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


def load_skills(path=DEFAULT_CACHE):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["skills"]


def download_and_cache(client_id, client_secret, path=DEFAULT_CACHE, version="latest", session=None):
    """One call for the notebook: token, download, simplify, save."""
    token = fetch_token(client_id, client_secret, session=session)
    raw, fields, _ = download_skills(token, version=version, session=session)
    skills = [simplify(r) for r in raw]
    save_skills(skills, path, version=version, fields=fields)
    return skills


# ---------------------------------------------------------------- names

_QUALIFIER = re.compile(r"\s*\([^)]*\)\s*$")
_CHAR_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
# A leading dot is kept (".NET"); a trailing one is sentence punctuation.
_LEAD_PUNCT = " \t\n,;:'\"*•-"
_TRAIL_PUNCT = " \t\n.,;:'\"*•-"


def normalize_name(text):
    """Lower-case, unify quotes and dashes, collapse whitespace, trim edge punctuation.

    Inner punctuation is kept on purpose: C++, C#, .NET and Node.js depend on it.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text.translate(_CHAR_MAP))
    return text.lstrip(_LEAD_PUNCT).rstrip(_TRAIL_PUNCT).lower()


def strip_qualifier(name):
    """'Python (Programming Language)' -> 'Python'."""
    return _QUALIFIER.sub("", name).strip() if isinstance(name, str) else name


def name_keys(skill):
    """Normalized keys a skill can be found under: full name and unqualified name."""
    keys = {normalize_name(skill["name"])}
    bare = normalize_name(strip_qualifier(skill["name"]))
    if bare:
        keys.add(bare)
    keys.discard("")
    return keys


def build_name_index(skills):
    """Map each normalized key to the IDs that carry it.

    A key can point to more than one skill (for example after the qualifier
    is removed); the linker then chooses between them.
    """
    index = defaultdict(list)
    for skill in skills:
        for key in name_keys(skill):
            if skill["id"] not in index[key]:
                index[key].append(skill["id"])
    return dict(index)


def by_id(skills):
    return {s["id"]: s for s in skills}


def taxonomy_stats(skills):
    """Counts to print after the download, to check the hierarchy came through."""
    index = build_name_index(skills)
    return {
        "skills": len(skills),
        "types": dict(Counter(s.get("type") for s in skills)),
        "categories": len({s["category"] for s in skills if s.get("category")}),
        "subcategories": len({s["subcategory"] for s in skills if s.get("subcategory")}),
        "missing_category": sum(1 for s in skills if not s.get("category")),
        "with_description": sum(1 for s in skills if s.get("description")),
        "index_keys": len(index),
        "ambiguous_keys": sum(1 for ids in index.values() if len(ids) > 1),
    }


# ================================================================ step 2: aliases and spelling

def load_aliases(path=DEFAULT_ALIASES):
    """Read the alias table {variant: Lightcast name}. Keys starting with '_' are comments."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {normalize_name(k): normalize_name(v) for k, v in raw.items() if not k.startswith("_")}


def check_aliases(aliases, index):
    """Alias targets that are not a Lightcast name; fix these after the download."""
    return sorted({target for target in aliases.values() if target not in index})


_LEADING = re.compile(
    r"^(?:(?:strong|solid|excellent|good|basic|advanced|working|hands-on|proven)\s+)?"
    r"(?:knowledge|experience|expertise|proficiency|familiarity|understanding|skills?)\s+"
    r"(?:of|in|with|using)\s+"
)
_TRAILING = re.compile(r"\s+(?:skills?|experience|knowledge)$")


def core_form(mention):
    """The mention without wrapper words: 'knowledge of SQL' -> 'sql', 'communication skills' -> 'communication'."""
    base = normalize_name(mention)
    core = _TRAILING.sub("", _LEADING.sub("", base)).strip()
    return core or base


def mention_variants(mention):
    """Normalized forms of a mention to look up, most literal first."""
    core = core_form(mention)
    variants = [
        normalize_name(mention),
        normalize_name(strip_qualifier(mention)),
        core,
        core.replace(" & ", " and "),
        core.replace(" and ", " & "),
    ]
    if len(core) > 4 and core.endswith("s") and not core.endswith("ss"):
        variants.append(core[:-1])  # "relational databases" -> "relational database"
    out = []
    for v in variants:
        v = v.strip()
        if v and v not in out:
            out.append(v)
    return out


_WORD = re.compile(r"[a-z0-9+#]+")


def _words(text):
    return set(_WORD.findall(normalize_name(text))) if text else set()


def choose(ids, skills_by_id, context=None):
    """Pick one of several skills that share a name.

    The skill whose name and description share the most words with the
    mention's context wins; a specialized skill beats a common one on a tie.
    Returns (skill_id, ambiguous): ambiguous means the context did not decide.
    """
    if len(ids) == 1:
        return ids[0], False
    ctx = _words(context)

    def overlap(sid):
        s = skills_by_id[sid]
        return len(ctx & (_words(s["name"]) | _words(s.get("description"))))

    ranked = sorted(ids, key=lambda sid: (overlap(sid), skills_by_id[sid].get("type") == "Specialized Skill"),
                    reverse=True)
    return ranked[0], overlap(ranked[0]) == overlap(ranked[1])


class FuzzyMatcher:
    """Closest Lightcast names by spelling; scores are 0-1."""

    def __init__(self, keys):
        self.keys = list(keys)
        try:
            from rapidfuzz import fuzz, process
            self._extract = lambda q, n: [(k, s / 100.0) for k, s, _ in
                                          process.extract(q, self.keys, scorer=fuzz.token_sort_ratio, limit=n)]
        except ImportError:  # slower, but needs nothing installed
            import difflib

            def _extract(q, n):
                close = difflib.get_close_matches(q, self.keys, n=n, cutoff=0.6)
                return [(k, difflib.SequenceMatcher(None, q, k).ratio()) for k in close]
            self._extract = _extract

    def top(self, query, limit=5):
        if not query:
            return []
        return sorted(self._extract(query, limit), key=lambda h: -h[1])


# ================================================================ step 3: bi-encoder retrieval

def skill_text(skill, max_desc=200):
    """What the bi-encoder embeds for a skill: its name plus the start of its description."""
    desc = (skill.get("description") or "").strip()
    return f"{skill['name']}: {desc[:max_desc]}" if desc else skill["name"]


class EmbeddingRetriever:
    """Bi-encoder retrieval over all Lightcast skills.

    encoder is any object with encode(texts, batch_size=...) returning one
    vector per text; by default a sentence-transformers model is loaded.
    Skill vectors are cached to cache_path (.npz) so they are computed once.
    """

    def __init__(self, skills, encoder=None, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 cache_path=None, batch_size=256):
        import numpy as np
        self.np = np
        self.ids = [s["id"] for s in skills]
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model_name)
        self.encoder = encoder
        self.batch_size = batch_size
        matrix = None
        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=False)
            if data["ids"].tolist() == self.ids:
                matrix = data["vectors"]
        if matrix is None:
            matrix = self._encode([skill_text(s) for s in skills])
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                np.savez(cache_path, ids=np.array(self.ids), vectors=matrix)
        self.matrix = matrix

    def _encode(self, texts):
        np = self.np
        vecs = np.asarray(self.encoder.encode(texts, batch_size=self.batch_size), dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def search(self, mention, context=None, k=5):
        """Top-k (skill_id, cosine) for the mention, read in its sentence when that adds anything."""
        query = mention
        if context and normalize_name(context) != normalize_name(mention):
            query = f"{mention}: {context}"
        scores = self.matrix @ self._encode([query])[0]
        k = min(k, len(self.ids))
        top = self.np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]


# ================================================================ step 4: decision step

DECIDE_SYSTEM = "You link skill mentions to a skills taxonomy. Return only JSON."

DECIDE_PROMPT = """Which Lightcast skill does this mention refer to?

Mention: "{mention}"
Where it appears: "{context}"

Candidates:
{candidates}

Rules:
- Choose the candidate that is the same skill as the mention. Wording may differ
  (plural, abbreviation, extra words such as "skills", "experience" or "design").
- Do not choose a skill that is only related, a different tool, or much broader
  or narrower than the mention.
- If no candidate is the same skill, answer NIL.

Return JSON only: {{"choice": "<candidate id or NIL>"}}"""


def _candidate_line(skill):
    where = " > ".join(x for x in (skill.get("category"), skill.get("subcategory")) if x)
    desc = (skill.get("description") or "").strip()
    line = f"- {skill['id']}: {skill['name']}"
    if where:
        line += f" [{where}]"
    if desc:
        line += f" - {desc[:140]}"
    return line


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_json_object(text):
    """The JSON object in a model reply (Qwen3 may add a <think> block or code fences)."""
    if not isinstance(text, str):
        raise ValueError("empty model reply")
    text = _THINK.sub("", text).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in model reply")
        return json.loads(text[start:end + 1])


def _status(exc):
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status


def _is_transient(exc):
    status = _status(exc)
    if status is not None:
        return status == 429 or status >= 500
    name = type(exc).__name__
    return "Timeout" in name or "Connection" in name


class LLMDecider:
    """Decision step: an LLM picks one candidate skill or NIL.

    client is an OpenAI-compatible client (the notebook's ASU Voyager client).
    constrained=True asks for schema-constrained output limited to the
    candidate IDs; if the server rejects that request, plain JSON prompting
    is used for the rest of the session.
    """

    def __init__(self, client, model="qwen3-235b-a22b", constrained=True, max_tokens=1024,
                 temperature=0.0, retries=2, backoff=2.0):
        self.client = client
        self.model = model
        self.constrained = constrained
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.backoff = backoff
        self.calls = 0
        self.errors = 0

    def _messages(self, mention, context, candidates):
        prompt = DECIDE_PROMPT.format(
            mention=mention,
            context=(context or mention)[:400],
            candidates="\n".join(_candidate_line(c) for c in candidates),
        )
        if self.model.lower().startswith("qwen3"):
            prompt += "\n/no_think"  # Qwen3 switch: answer without a long reasoning block
        return [{"role": "system", "content": DECIDE_SYSTEM}, {"role": "user", "content": prompt}]

    def _request(self, messages, ids):
        args = dict(model=self.model, messages=messages, max_tokens=self.max_tokens,
                    temperature=self.temperature)
        if self.constrained:
            args["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "skill_choice", "strict": True,
                "schema": {"type": "object",
                           "properties": {"choice": {"type": "string", "enum": ids + ["NIL"]}},
                           "required": ["choice"], "additionalProperties": False}}}
        return self.client.chat.completions.create(**args)

    def decide(self, mention, context, candidates):
        """Return the chosen skill ID, or None for NIL. Raises if no valid answer comes back."""
        ids = [c["id"] for c in candidates]
        messages = self._messages(mention, context, candidates)
        attempt = 0
        while True:
            try:
                reply = self._request(messages, ids)
                break
            except Exception as exc:
                if self.constrained and _status(exc) in (400, 422):
                    self.constrained = False
                    continue
                attempt += 1
                if attempt > self.retries or not _is_transient(exc):
                    self.errors += 1
                    raise
                time.sleep(self.backoff * 2 ** (attempt - 1))
        self.calls += 1
        choice_obj = reply.choices[0]
        if getattr(choice_obj, "finish_reason", None) == "length":
            self.errors += 1
            raise ValueError("reply cut off at max_tokens")
        choice = _parse_json_object(choice_obj.message.content).get("choice")
        if isinstance(choice, str) and choice.strip().upper() == "NIL":
            return None
        if choice not in ids:
            self.errors += 1
            raise ValueError(f"model chose {choice!r}, not one of the candidates")
        return choice


# ================================================================ the linker

class Linker:
    """Links one mention at a time; see the module docstring for the stages.

    fuzzy and contains switch those stages off (for the alias-only baseline);
    retriever adds embedding candidates and decider adds the decision step.
    """

    def __init__(self, skills, aliases=None, fuzzy=True, contains=True, retriever=None, decider=None,
                 auto_fuzzy=0.92, review_fuzzy=0.85, auto_embed=None, review_embed=0.5,
                 pool_embed=0.35, decide_k=6):
        self.skills = by_id(skills)
        self.index = build_name_index(skills)
        self.aliases = aliases or {}
        self.fuzzy = FuzzyMatcher(self.index) if fuzzy else None
        self.use_contains = contains
        self.retriever = retriever
        self.decider = decider
        self.auto_fuzzy = auto_fuzzy
        self.review_fuzzy = review_fuzzy
        self.auto_embed = auto_embed
        self.review_embed = review_embed
        self.pool_embed = pool_embed
        self.decide_k = decide_k
        self._cache = {}

    def link(self, mention, context=None):
        key = (normalize_name(mention), normalize_name(context))
        if key not in self._cache:
            result = self._link(mention, context)
            if self.decider is not None and (result["needs_review"] or result["id"] is None):
                pool = self._pool(mention, context, result)
                if pool:
                    result = self._decide(mention, context, result, pool)
            self._cache[key] = result
        return dict(self._cache[key], mention=mention)

    def candidates_for(self, mention, context=None, k=6):
        """Up to k plausible skill IDs for a mention, from every stage (for the labeling sheet)."""
        result = self._link(mention, context)
        pool = self._pool(mention, context, result)
        if self.fuzzy is not None and len(pool) < k:
            for name_key, _ in self.fuzzy.top(core_form(mention), limit=k):
                for sid in self.index[name_key]:
                    if sid not in pool:
                        pool.append(sid)
        return pool[:k]

    def _link(self, mention, context):
        variants = mention_variants(mention)
        if not variants:
            return self._nil(mention, [])

        for v in variants:
            if v in self.index:
                return self._linked(mention, self.index[v], "exact", 1.0, context)

        for v in variants:
            target = self.aliases.get(v)
            if target in self.index:
                return self._linked(mention, self.index[target], "alias", 1.0, context)

        suggestions = []
        if self.fuzzy is not None:
            for name_key, score in self.fuzzy.top(core_form(mention), limit=3):
                if score >= self.auto_fuzzy and not suggestions:
                    return self._linked(mention, self.index[name_key], "fuzzy", round(score, 3), context)
                if score >= self.review_fuzzy:
                    suggestions += [(sid, score, "fuzzy") for sid in self.index[name_key]]

        if self.use_contains:
            inner = self._contained(core_form(mention))
            if inner:
                ids, covered = inner
                result = self._linked(mention, ids, "contains", round(covered, 3), context)
                # a name covering half of the mention or less is only a partial reading of it
                result["needs_review"] = result["needs_review"] or covered <= 0.5
                return result

        if self.retriever is not None:
            for rank, (sid, score) in enumerate(self.retriever.search(mention, context, k=3)):
                if rank == 0 and self.auto_embed is not None and score >= self.auto_embed:
                    result = self._linked(mention, [sid], "embedding", round(score, 3), context)
                    result["needs_review"] = True  # only a decider can confirm it
                    return result
                if score >= self.review_embed:
                    suggestions.append((sid, score, "embedding"))

        return self._nil(mention, suggestions)

    def _contained(self, core):
        """Longest Lightcast name made of whole consecutive words of the mention.

        Returns (ids, share of the mention's words it covers), or None. Plural
        and singular forms are both tried ("satellite communication" also
        finds "Satellite Communications").
        """
        words = core.split()
        for n in range(len(words) - 1, 0, -1):  # the full mention was already tried
            for start in range(len(words) - n + 1):
                gram = " ".join(words[start:start + n])
                for form in (gram, gram + "s", gram[:-1] if gram.endswith("s") else None):
                    ids = self.index.get(form) if form else None
                    if not ids:
                        continue
                    if n == 1:
                        # one-letter or two-letter names (C, R, Go) are too easy to hit by accident
                        if len(gram) < 3:
                            continue
                        ids = [sid for sid in ids if self.skills[sid].get("type") == "Specialized Skill"]
                        if not ids:
                            continue
                    return ids, n / len(words)
        return None

    def _pool(self, mention, context, result):
        """Candidate IDs for the decision step: what the stages found, then the retriever's top hits."""
        pool = []
        found = ([result["id"]] if result["id"] else []) + result["candidates"] + \
            [s["id"] for s in result["suggestions"]]
        for sid in found:
            if sid not in pool:
                pool.append(sid)
        if self.retriever is not None:
            for sid, score in self.retriever.search(mention, context, k=self.decide_k):
                if score >= self.pool_embed and sid not in pool:
                    pool.append(sid)
        return pool[: self.decide_k + 2]

    def _decide(self, mention, context, result, pool):
        try:
            choice = self.decider.decide(mention, context, [self.skills[sid] for sid in pool])
        except Exception as exc:  # keep the stage result, still flagged for review
            return dict(result, decision_error=f"{type(exc).__name__}: {exc}")
        if choice is None:
            decided = self._nil(mention, [])
            decided["suggestions"] = result["suggestions"]
        else:
            confirmed = choice == result["id"]
            decided = self._linked(mention, [choice], result["method"] if confirmed else "llm",
                                   result["score"] if confirmed else None, context)
        decided.update(needs_review=False, decided_by="llm", proposed=result["method"])
        return decided

    def _linked(self, mention, ids, method, score, context):
        sid, ambiguous = choose(ids, self.skills, context)
        s = self.skills[sid]
        return {
            "mention": mention, "id": sid, "name": s["name"], "type": s.get("type"),
            "category": s.get("category"), "subcategory": s.get("subcategory"),
            "method": method, "score": score, "needs_review": ambiguous,
            "candidates": list(ids) if len(ids) > 1 else [], "suggestions": [],
        }

    def _nil(self, mention, suggestions):
        best = {}
        for sid, score, via in suggestions:
            if sid not in best or score > best[sid][0]:
                best[sid] = (score, via)
        ranked = sorted(best.items(), key=lambda item: -item[1][0])[:3]
        return {
            "mention": mention, "id": None, "name": None, "type": None,
            "category": None, "subcategory": None, "method": "nil", "score": None,
            "needs_review": bool(ranked), "candidates": [],
            "suggestions": [{"id": sid, "name": self.skills[sid]["name"], "score": round(score, 3), "via": via}
                            for sid, (score, via) in ranked],
        }


# ---------------------------------------------------------------- applying it to extraction output

SKILL_FIELDS = {"resume": "skills", "posting": "required_skills"}


def link_facts(linker, facts):
    """Add a 'link' to each skill fact; the fact's source_span is the context."""
    return [{**fact, "link": linker.link(fact.get("name"), context=fact.get("source_span"))}
            for fact in facts or []]


def link_record(linker, record):
    """Link every skill fact of one record written by extraction.run_batch."""
    if record.get("status") != "ok":
        return record
    field = SKILL_FIELDS[record["kind"]]
    facts = dict(record["facts"])
    facts[field] = link_facts(linker, facts.get(field))
    return {**record, "facts": facts}


def link_summary(links):
    """Counts for a list of link results."""
    methods = Counter(link["method"] for link in links)
    n = len(links)
    nil = methods.get("nil", 0)
    return {
        "mentions": n, "linked": n - nil, "nil": nil,
        "link_rate": round((n - nil) / n, 4) if n else None,
        "by_method": dict(methods),
        "needs_review": sum(1 for link in links if link["needs_review"]),
        "decided_by_llm": sum(1 for link in links if link.get("decided_by") == "llm"),
        "decision_errors": sum(1 for link in links if link.get("decision_error")),
    }


def to_skill_node(link):
    """Skill node properties as defined in CODE/ontology.json."""
    if link.get("id"):
        return {"skill_id": link["id"], "name": link["name"], "category": link["category"],
                "subcategory": link["subcategory"], "source": "lightcast"}
    slug = re.sub(r"[^a-z0-9+#.]+", "-", normalize_name(link["mention"])).strip("-")
    return {"skill_id": f"EXT:{slug}", "name": link["mention"], "category": None,
            "subcategory": None, "source": "extracted"}


# ================================================================ step 5: the whole corpus

def iter_jsonl(path):
    """Records of a JSONL file; a partial last line left by a killed run is skipped."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def latest_records(path):
    """Latest record per document, like extraction.load_records."""
    latest = {}
    for rec in iter_jsonl(path):
        latest[str(rec.get("doc_id"))] = rec
    return list(latest.values())


def run_linking(linker, in_path, out_path, limit=None, log_every=100):
    """Link every record of an extraction JSONL file into out_path.

    Documents already linked successfully are skipped, so the call can be
    rerun after an interruption.
    """
    done = set()
    if os.path.exists(out_path):
        done = {str(r.get("doc_id")) for r in iter_jsonl(out_path) if r.get("status") == "ok"}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    counts = {"ok": 0, "error": 0, "skipped": 0}
    with open(out_path, "a", encoding="utf-8") as out:
        if out.tell() > 0:  # do not glue the first new record onto a partial line
            with open(out_path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    out.write("\n")
        for rec in latest_records(in_path):
            if str(rec.get("doc_id")) in done:
                counts["skipped"] += 1
                continue
            if limit is not None and counts["ok"] + counts["error"] >= limit:
                break
            linked = link_record(linker, rec)
            out.write(json.dumps(linked, ensure_ascii=False) + "\n")
            out.flush()
            counts["ok" if linked.get("status") == "ok" else "error"] += 1
            n = counts["ok"] + counts["error"]
            if log_every and n % log_every == 0:
                print(f"  linked {n} documents")
    return counts


def iter_links(records):
    """(kind, doc_id, fact) for every linked skill fact."""
    for rec in records:
        if rec.get("status") != "ok":
            continue
        field = SKILL_FIELDS.get(rec.get("kind"))
        for fact in (rec.get("facts") or {}).get(field) or []:
            if "link" in fact:
                yield rec["kind"], str(rec["doc_id"]), fact


def corpus_report(records, top=25):
    """Link counts per kind, plus the most frequent NIL and flagged mentions (to grow the alias table)."""
    by_kind = defaultdict(list)
    for kind, _, fact in iter_links(records):
        by_kind[kind].append(fact["link"])
    report = {}
    for kind, links in by_kind.items():
        summary = link_summary(links)
        summary["distinct_skills"] = len({link["id"] for link in links if link["id"]})
        summary["top_nil"] = Counter(normalize_name(link["mention"]) for link in links
                                     if link["id"] is None).most_common(top)
        summary["top_review"] = Counter(f"{normalize_name(link['mention'])} -> {link['name']}" for link in links
                                        if link["id"] and link["needs_review"]).most_common(top)
        report[kind] = summary
    return report


GRAPH_EDGE_FILES = {"resume": "candidate_has_skill.csv", "posting": "job_requires_skill.csv"}


def export_graph(records, out_dir):
    """CSV files for Neo4j LOAD CSV: skill nodes plus HAS_SKILL and REQUIRES_SKILL edges.

    Edge columns follow the ontology: source_span, span_start, span_end, and
    importance for postings. Returns the paths written.
    """
    os.makedirs(out_dir, exist_ok=True)
    nodes, edges = {}, {"resume": [], "posting": []}
    for kind, doc_id, fact in iter_links(records):
        node = to_skill_node(fact["link"])
        nodes.setdefault(node["skill_id"], node)
        edges[kind].append({
            "doc_id": doc_id, "skill_id": node["skill_id"], "importance": fact.get("importance"),
            "mention": fact.get("name"), "source_span": fact.get("source_span"),
            "span_start": fact.get("start"), "span_end": fact.get("end"),
            "link_method": fact["link"]["method"], "needs_review": fact["link"]["needs_review"],
        })
    paths = {"skills": os.path.join(out_dir, "skills.csv")}
    _write_csv(paths["skills"], ["skill_id", "name", "category", "subcategory", "source"], nodes.values())
    for kind, name in GRAPH_EDGE_FILES.items():
        columns = ["doc_id", "skill_id", "mention", "source_span", "span_start", "span_end",
                   "link_method", "needs_review"]
        if kind == "posting":
            columns.insert(2, "importance")
        paths[kind] = os.path.join(out_dir, name)
        _write_csv(paths[kind], columns, edges[kind])
    return paths


def _write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ================================================================ missed-skill check (proposal feedback)

class SkillTextFinder:
    """Look for a Lightcast skill in raw text by its names and aliases.

    Before a job skill is called missing from a resume, the resume text is
    searched for it. If the skill's name or an alias is there, extraction
    missed it: the skill is credited and the case logged. Names shorter than
    min_len characters (R, C, Go, AI) are not searched, to avoid false hits.
    """

    def __init__(self, skills, aliases=None, min_len=3):
        self.names = defaultdict(set)
        for skill in skills:
            self.names[skill["id"]] |= name_keys(skill)
        index = build_name_index(skills)
        for alias, target in (aliases or {}).items():
            for sid in index.get(target, []):
                self.names[sid].add(alias)
        self.min_len = min_len
        self._patterns = {}

    def _pattern(self, skill_id):
        if skill_id not in self._patterns:
            forms = sorted((n for n in self.names.get(skill_id, ()) if len(n) >= self.min_len),
                           key=len, reverse=True)
            parts = [r"\s+".join(re.escape(w) for w in form.split()) for form in forms]
            self._patterns[skill_id] = (
                re.compile(r"(?<![\w+#])(?:" + "|".join(parts) + r")(?![\w+#])", re.IGNORECASE)
                if parts else None)
        return self._patterns[skill_id]

    def find(self, skill_id, text):
        """(start, end) of the first mention of the skill in text, or None."""
        pattern = self._pattern(skill_id)
        if pattern is None or not isinstance(text, str):
            return None
        match = pattern.search(text)
        return (match.start(), match.end()) if match else None


def check_missing(job_facts, resume_facts, resume_text, finder):
    """Status of each job skill against one resume.

    matched       - the resume's linked skills include it
    found_in_text - extraction missed it but the resume text names it: credit it
    not_found     - unknown, never treated as a hard-requirement conflict
    unresolved    - the job skill itself could not be linked
    """
    have = {f["link"]["id"] for f in resume_facts if f["link"].get("id")}
    out = []
    for job in job_facts:
        link = job["link"]
        row = {"job_skill": job.get("name"), "skill_id": link.get("id"), "lightcast": link.get("name"),
               "importance": job.get("importance", "required"), "status": None, "span": None}
        if not link.get("id"):
            row["status"] = "unresolved"
        elif link["id"] in have:
            row["status"] = "matched"
        else:
            row["span"] = finder.find(link["id"], resume_text)
            row["status"] = "found_in_text" if row["span"] else "not_found"
        out.append(row)
    return out


# ---------------------------------------------------------------- partial credit (demo illustration)
# The ranking step owns the tuned graph score; these placeholders only show
# what linking makes possible (proposal eqs. 2 and 3).

LAMBDA_SUB = 0.5
LAMBDA_CAT = 0.25
WEIGHTS = {"required": 2.0, "preferred": 1.0}


def credit(job_link, resume_link, lam_sub=LAMBDA_SUB, lam_cat=LAMBDA_CAT):
    """sigma(s, t): 1 same skill, lam_sub same subcategory, lam_cat same category only, else 0."""
    if not job_link.get("id") or not resume_link.get("id"):
        return 0.0, None
    if job_link["id"] == resume_link["id"]:
        return 1.0, "same skill"
    same_cat = job_link.get("category") and job_link["category"] == resume_link.get("category")
    if same_cat and job_link.get("subcategory") and job_link["subcategory"] == resume_link.get("subcategory"):
        return lam_sub, "same subcategory"
    if same_cat:
        return lam_cat, "same category"
    return 0.0, None


def coverage(job_facts, resume_facts, weights=WEIGHTS, lam_sub=LAMBDA_SUB, lam_cat=LAMBDA_CAT):
    """Weighted share of the job's linked skills that the resume covers.

    The job's skills are a set: mentions linked to the same Lightcast skill are
    scored once, as required if any of them is required. Returns
    (G, rows, unresolved); job skills that could not be linked are not scored
    but are returned in unresolved so they stay visible.
    """
    unresolved, grouped = [], {}
    for job in job_facts:
        link = job["link"]
        if not link.get("id"):
            unresolved.append(job["name"])
            continue
        entry = grouped.setdefault(link["id"], {"link": link, "mentions": [], "importance": "preferred"})
        entry["mentions"].append(job["name"])
        if job.get("importance", "required") == "required":
            entry["importance"] = "required"
    rows = []
    num = den = 0.0
    for entry in grouped.values():
        weight = weights.get(entry["importance"], 1.0)
        best, level, match = 0.0, None, None
        for res in resume_facts:
            value, why = credit(entry["link"], res["link"], lam_sub, lam_cat)
            if value > best:
                best, level, match = value, why, res
        rows.append({"job_skill": "; ".join(entry["mentions"]), "lightcast": entry["link"]["name"],
                     "importance": entry["importance"], "credit": best,
                     "level": level, "matched_by": match["name"] if match else None})
        num += weight * best
        den += weight
    return (num / den if den else 0.0), rows, unresolved
