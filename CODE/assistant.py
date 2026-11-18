"""Chat assistant over the team's pipeline; CODE/app.py is its web UI.

Every answer comes from modules already in CODE/:

  ranking   retrieval.py (BM25, dense, weighted hybrid, RRF) and rerank.py
            (cross-encoder over the RRF top results), called the way
            run_baselines.py calls them, with the resume text as the query
  facts     extraction.py records in DATA/extracted/*.jsonl. A document with
            no record is extracted on demand when ASU_VOYAGER_KEY is set.
  skills    skill_linking.Linker links each skill fact to Lightcast, and
            skill_linking.check_missing compares a job's skills with a resume,
            searching the resume text before a skill is called missing

The assistant adds no scoring of its own. Ranks and scores are the
retriever's; the skill table is evidence shown next to them.
"""

import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import extraction as ex  # noqa: E402
import rerank as rr  # noqa: E402
import retrieval as rt  # noqa: E402
import skill_linking as sl  # noqa: E402

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # same as run_baselines.DEFAULT_MODEL
VOYAGER_URL = "https://openai.rc.asu.edu/v1"
VOYAGER_MODEL = "qwen3-235b-a22b"

# Relative paths are from the repository root.
PATHS = {
    "processed": "DATA/processed",                 # postings_cleaned.csv, resumes_cleaned.csv
    "extracted": "DATA/extracted",                 # resumes.jsonl, postings.jsonl from extraction.run_batch
    "skills": sl.DEFAULT_CACHE,                    # Lightcast Open Skills cache
    "aliases": sl.DEFAULT_ALIASES,
    "embeddings": "DATA/embeddings/postings.npz",  # run_baselines --cache
}
SAMPLE_PATHS = dict(PATHS, processed="CODE/sample_data", extracted="CODE/sample_data/extracted",
                    skills="CODE/sample_data/skills.json", embeddings="DATA/embeddings/sample_postings.npz")

METHODS = {
    "bm25": "BM25",
    "dense": "Dense (SBERT)",
    "hybrid": "Weighted hybrid (TF-IDF + SBERT)",
    "rrf": "RRF (BM25 + dense)",
    "rerank": "RRF + cross-encoder rerank",
}
POOL = 100       # results per ranker before fusion (run_baselines --k)
KEEP = 50        # ranked jobs kept per resume and method
GAP_JOBS = 5     # jobs the skill-gap summary looks at


class SetupError(Exception):
    """A required input file is missing."""


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _val(x):
    """None for blanks and pandas NaN."""
    if x is None or (isinstance(x, float) and x != x):
        return None
    return (x.strip() or None) if isinstance(x, str) else x


def voyager_extractor(model=None):
    """extraction.Extractor on ASU Voyager, or None without a key or the openai package."""
    key = os.environ.get("ASU_VOYAGER_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        print("[assistant] ASU_VOYAGER_KEY is set but openai is not installed; no on-demand extraction")
        return None
    client = OpenAI(api_key=key, base_url=os.environ.get("VOYAGER_BASE_URL", VOYAGER_URL))
    return ex.Extractor(client, model or os.environ.get("VOYAGER_MODEL", VOYAGER_MODEL))


# ================================================================ ranking

class JobIndex:
    """The team's rankers over the postings, built once.

    Without sentence-transformers (or with dense_model=None) only BM25 is
    available. encoder and reranker can be passed in (the tests use fakes).
    """

    def __init__(self, ids, texts, dense_model=DENSE_MODEL, cache=None, rerank_model=rr.DEFAULT_MODEL,
                 rerank_top=rr.DEFAULT_TOP_N, encoder=None, reranker=None):
        self.ids = list(ids)
        self.text = dict(zip(self.ids, texts))
        self.bm25 = rt.BM25Index(self.ids, texts)
        self.encoder = self.dense = self.tfidf = self.reranker = None
        self.rerank_top = rerank_top
        self.notes = []
        if encoder is None and dense_model:
            try:
                from sentence_transformers import SentenceTransformer
                import run_baselines
                self.dense = run_baselines.build_dense_index(self.ids, texts, cache, dense_model)
                encoder = SentenceTransformer(dense_model)
            except ImportError as exc:
                self.notes.append(f"dense ranking is off ({exc.name} is not installed)")
                self.dense = None
        elif encoder is not None:
            self.dense = rt.DenseIndex(self.ids, rt.encode_texts(texts, model=encoder))
        if self.dense is None:
            return
        self.encoder = encoder
        try:
            self.tfidf = rt.TfidfIndex(self.ids, texts)
        except ImportError:
            self.notes.append("weighted hybrid is off (scikit-learn is not installed)")
        if reranker is not None or rerank_model:
            self.reranker = reranker or rr.CrossEncoderReranker(rerank_model)

    @property
    def methods(self):
        out = ["bm25"]
        if self.dense is not None:
            out += ["dense"] + (["hybrid"] if self.tfidf is not None else []) + ["rrf"]
            out += ["rerank"] if self.reranker is not None else []
        return out

    def rank(self, query, method, k=KEEP):
        """[(job_id, score)], best first."""
        if method not in self.methods:
            raise ValueError(f"ranking method {method!r} is not available here")
        if method == "bm25":
            return self.bm25.search(query, k)
        qvec = rt.encode_texts([query], model=self.encoder)[0]
        if method == "dense":
            return self.dense.search(qvec, k)
        if method == "hybrid":
            return rt.weighted_hybrid(self.tfidf.similarities(query), self.dense.similarities(qvec),
                                      self.dense.ids, k=k)
        pool = max(k, POOL)
        fused = rt.rrf([self.bm25.search(query, pool), self.dense.search(qvec, pool)], top=pool)
        if method == "rrf":
            return fused[:k]
        out = rr.rerank_run({"q": fused}, {"q": query}, self.text, self.reranker, self.rerank_top)["q"]
        return [(jid, raw) for jid, raw, _ in out][:k]


# ================================================================ facts

class FactStore:
    """Extraction records, linked to Lightcast the first time they are used."""

    def __init__(self, extracted_dir, linker, extractor=None):
        self.dir, self.linker, self.extractor = extracted_dir, linker, extractor
        self.records, self._linked = {}, {}
        for kind in ex.KINDS:
            if os.path.exists(self._path(kind)):
                for rec in ex.load_records(self._path(kind)):
                    if rec.get("status") == "ok":
                        self.records[(kind, str(rec["doc_id"]))] = rec

    def _path(self, kind):
        return os.path.join(self.dir, f"{kind}s.jsonl")

    def count(self, kind):
        return sum(1 for k, _ in self.records if k == kind)

    def has(self, kind, doc_id):
        return (kind, str(doc_id)) in self.records

    def get(self, kind, doc_id, text=None, extract=True, save=True):
        """The linked record of one document, or None.

        With extract and an extractor, a document with no record is extracted
        now; with save the record is appended to the JSONL file, so
        extraction.run_batch will skip it later.
        """
        key = (kind, str(doc_id))
        if key in self._linked:
            return self._linked[key]
        rec = self.records.get(key)
        if rec is None and extract and self.extractor is not None and text:
            rec = self.extractor.extract(kind, doc_id, text)
            if rec.get("status") != "ok":
                print(f"[assistant] extraction of {kind} {doc_id} failed: {rec.get('error')}")
                return None
            self.records[key] = rec
            if save:
                self._append(kind, rec)
        if rec is None:
            return None
        self._linked[key] = sl.link_record(self.linker, rec)
        return self._linked[key]

    def _append(self, kind, rec):
        path = self._path(kind)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as out:
            if out.tell() > 0:
                with open(path, "rb") as f:
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b"\n":
                        out.write("\n")
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ================================================================ the shared backend

class Assistant:
    """Corpus, rankers, linker and facts, shared by every chat session."""

    def __init__(self, sample=False, paths=None, dense_model=DENSE_MODEL, rerank_model=rr.DEFAULT_MODEL,
                 rerank_top=rr.DEFAULT_TOP_N, use_llm=True, limit_postings=None, encoder=None,
                 reranker=None, extractor=None):
        self.paths = {k: resolve(v) for k, v in dict(SAMPLE_PATHS if sample else PATHS, **(paths or {})).items()}
        self._check_inputs()

        posts, resumes = rt.load_corpus(self.paths["processed"])
        if limit_postings:
            posts = posts.head(limit_postings)
        posts = posts.assign(job_id=posts["job_id"].astype(str))
        self.jobs = {r["job_id"]: r for r in posts.to_dict("records")}
        self.resumes = {}
        for r in resumes.to_dict("records"):
            text = r.get("Resume_clean")
            if isinstance(text, str) and text.strip():
                self.resumes[str(r["ID"])] = {"text": text, "category": _val(r.get("Category"))}
        self.uploads = {}

        skills = sl.load_skills(self.paths["skills"])
        aliases = sl.load_aliases(self.paths["aliases"]) if os.path.exists(self.paths["aliases"]) else {}
        self.linker = sl.Linker(skills, aliases)
        self.finder = sl.SkillTextFinder(skills, aliases)
        self.extractor = extractor if extractor is not None else (voyager_extractor() if use_llm else None)
        self.facts = FactStore(self.paths["extracted"], self.linker, self.extractor)

        ids = list(self.jobs)
        self.index = JobIndex(ids, [rt.posting_text(self.jobs[j]) for j in ids], dense_model,
                              self.paths["embeddings"], rerank_model, rerank_top, encoder, reranker)
        self._runs = {}

    def _check_inputs(self):
        p = self.paths
        need = {os.path.join(p["processed"], "postings_cleaned.csv"): "Step 1 of the notebook (preprocessing)",
                os.path.join(p["processed"], "resumes_cleaned.csv"): "Step 1 of the notebook (preprocessing)",
                p["skills"]: "skill_linking.download_and_cache (Lightcast Open Skills)"}
        missing = [f"  {os.path.relpath(f, ROOT)}  <- {how}" for f, how in need.items() if not os.path.exists(f)]
        if missing:
            raise SetupError("missing input files:\n" + "\n".join(missing))

    @property
    def methods(self):
        return self.index.methods

    def status(self):
        """One line on what this instance runs with."""
        parts = [f"{len(self.jobs):,} postings", f"{len(self.resumes):,} resumes",
                 f"extracted: {self.facts.count('posting'):,} postings / {self.facts.count('resume'):,} resumes",
                 "on-demand extraction: " + (self.extractor.model if self.extractor else "off (no ASU_VOYAGER_KEY)")]
        return "; ".join(parts + self.index.notes)

    # ------------------------------------------------------------ resumes
    def sample_resumes(self, limit=None):
        """[(label, resume_id)]; resumes with extracted facts first."""
        ids = sorted(self.resumes, key=lambda r: (not self.facts.has("resume", r), r))[:limit]
        return [(f"{r} · {self.resumes[r]['category'] or 'resume'}" + ("" if self.facts.has("resume", r)
                                                                      else " (not extracted)"), r) for r in ids]

    def add_upload(self, text):
        text = (text or "").strip()
        if len(text) < 30:
            raise ValueError("that resume text is too short")
        rid = "upload-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        self.uploads[rid] = text
        return rid

    def has_resume(self, rid):
        return rid in self.resumes or rid in self.uploads

    def resume_text(self, rid):
        return self.uploads[rid] if rid in self.uploads else self.resumes[rid]["text"]

    def resume_facts(self, rid, extract=True):
        # uploaded resumes are never written to disk
        return self.facts.get("resume", rid, self.resume_text(rid), extract=extract, save=rid not in self.uploads)

    # ------------------------------------------------------------ ranking and matching
    def ranked(self, rid, method):
        """Top KEEP jobs as [{rank, job_id, score}], cached per resume and method."""
        key = (rid, method)
        if key not in self._runs:
            hits = self.index.rank(self.resume_text(rid), method, KEEP)
            self._runs[key] = [{"rank": i, "job_id": jid, "score": float(s)} for i, (jid, s) in enumerate(hits, 1)]
        return self._runs[key]

    def match(self, rid, job_id, extract=True):
        """The job's skills checked against the resume, or None if the posting has no facts.

        Rows come from skill_linking.check_missing (matched, found_in_text,
        not_found, unresolved), merged per Lightcast skill, with the posting's
        span and the resume evidence attached.
        """
        job = self.facts.get("posting", job_id, self.jobs[job_id].get("description"), extract=extract)
        if job is None:
            return None
        text = self.resume_text(rid)
        res = self.resume_facts(rid, extract=extract)
        res_skills = res["facts"]["skills"] if res else []
        job_skills = job["facts"]["required_skills"]
        have = {}
        for f in res_skills:
            have.setdefault(f["link"].get("id"), f)
        rows, seen = [], {}
        for fact, row in zip(job_skills, sl.check_missing(job_skills, res_skills, text, self.finder)):
            key = row["skill_id"] or "nil:" + sl.normalize_name(row["job_skill"])
            if key in seen:
                if row["importance"] == "required":
                    seen[key]["importance"] = "required"
                continue
            row = dict(row, name=sl.strip_qualifier(row["lightcast"] or row["job_skill"]),
                       posting_span=fact.get("source_span"), evidence=None)
            if row["status"] == "matched":
                row["evidence"] = have[row["skill_id"]]["source_span"]
            elif row["status"] == "found_in_text":
                row["evidence"] = _around(text, *row["span"])
            seen[key] = row
            rows.append(row)
        return {"job_id": job_id, "skills": rows, "job_facts": job["facts"],
                "resume_facts": res["facts"] if res else None}


def _around(text, start, end, width=40):
    a, b = max(0, start - width), min(len(text), end + width)
    return ("..." if a else "") + text[a:b] + ("..." if b < len(text) else "")


# ================================================================ one chat session

HELP = """I rank job postings for a resume and show the skill evidence behind each match.

**Load a resume** (pick a sample, upload a .txt/.pdf, or paste it), then ask:
- `show my top jobs` (or `top 5`)
- `explain #3`: the skills job #3 asks for, and where your resume shows them
- `what skills am I missing`: skills the top jobs ask for that your resume does not show
- `compare these` (top 3) or `compare 1 4`: jobs side by side
- `compare methods`: where each ranker puts your top jobs
- `job 2`: details of job #2
- `my skills`: what was extracted from your resume
- `use bm25` / `use dense` / `use hybrid` / `use rrf` / `use rerank`: switch the ranker
"""

GREETINGS = {"help", "?", "hi", "hello", "hey", "commands"}
_METHOD_WORDS = [("rerank", r"re-?rank\w*|cross-?encoder"), ("hybrid", r"hybrid|tf-?idf"), ("rrf", r"rrf|fusion"),
                 ("dense", r"dense|sbert|embedding\w*"), ("bm25", r"bm-?25")]
STATUS = {"matched": "✓ in your skills", "found_in_text": "✓ in your resume text (extraction missed it)",
          "not_found": "✗ not found", "unresolved": "– not linked to Lightcast, not checked"}


def _q(text, limit=70):
    text = " ".join(str(text or "").split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _salary(job):
    lo, hi, med = (_val(job.get(k)) for k in ("min_salary", "max_salary", "med_salary"))
    period = (_val(job.get("pay_period")) or "").lower()
    if lo and hi:
        s = f"${lo:,.0f} to ${hi:,.0f}"
    elif med:
        s = f"${med:,.0f}"
    else:
        return None
    return f"{s} {period}".strip()


def _degree(d):
    return d["level"].replace("_", " ") + (f" in {d['field']}" if d.get("field") else "")


class Chat:
    """Per-user state (resume, ranking method) over a shared Assistant."""

    def __init__(self, assistant, resume_id=None, method=None):
        self.a = assistant
        self.resume_id = resume_id if resume_id and assistant.has_resume(resume_id) else None
        self.method = method if method in assistant.methods else self.default_method(assistant)

    @staticmethod
    def default_method(assistant):
        return "rrf" if "rrf" in assistant.methods else "bm25"

    # ------------------------------------------------------------ loading
    def use_resume(self, rid):
        rid = str(rid)
        if not self.a.has_resume(rid):
            raise KeyError(f"no resume {rid}")
        self.resume_id = rid
        return self.profile(loaded=True)

    def load_text(self, text):
        return self.use_resume(self.a.add_upload(text))

    # ------------------------------------------------------------ dialogue
    def reply(self, message):
        """Markdown answer to one message."""
        msg = (message or "").strip()
        low = msg.lower()
        if not msg or low in GREETINGS:
            return HELP
        if len(msg) > 500 or (len(msg) > 250 and "\n" in msg):
            return self.load_text(msg)
        method = self._method_named(low)
        if method and (low == method or re.search(r"\b(use|switch|method|change|rank(ing)?( by| with)?)\b", low)):
            return self.set_method(method)
        if self.resume_id is None:
            return "Load a resume first: pick a sample, upload a file or paste the text.\n\n" + HELP
        nums = [int(n) for n in re.findall(r"#?\b(\d{1,3})\b", low)]
        if re.search(r"\b(compare|comparison|vs|versus)\b", low):
            if re.search(r"\b(methods?|rankers?|ranking)\b", low):
                return self.compare_methods()
            return self.compare(nums)
        if re.search(r"\b(why|explain|reason|evidence)\b", low):
            return self.explain(nums[0] if nums else 1)
        if re.search(r"\b(gaps?|missing|lack\w*|learn|improve)\b", low):
            return self.gaps()
        if nums and re.search(r"\b(job|details?|about|describe|posting)\b", low) and not re.search(r"\bjobs\b", low):
            return self.details(nums[0])
        if re.search(r"\b(skills|profile|resume)\b", low) and not re.search(r"\bjobs?\b", low):
            return self.profile()
        if re.search(r"\b(jobs?|top|match\w*|recommend\w*|roles?|positions?|openings?|show|list)\b", low):
            return self.recommend(nums[0] if nums else 10)
        return "I didn't catch that.\n\n" + HELP

    @staticmethod
    def _method_named(low):
        for key, pattern in _METHOD_WORDS:
            if re.search(rf"\b({pattern})\b", low):
                return key
        return None

    def set_method(self, method):
        if method not in self.a.methods:
            return (f"`{METHODS.get(method, method)}` is not available in this run "
                    f"(available: {', '.join(self.a.methods)}).")
        self.method = method
        if self.resume_id is None:
            return f"Ranking with **{METHODS[method]}** from now on. Load a resume to see jobs."
        return self.recommend(10)

    def _ranked(self):
        return self.a.ranked(self.resume_id, self.method)

    def _job(self, n):
        jobs = self._ranked()
        if not 1 <= n <= len(jobs):
            raise IndexError(f"There are {len(jobs)} ranked jobs; pick a number from 1 to {len(jobs)}.")
        return jobs[n - 1]

    def _title(self, job_id):
        j = self.a.jobs[job_id]
        return _val(j.get("title")) or f"job {job_id}", _val(j.get("company_name")) or ""

    # ------------------------------------------------------------ answers
    def profile(self, loaded=False):
        rid, a = self.resume_id, self.a
        head = f"**Resume {rid}**" + (f" ({a.resumes[rid]['category']})" if rid in a.resumes and
                                      a.resumes[rid]["category"] else "") + (" loaded." if loaded else "")
        rec = a.resume_facts(rid)
        if rec is None:
            return "\n".join([head, "", "No extracted facts for this resume (no extraction record, and no "
                              "ASU_VOYAGER_KEY for on-demand extraction). Jobs are still ranked on the full text, "
                              "and each job skill is searched for in the resume text.", "",
                              "Ask `show my top jobs`."])
        f = rec["facts"]
        linked = sorted({sl.strip_qualifier(s["link"]["name"]) for s in f["skills"] if s["link"].get("id")})
        nil = sorted({s["name"] for s in f["skills"] if not s["link"].get("id")})
        roles = [e["title"] + (f" at {e['company']}" if e.get("company") else "") +
                 (f" ({e['duration_months']} months)" if e.get("duration_months") else "") for e in f["experience"]]
        lines = [head, "",
                 f"- **Skills linked to Lightcast ({len(linked)}):** {', '.join(linked) or 'none'}",
                 f"- **Not linked (kept as mentions):** {', '.join(nil) or 'none'}",
                 f"- **Degrees:** {', '.join(_degree(d) for d in f['degrees']) or 'none stated'}",
                 f"- **Experience:** {'; '.join(roles) or 'none stated'}",
                 "", f"_Facts extracted by `{rec.get('model')}`, every one grounded in a span of the resume._"]
        return "\n".join(lines + (["", "Ask `show my top jobs`."] if loaded else []))

    def _coverage(self, m, importance):
        rows = [r for r in m["skills"] if r["importance"] == importance and r["status"] != "unresolved"]
        return sum(r["status"] in ("matched", "found_in_text") for r in rows), len(rows)

    def recommend(self, n=10):
        jobs = self._ranked()[: max(1, min(n, KEEP))]
        if not jobs:
            return "No postings matched this resume."
        lines = [f"**Top {len(jobs)} jobs**, ranked by {METHODS[self.method]}", "",
                 "| # | Job | Company | Location | Score | Required skills you show |", "|---|---|---|---|---|---|"]
        for j in jobs:
            title, company = self._title(j["job_id"])
            m = self.a.match(self.resume_id, j["job_id"], extract=False)
            have, total = self._coverage(m, "required") if m else (0, 0)
            cover = f"{have} of {total}" if m and total else ("none linked" if m else "not extracted")
            lines.append(f"| {j['rank']} | {_q(title, 50)} | {_q(company, 30)} | "
                         f"{_q(_val(self.a.jobs[j['job_id']].get('location')) or '', 30)} | {j['score']:.4g} | {cover} |")
        lines += ["", "_Score is the ranker's own (scales differ between methods). The skill column only "
                  "covers postings already extracted._", "",
                  "Next: `explain #1`, `job 2`, `what skills am I missing`, `compare these`."]
        return "\n".join(lines)

    def explain(self, n):
        try:
            j = self._job(n)
        except IndexError as exc:
            return str(exc)
        title, company = self._title(j["job_id"])
        lines = [f"**#{n} {title}**" + (f" at {company}" if company else ""),
                 f"Ranked #{n} by {METHODS[self.method]} (score {j['score']:.4g}). The rank comes from text "
                 "similarity between your resume and the posting; below is the skill evidence.", ""]
        m = self.a.match(self.resume_id, j["job_id"])
        if m is None:
            return "\n".join(lines + ["This posting has no extracted facts yet, so I can't list its skills. "
                                      "Set ASU_VOYAGER_KEY to extract it on demand."])
        if not m["skills"]:
            lines.append("Extraction found no skill requirements in this posting.")
        else:
            lines += ["| Skill (Lightcast) | Importance | Posting says | Your resume |", "|---|---|---|---|"]
            order = {"required": 0, "preferred": 1}
            for r in sorted(m["skills"], key=lambda r: (order.get(r["importance"], 2), r["status"] == "unresolved")):
                ev = f': "{_q(r["evidence"], 60)}"' if r["evidence"] else ""
                lines.append(f"| {_q(r['name'], 40)} | {r['importance']} | \"{_q(r['posting_span'], 50)}\" | "
                             f"{STATUS[r['status']]}{ev} |")
            req, pref = self._coverage(m, "required"), self._coverage(m, "preferred")
            lines += ["", f"You show **{req[0]} of {req[1]}** required and **{pref[0]} of {pref[1]}** preferred "
                      "linked skills."]
        jf, rf = m["job_facts"], m["resume_facts"]
        other = [f"- Degree: {_degree(d)} ({d['importance']}): \"{_q(d['source_span'], 60)}\""
                 for d in jf.get("required_degrees") or []]
        if jf.get("experience_required"):
            e = jf["experience_required"]
            other.append(f"- Experience: {e['min_years']:g}+ years ({e['importance']}): \"{_q(e['source_span'], 60)}\"")
        if other:
            lines += ["", "**Other requirements in the posting**"] + other
            if rf:
                lines.append(f"- Your resume: degrees {', '.join(_degree(d) for d in rf['degrees']) or 'not stated'}; "
                             f"{len(rf['experience'])} role(s) listed")
        if rf is None:
            lines += ["", "_Your resume has no extracted facts, so each skill was searched for in its text._"]
        return "\n".join(lines)

    def gaps(self):
        jobs = self._ranked()[:GAP_JOBS]
        agg, skipped = {}, 0
        for j in jobs:
            m = self.a.match(self.resume_id, j["job_id"])
            if m is None:
                skipped += 1
                continue
            for r in m["skills"]:
                if r["status"] != "not_found":
                    continue
                g = agg.setdefault(r["skill_id"], {"name": r["name"], "required": 0, "jobs": []})
                g["required"] += r["importance"] == "required"
                g["jobs"].append(j["rank"])
        n = len(jobs) - skipped
        if not n:
            return "None of your top jobs have extracted facts yet, so I can't list missing skills."
        if not agg:
            return f"Your resume shows every linked skill your top {n} jobs ask for."
        lines = [f"**Skills your top {n} jobs ask for that your resume does not show** ({METHODS[self.method]})", "",
                 "| Skill (Lightcast) | Required in | Asked by jobs |", "|---|---|---|"]
        for g in sorted(agg.values(), key=lambda g: (-g["required"], -len(g["jobs"]), g["name"])):
            lines.append(f"| {g['name']} | {g['required']} | {', '.join('#' + str(r) for r in g['jobs'])} |")
        lines += ["", "_Each skill was also searched for in your resume text before being listed. "
                  "Add one only if you really have it._"]
        if skipped:
            lines.append(f"_{skipped} of the top {len(jobs)} jobs have no extracted facts and were left out._")
        return "\n".join(lines)

    def compare(self, nums=None):
        nums = list(dict.fromkeys(nums or [])) or [1, 2, 3]
        if len(nums) < 2:
            return "Name at least two jobs, e.g. `compare 1 3`."
        try:
            picked = [self._job(n) for n in nums[:4]]
        except IndexError as exc:
            return str(exc)
        cols = []
        for j in picked:
            job, m = self.a.jobs[j["job_id"]], self.a.match(self.resume_id, j["job_id"])
            title, company = self._title(j["job_id"])
            col = {"Job": title, "Company": company, "Location": _val(job.get("location")),
                   "Work type": _val(job.get("formatted_work_type")),
                   "Experience level": _val(job.get("formatted_experience_level")),
                   "Salary": _salary(job), f"Score ({self.method})": f"{j['score']:.4g}"}
            if m is None:
                col["Required skills you show"] = "not extracted"
            else:
                for imp in ("required", "preferred"):
                    have, total = self._coverage(m, imp)
                    col[f"{imp.title()} skills you show"] = f"{have} of {total}"
                    col[f"Missing ({imp})"] = ", ".join(r["name"] for r in m["skills"] if r["importance"] == imp
                                                        and r["status"] == "not_found") or "none"
            cols.append(col)
        rows = list(dict.fromkeys(k for c in cols for k in c))
        lines = ["| | " + " | ".join(f"#{j['rank']}" for j in picked) + " |", "|---" * (len(picked) + 1) + "|"]
        for r in rows:
            lines.append(f"| **{r}** | " + " | ".join(_q(c.get(r) or "–", 60) for c in cols) + " |")
        return "\n".join(lines + ["", "Ask `explain #N` for the evidence behind any of them."])

    def compare_methods(self):
        methods = self.a.methods
        runs = {m: {j["job_id"]: j["rank"] for j in self.a.ranked(self.resume_id, m)} for m in methods}
        top = self._ranked()[:10]
        lines = [f"**Where each ranker puts your top 10** (current: {METHODS[self.method]})", "",
                 "| Job | " + " | ".join(methods) + " |", "|---" * (len(methods) + 1) + "|"]
        for j in top:
            title, company = self._title(j["job_id"])
            label = _q(title, 40) + (f" ({_q(company, 25)})" if company else "")
            lines.append(f"| {label} | " + " | ".join(str(runs[m].get(j["job_id"], f">{KEEP}")) for m in methods) + " |")
        if len(methods) == 1:
            lines += ["", "_Only BM25 is available in this run; install sentence-transformers for the others._"]
        return "\n".join(lines)

    def details(self, n):
        try:
            j = self._job(n)
        except IndexError as exc:
            return str(exc)
        job = self.a.jobs[j["job_id"]]
        title, company = self._title(j["job_id"])
        info = " · ".join(x for x in (_val(job.get("location")), _val(job.get("formatted_work_type")),
                                      _val(job.get("formatted_experience_level")), _salary(job)) if x)
        lines = [f"**#{n} {title}**" + (f" at {company}" if company else "") + f" (job {j['job_id']})",
                 f"_{info}_" if info else ""]
        rec = self.a.facts.get("posting", j["job_id"], job.get("description"))
        if rec:
            skills = rec["facts"]["required_skills"]
            for imp in ("required", "preferred"):
                names = list(dict.fromkeys(sl.strip_qualifier(s["link"]["name"] or s["name"]) for s in skills
                                           if s.get("importance") == imp))
                lines.append(f"- **{imp.title()} skills:** {', '.join(names) or 'none'}")
        desc = " ".join(str(_val(job.get("description")) or "").split())
        lines += ["", "> " + (desc[:700] + ("..." if len(desc) > 700 else ""))]
        return "\n".join(lines)
