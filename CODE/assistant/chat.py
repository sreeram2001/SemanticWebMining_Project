"""Chat agent: answers come from grounded facts; LLM replies are shown only if verified."""

import os
import re

from core.config import read_json_or
from linking.entity_linking import NIL
from assistant.explanations import explain, skill_gaps, verify

HELP = """I match your resume to job postings using a skill knowledge graph.

**Load a resume** (paste it, upload a file, or pick a sample), then ask:
- `jobs` or `top 5`: best matching jobs
- `why 2` / `explain 2`: why job #2 matches, with evidence from both texts
- `gaps`: skills the top jobs ask for that your resume does not show
- `skills`: the skills I extracted from your resume
- `job 3`: details of job #3
- `compare`: ranking with vs. without the graph score (ablation)
- `method bm25|dense|weighted_hybrid|rrf|full`: switch the ranker
"""

_NUM = re.compile(r"#?\b(\d{1,3})\b")


class CareerAgent:
    """Per-user state: current resume, ranking method, last results."""

    def __init__(self, pipeline, scoring=None, use_llm=True, top=10):
        self.p = pipeline
        self.p.graph_or_build()
        self.scoring = scoring if scoring is not None else self._tuned()
        self.top = top
        self.method = "full"
        self.resume_id = None
        self.jobs = []
        self.client, self.model = pipeline.llm_client() if use_llm else (None, None)

    def session(self, resume_id=None, method="full"):
        """Per-user copy sharing the pipeline."""
        import copy
        s = copy.copy(self)
        s.resume_id, s.method, s.jobs = resume_id, method or "full", []
        return s

    def _tuned(self):
        path = os.path.join(self.p.p["out_dir"], "tuned_scoring.json")
        return (read_json_or(path, {}, "tuned scoring") or {}).get("scoring")

    @property
    def ranker(self):
        return self.p.ranker

    # ------------------------------------------------------------------ resume
    def sample_resumes(self, limit=50):
        """[(label, resume_id)] for graph resumes."""
        g = self.p.graph
        ids = [r for r in g.doc_ids("resume") if not r.startswith("user-")][:limit]
        return [(f"{r} ({g.meta[r].get('category') or 'resume'})", r) for r in ids]

    def use_resume(self, resume_id):
        resume_id = str(resume_id)
        if resume_id not in self.p.graph.texts:
            raise KeyError(f"resume {resume_id} is not in the graph")
        self.resume_id, self.jobs = resume_id, []
        return self._profile(loaded=True)

    def upload(self, text):
        rid = self.p.add_resume(text)
        return self.use_resume(rid)

    # ------------------------------------------------------------------ dialogue
    def reply(self, message):
        """Markdown answer for one user message."""
        msg = (message or "").strip()
        low = msg.lower()
        if not msg or low in ("help", "?", "hi", "hello", "hey"):
            return HELP
        if self.resume_id is None:
            if len(msg) > 200:                           # a pasted resume
                return self.upload(msg)
            return "Please load a resume first (paste it, upload a file or pick a sample).\n\n" + HELP
        if low.startswith("method"):
            return self._set_method(low.split()[-1])
        n = self._number(low)
        if re.search(r"\b(why|explain|reason)\b", low):
            return self.explain(n or 1)
        if re.search(r"\b(gap|gaps|missing|lack|learn|improve)\b", low):
            return self.gaps()
        if re.search(r"\b(compare|ablation|without graph|no graph)\b", low):
            return self.compare()
        if re.fullmatch(r"(show |list )?(my )?(skills|profile)\??", low):
            return self._profile()
        if re.search(r"\bjob\b", low) and n:
            return self.job_details(n)
        if re.search(r"\b(jobs?|top|match|matches|recommend|rank|roles?|positions?|openings?)\b", low):
            return self.recommend(n or self.top)
        if len(msg) > 200:
            return self.upload(msg)
        return self._free_form(msg)

    @staticmethod
    def _number(text):
        m = _NUM.search(text)
        return int(m.group(1)) if m else None

    def _set_method(self, name):
        from rank.ranking import METHODS
        if name not in METHODS:
            return f"Unknown method `{name}`. Choose one of: {', '.join(METHODS)}."
        self.method, self.jobs = name, []
        return f"Ranking with `{name}` now.\n\n" + self.recommend(self.top)

    def _ranked(self, n=None):
        n = max(n or self.top, self.top)
        if len(self.jobs) < n:
            self.jobs = self.p.recommend(self.resume_id, self.method, top=n, scoring=self.scoring)["jobs"]
        return self.jobs

    def _job(self, n):
        jobs = self._ranked(n)
        if not 1 <= n <= len(jobs):
            raise IndexError(f"there are only {len(jobs)} ranked jobs")
        return jobs[n - 1]

    # ------------------------------------------------------------------ answers
    def _profile(self, loaded=False):
        g, rid = self.p.graph, self.resume_id
        facts = g.facts.get(rid, {})
        linked = [self.ranker.index.display_name(s) for s in g.linked_skills(rid)]
        unlinked = sorted({f["name"] for f in facts.get("skills", []) if f["skill_id"] == NIL})
        degrees = [d["level"] + (f" in {d['field']}" if d.get("field") else "") for d in facts.get("degrees", [])]
        years = self.ranker.resume_years(rid)
        lines = [f"**Resume {rid}**" + (" loaded." if loaded else ""), "",
                 f"- Skills linked to Lightcast ({len(linked)}): " + (", ".join(linked) or "none"),
                 f"- Unlinked mentions (shown, not scored): {', '.join(unlinked) or 'none'}",
                 f"- Degrees: {', '.join(degrees) or 'not stated'}",
                 f"- Experience: {f'{years:g} years' if years is not None else 'unknown (no dates found)'}",
                 "", "Ask `jobs` to see your best matches." if loaded else ""]
        return "\n".join(lines)

    def recommend(self, n=None):
        jobs = self._ranked(n)[: n or self.top]
        if not jobs:
            return "No jobs could be ranked."
        graph_cols = "G" in jobs[0]
        head = "| # | Job | Company | Score |" + (" Skill fit G | Penalty P |" if graph_cols else "")
        sep = "|---|---|---|---|" + ("---|---|" if graph_cols else "")
        rows = [f"Top {len(jobs)} jobs (`{self.method}`):", "", head, sep]
        for j in jobs:
            row = f"| {j['rank']} | {j['title'] or j['job_id']} | {j['company'] or ''} | {j['score']:.3f} |"
            if graph_cols:
                row += f" {j['G']:.2f} | {j['P']:.2f} |"
            rows.append(row)
        rows += ["", "Ask `why 1` for the evidence behind a match, or `gaps` for skills to add."]
        return "\n".join(rows)

    def explain(self, n):
        try:
            job = self._job(n)
        except IndexError as exc:
            return str(exc)
        e = explain(self.ranker, self.resume_id, job["job_id"], self.scoring, self.client, self.model)
        badge = "verified against stored facts" if e["verified"] else \
            "NOT verified: mentions " + ", ".join(e["unsupported_skills"])
        return f"**#{n}: why it matches** ({badge})\n\n```\n{e['text']}\n```"

    def gaps(self):
        jobs = self._ranked()
        ranked = [{"job_id": j["job_id"]} for j in jobs]
        gaps = skill_gaps(self.ranker, self.resume_id, ranked, top=min(5, len(ranked)), scoring=self.scoring)
        if not gaps:
            return "Your resume covers every linked skill the top jobs ask for."
        lines = [f"Skills the top {min(5, len(ranked))} jobs ask for that your resume does not show:", ""]
        for g in gaps:
            hint = f" (closest you have: {g['closest']})" if g["closest"] else ""
            lines.append(f"- **{g['skill']}**: required in {g['required_in']}, asked in {len(g['jobs'])}{hint}")
        lines += ["", "Skills not found carry no penalty; add them only if you have them."]
        return "\n".join(lines)

    def job_details(self, n):
        try:
            job = self._job(n)
        except IndexError as exc:
            return str(exc)
        g = self.p.graph
        meta, text = g.meta.get(job["job_id"], {}), g.texts.get(job["job_id"], "")
        reqs = g.skill_requirements(job["job_id"])
        req = [self.ranker.index.display_name(r["skill_id"]) if r["skill_id"] != NIL else r["name"]
               for r in reqs if r["importance"] == "required"]
        pref = [self.ranker.index.display_name(r["skill_id"]) if r["skill_id"] != NIL else r["name"]
                for r in reqs if r["importance"] == "preferred"]
        info = ", ".join(v for v in (meta.get("location"), meta.get("experience_level"), meta.get("work_type")) if v)
        snippet = " ".join(text.split())[:600]
        return "\n".join([f"**#{n}: {meta.get('title') or job['job_id']}** at {meta.get('company_name') or '?'}",
                          f"_{info}_" if info else "", "",
                          f"- Required skills: {', '.join(req) or 'none extracted'}",
                          f"- Preferred skills: {', '.join(pref) or 'none'}", "", f"> {snippet}..."])

    def compare(self):
        """Graph ablation: full vs. full_no_graph."""
        full = self.p.recommend(self.resume_id, "full", top=self.top, scoring=self.scoring)["jobs"]
        base = self.p.recommend(self.resume_id, "full_no_graph", top=self.top, scoring=self.scoring)["jobs"]
        pos = {j["job_id"]: j["rank"] for j in base}
        lines = ["Ranking **with** the graph score vs. **without** it (beta2 = 0):", "",
                 "| # | Job | Rank without graph | G |", "|---|---|---|---|"]
        for j in full:
            lines.append(f"| {j['rank']} | {j['title'] or j['job_id']} | {pos.get(j['job_id'], '>' + str(self.top))} "
                         f"| {j.get('G', 0):.2f} |")
        return "\n".join(lines)

    def _free_form(self, question):
        """LLM answer from top-job facts, shown only if verified."""
        if self.client is None:
            return "I did not understand that. " + HELP
        jobs = self._ranked()[:5]
        facts, allowed, ignore = [], set(self.p.graph.linked_skills(self.resume_id)), []
        for j in jobs:
            m = self.ranker.match(self.resume_id, j["job_id"], self.scoring)
            allowed |= {x["skill_id"] for x in m["skills"] if x["skill_id"] != NIL}
            ignore += [j["title"] or "", j["company"] or ""]
            facts.append(f"#{j['rank']} {j['title']} at {j['company']}: " + "; ".join(
                f"{x['name']}={x['status']}" for x in m["skills"]))
        prompt = ("Answer the job seeker's question using only these facts about their top job matches. "
                  "Do not name any skill that is not listed. If the facts do not answer it, say so.\n\n"
                  + "\n".join(facts) + f"\n\nQuestion: {question}")
        try:
            r = self.client.chat.completions.create(model=self.model, temperature=0.0, max_tokens=600,
                                                    messages=[{"role": "user", "content": prompt}])
            from extract.extraction import _THINK
            answer = _THINK.sub("", r.choices[0].message.content or "").strip()
        except Exception as exc:
            return f"The LLM is unavailable ({type(exc).__name__}). " + HELP
        bad = verify(answer, allowed, self.ranker.gazetteer, ignore)
        if bad:
            return "I could not give a fact-checked answer to that. " + HELP
        return answer + "\n\n_(checked against the stored facts)_"
