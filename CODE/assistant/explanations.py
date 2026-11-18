"""Grounded match explanations and skill gaps; verify() rejects unsupported skill mentions."""

from linking.entity_linking import NIL

ICON = {"met": "[met]", "recovered": "[met*]", "related": "[partial]", "not_found": "[not found]",
        "unresolved": "[unresolved]", "conflict": "[conflict]", "unknown": "[unknown]"}


def _q(text, limit=80):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


def build(graph, match):
    """Explain one pair -> (text, allowed_skill_ids, ignore)."""
    meta = graph.meta.get(match["job_id"], {})
    title, company = meta.get("title") or match["job_id"], meta.get("company_name") or ""
    lines = [f"{title} at {company} (job {match['job_id']})",
             f"Graph skill coverage G = {match['G']:.2f}; hard-requirement penalty P = {match['P']:.2f}"]
    if match["insufficient_evidence"]:
        lines.append("None of this posting's skills could be linked to Lightcast, so G is 0 for lack of evidence.")
    for heading, imp in (("Required skills", "required"), ("Preferred skills", "preferred")):
        items = [x for x in match["skills"] if x["importance"] == imp]
        if not items:
            continue
        lines.append(f"{heading}:")
        for x in items:
            asked = f'posting: "{_q(x["requirement_span"]["text"], 40)}"'
            if x["status"] == "met":
                why = f'your resume lists it: "{_q(x["evidence"]["text"])}"'
            elif x["status"] == "recovered":
                why = (f'found in your resume text ("{_q(x["evidence"]["text"])}", {x["evidence"]["how"]}) '
                       "although the extractor missed it")
            elif x["status"] == "related":
                why = (f"related skill {x['related_skill']} (same {x['relation']}, partial credit "
                       f'{x["credit"]:.1f}): "{_q(x["evidence"]["text"])}". Related is not the same skill')
            elif x["status"] == "not_found":
                why = "not found in your resume (no penalty; add it if you have it)"
            else:
                why = "could not be linked to a Lightcast skill, so it is not scored"
            lines.append(f"  {ICON[x['status']]} {x['name']}: {why} ({asked})")
    if match["hard"]:
        lines.append("Hard requirements:")
        for h in match["hard"]:
            need = f'"{_q(h["requirement_span"]["text"], 50)}"'
            tag = "" if h["importance"] == "required" else " (preferred, not penalized)"
            if h["status"] == "unknown":
                lines.append(f"  {ICON['unknown']} {h['kind']}: posting asks for {need}; "
                             f"your resume does not say, so there is no penalty")
            else:
                have = h["have"] if h["kind"] == "degree" else f"{h['have']:g} years"
                lines.append(f"  {ICON[h['status']]} {h['kind']}: posting asks for {need}; "
                             f"your resume shows {have}{tag}")
    allowed = {x["skill_id"] for x in match["skills"] if x["skill_id"] != NIL}
    allowed |= set(graph.linked_skills(match["resume_id"]))
    # quoted spans aren't new claims
    quoted = [x["requirement_span"]["text"] for x in match["skills"]]
    quoted += [x["evidence"]["text"] for x in match["skills"] if x.get("evidence")]
    quoted += [h["requirement_span"]["text"] for h in match["hard"]]
    return "\n".join(lines), allowed, [title, company] + quoted


def mentioned_skills(text, gazetteer):
    out = set()
    for _, _, ids in gazetteer.find(text):
        out.update(ids)
    return out


def verify(text, allowed, gazetteer, ignore=()):
    """Skill IDs in text that are not stored facts (empty = verified); `ignore` strings removed first."""
    for s in ignore:
        if s:
            text = text.replace(s, " ")
    problems = []
    for _, _, ids in gazetteer.find(text):
        if not set(ids) & set(allowed):
            problems.append(ids[0])
    return sorted(set(problems))


def explain(ranker, resume_id, job_id, scoring=None, llm_client=None, llm_model=None):
    """Verified explanation dict for one pair."""
    gaz = ranker.gazetteer
    m = ranker.match(resume_id, job_id, scoring)
    text, allowed, ignore = build(ranker.g, m)
    source = "template"
    if llm_client is not None:
        try:
            reply = llm_client.chat.completions.create(
                model=llm_model, temperature=0.0, max_tokens=800,
                messages=[{"role": "system", "content": "Rewrite this job-match explanation so it reads naturally. "
                           "Do not add any skill, tool, degree or claim that is not in it. Keep the quotes."},
                          {"role": "user", "content": text}])
            new = reply.choices[0].message.content
            if new and not verify(new, allowed, gaz, ignore):
                text, source = new, "llm"
        except Exception as exc:
            print(f"[explanations] LLM rewrite skipped ({type(exc).__name__})")
    problems = verify(text, allowed, gaz, ignore)
    return {"resume_id": resume_id, "job_id": job_id, "text": text, "source": source,
            "verified": not problems, "unsupported_skills": problems, "G": m["G"], "P": m["P"]}


def skill_gaps(ranker, resume_id, ranked, top=5, limit=8, scoring=None):
    """Skills top jobs want that the resume lacks, required first."""
    agg = {}
    for row in ranked[:top]:
        m = ranker.match(resume_id, row["job_id"], scoring)
        for x in m["skills"]:
            if x["status"] not in ("not_found", "related"):
                continue
            a = agg.setdefault(x["skill_id"], {"skill_id": x["skill_id"], "skill": x["name"], "jobs": [],
                                               "required_in": 0, "closest": None})
            a["jobs"].append(row["job_id"])
            a["required_in"] += x["importance"] == "required"
            # hint only if a sibling exists
            if x["status"] == "related" and x["relation"] == "subcategory" and not a["closest"]:
                a["closest"] = x["related_skill"]
    return sorted(agg.values(), key=lambda a: (-a["required_in"], -len(a["jobs"]), a["skill"]))[:limit]
