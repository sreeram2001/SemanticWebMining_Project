"""Graph-aware scoring: S(r, j) = b1*CE(r, j) + b2*G(r, j) - b3*P(r, j).

Inputs are the records on main, so no new format is needed:
  * extraction.run_batch records (Richitha), after skill_linking.link_record
    (Kaushal) has added a "link" to every skill fact:
      resume:  facts.skills, facts.degrees, facts.experience
      posting: facts.required_skills, facts.required_degrees,
               facts.experience_required
  * G uses skill_linking.coverage(): weighted coverage of the job's linked
    skills with partial credit for the same Lightcast subcategory or category.
    Its lambda placeholders are the defaults here; tune() picks real values.

This module adds what ranking owns: the penalty P, the final score S, the
no-graph ablation (b2 = 0) and rank_candidates() for a reranked shortlist.

Missing information counts as unknown, never as a mismatch. Only facts marked
"required" are hard requirements; "preferred" ones never cost anything.
"""

import skill_linking as sl

DEGREE_RANK = {"high_school": 0, "associate": 1, "bachelors": 2, "masters": 3, "phd": 4}

# Placeholders until tuned on the dev split. b2 = 0 is the ablation.
DEFAULT_BETAS = {"b1": 1.0, "b2": 0.5, "b3": 0.5}
DEFAULT_LAMBDAS = {"lam_sub": sl.LAMBDA_SUB, "lam_cat": sl.LAMBDA_CAT}


def _facts(record):
    if not record or record.get("status", "ok") != "ok":
        return {}
    return record.get("facts") or {}


# ------------------------------------------------------------------ G score
def graph_score(
    job_record,
    resume_record,
    lam_sub=sl.LAMBDA_SUB,
    lam_cat=sl.LAMBDA_CAT,
    weights=None,
):
    """G in [0, 1] with its evidence rows, via skill_linking.coverage().

    insufficient_evidence is True when the job has no linked skills; G is then
    0 and the job's unlinked skills are listed in "unresolved".
    """
    if not 0 < lam_cat < lam_sub < 1:
        raise ValueError("need 0 < lam_cat < lam_sub < 1")
    job_skills = [
        f for f in _facts(job_record).get("required_skills") or [] if "link" in f
    ]
    resume_skills = [
        f for f in _facts(resume_record).get("skills") or [] if "link" in f
    ]
    g, rows, unresolved = sl.coverage(
        job_skills, resume_skills, weights or sl.WEIGHTS, lam_sub, lam_cat
    )
    return {
        "score": g,
        "rows": rows,
        "unresolved": unresolved,
        "insufficient_evidence": not rows,
    }


# ------------------------------------------------------------------ penalty
def resume_years(resume_facts):
    """Total years from experience durations, or None if any entry is undated.

    A partial sum would understate experience and cause false mismatches, so
    one entry without duration_months makes the total unknown.
    """
    exp = resume_facts.get("experience") or []
    if not exp or any(e.get("duration_months") is None for e in exp):
        return None
    return sum(e["duration_months"] for e in exp) / 12.0


def penalty(job_record, resume_record):
    """Share of the job's hard requirements the resume explicitly fails.

    Hard requirements: the required degree (the lowest level the posting
    accepts, since "bachelor's or master's" lists both) and the required
    minimum years. Each is met, mismatch or unknown; only mismatches count.
    """
    job, res = _facts(job_record), _facts(resume_record)
    details = []

    levels = [
        DEGREE_RANK[d["level"]]
        for d in job.get("required_degrees") or []
        if d.get("importance") == "required" and d.get("level") in DEGREE_RANK
    ]
    if levels:
        need = min(levels)
        have = [
            DEGREE_RANK[d["level"]]
            for d in res.get("degrees") or []
            if d.get("level") in DEGREE_RANK
        ]
        status = "unknown" if not have else "met" if max(have) >= need else "mismatch"
        details.append(
            {
                "type": "degree",
                "required_rank": need,
                "resume_rank": max(have) if have else None,
                "status": status,
            }
        )

    exp = job.get("experience_required")
    if exp and exp.get("importance") == "required" and exp.get("min_years") is not None:
        years = resume_years(res)
        status = (
            "unknown"
            if years is None
            else "met"
            if years >= exp["min_years"]
            else "mismatch"
        )
        details.append(
            {
                "type": "experience",
                "min_years": exp["min_years"],
                "resume_years": years,
                "status": status,
            }
        )

    if not details:
        return {"score": 0.0, "details": []}
    bad = sum(d["status"] == "mismatch" for d in details)
    return {"score": bad / len(details), "details": details}


# -------------------------------------------------------------- final score
def final_score(ce, g, p, betas=None):
    b = {**DEFAULT_BETAS, **(betas or {})}
    return b["b1"] * ce + b["b2"] * g - b["b3"] * p


def without_graph(betas=None):
    """Same weights with b2 = 0: the graph ablation."""
    return {**DEFAULT_BETAS, **(betas or {}), "b2": 0.0}


def rank_candidates(candidates, job_records, resume_record, betas=None, lambdas=None):
    """Score a reranked shortlist by S and sort it, best first.

    `candidates`: [(job id, normalized CE score), ...] from rerank.py.
    `job_records`: {job id: linked posting record}. A job with no record
    gets G = 0 and P = 0 and is marked has_facts = False.
    """
    lam = {**DEFAULT_LAMBDAS, **(lambdas or {})}
    out = []
    for job_id, ce in candidates:
        rec = job_records.get(str(job_id))
        g = graph_score(rec, resume_record, **lam)
        p = penalty(rec, resume_record)
        out.append(
            {
                "job_id": str(job_id),
                "ce": ce,
                "g": g["score"],
                "p": p["score"],
                "score": final_score(ce, g["score"], p["score"], betas),
                "has_facts": rec is not None,
                "graph": g,
                "penalty": p,
            }
        )
    return sorted(out, key=lambda r: -r["score"])  # stable: ties keep CE order


def load_linked(path):
    """{doc id: record} from a linked JSONL file (latest record per document)."""
    return {str(r["doc_id"]): r for r in sl.latest_records(path)}
