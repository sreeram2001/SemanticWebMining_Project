import skill_linking as sl

DEGREE_RANK = {"high_school": 0, "associate": 1, "bachelors": 2, "masters": 3, "phd": 4}

DEFAULT_BETAS = {"b1": 1.0, "b2": 0.5, "b3": 0.5}
DEFAULT_LAMBDAS = {"lam_sub": sl.LAMBDA_SUB, "lam_cat": sl.LAMBDA_CAT}


def _facts(record):
    if not record or record.get("status", "ok") != "ok":
        return {}
    return record.get("facts") or {}


def graph_score(
    job_record,
    resume_record,
    lam_sub=sl.LAMBDA_SUB,
    lam_cat=sl.LAMBDA_CAT,
    weights=None,
):
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


def resume_years(resume_facts):
    exp = resume_facts.get("experience") or []
    dated = [e["duration_months"] for e in exp if e.get("duration_months") is not None]
    if not dated:
        return None, False
    return sum(dated) / 12.0, len(dated) == len(exp)


def experience_status(min_years, years, complete):
    if years is not None and years >= min_years:
        return "met"
    if years is not None and complete:
        return "mismatch"
    return "unknown"


def penalty(job_record, resume_record):
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
        years, complete = resume_years(res)
        status = experience_status(exp["min_years"], years, complete)
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


def final_score(ce, g, p, betas=None):
    b = {**DEFAULT_BETAS, **(betas or {})}
    return b["b1"] * ce + b["b2"] * g - b["b3"] * p


def without_graph(betas=None):
    return {**DEFAULT_BETAS, **(betas or {}), "b2": 0.0}


def rank_candidates(candidates, job_records, resume_record, betas=None, lambdas=None, weights=None):
    lam = {**DEFAULT_LAMBDAS, **(lambdas or {})}
    out = []
    for job_id, ce in candidates:
        rec = job_records.get(str(job_id))
        g = graph_score(rec, resume_record, weights=weights, **lam)
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
    return sorted(out, key=lambda r: -r["score"])


def load_linked(path):
    return {str(r["doc_id"]): r for r in sl.latest_records(path)}
