import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_score as gs
import tune_ranking as tr

CAUSES = ("extraction", "linking", "retrieval", "graph scoring", "penalty", "ranking")


def job_detail(rec, resume, lambdas, weights):
    g = gs.graph_score(rec, resume, weights=weights, **lambdas)
    p = gs.penalty(rec, resume)
    return {
        "g": round(g["score"], 3),
        "p": round(p["score"], 3),
        "linked": len(g["rows"]),
        "unresolved": len(g["unresolved"]),
        "conflicts": [d["type"] for d in p["details"] if d["status"] == "mismatch"],
        "evidence": [
            {"skill": r["lightcast"], "credit": r["credit"], "level": r["level"]}
            for r in g["rows"]
            if r["credit"] > 0
        ],
    }


def cause(detail, has_job, has_resume, demoted):
    if not has_job or not has_resume:
        return "extraction"
    if detail["unresolved"] > detail["linked"]:
        return "linking"
    if demoted and detail["p"] > 0:
        return "penalty"
    return "graph scoring"


def analyze(data, jobs, resumes, qrels, ids, setting, k=10):
    args = tr.setting_args(setting)
    lambdas = {**gs.DEFAULT_LAMBDAS, **(args["lambdas"] or {})}
    full = tr.full_run(data, jobs, resumes, setting["betas"], ids=set(ids), **args)
    base = tr.full_run(
        data, jobs, resumes, gs.without_graph(setting["betas"]), ids=set(ids), **args
    )
    cases = []
    for rid in ids:
        rid = str(rid)
        grades = qrels.get(rid, {})
        res = resumes.get(rid)
        with_g, without_g = full.get(rid, []), base.get(rid, [])
        top_with, top_without = set(with_g[:k]), set(without_g[:k])
        shortlist = set(with_g)
        for jid in sorted(top_with ^ top_without):
            grade = grades.get(jid)
            promoted = jid in top_with
            if grade is None:
                kind = "unjudged"
            elif promoted:
                kind = "graph helped" if grade >= 1 else "graph hurt"
            else:
                kind = "graph hurt" if grade >= 1 else "graph helped"
            d = job_detail(jobs.get(jid), res, lambdas, args["weights"])
            cases.append(
                {
                    "resume": rid,
                    "job": jid,
                    "grade": grade,
                    "move": "into top" if promoted else "out of top",
                    "rank_with": with_g.index(jid) + 1,
                    "rank_without": without_g.index(jid) + 1,
                    "kind": kind,
                    **d,
                    "cause": None
                    if kind != "graph hurt"
                    else cause(d, jid in jobs, res is not None, demoted=not promoted),
                }
            )
        for jid, grade in grades.items():
            if grade < 1 or jid in top_with:
                continue
            kind = "retrieval" if jid not in shortlist else "ranking"
            cases.append(
                {
                    "resume": rid,
                    "job": jid,
                    "grade": grade,
                    "move": "missed",
                    "rank_with": with_g.index(jid) + 1 if jid in shortlist else None,
                    "rank_without": None,
                    "kind": "missed relevant",
                    "cause": kind,
                }
            )
    return cases


def summarize(cases):
    return {
        "by_kind": dict(Counter(c["kind"] for c in cases)),
        "graph_hurt_by_cause": dict(
            Counter(c["cause"] for c in cases if c["kind"] == "graph hurt")
        ),
        "missed_relevant_by_cause": dict(
            Counter(c["cause"] for c in cases if c["kind"] == "missed relevant")
        ),
    }


def markdown(summary, split, k=10):
    lines = [
        f"# Error analysis ({split})",
        "",
        f"Jobs that entered or left the top {k} when the graph term was added, and relevant "
        f"jobs (grade 1 or 2) that the full pipeline left out of its top {k}.",
        "",
        "| Kind | Count |",
        "|---|---|",
    ]
    lines += [f"| {k_} | {v} |" for k_, v in sorted(summary["by_kind"].items())]
    for title, key in (
        ("Graph hurt, by cause", "graph_hurt_by_cause"),
        ("Missed relevant jobs, by cause", "missed_relevant_by_cause"),
    ):
        lines += ["", f"## {title}", "", "| Cause | Count |", "|---|---|"]
        lines += [f"| {c} | {n} |" for c, n in sorted(summary[key].items())] or [
            "| none | 0 |"
        ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="Sort ranking errors by pipeline stage.")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--runs", default="EVALUATIONS/runs/baselines.json")
    p.add_argument("--resumes", default="DATA/linked/resumes.jsonl")
    p.add_argument("--postings", default="DATA/linked/postings.jsonl")
    p.add_argument("--relevance", default="EVALUATIONS/relevance")
    p.add_argument("--out", default="EVALUATIONS/runs")
    p.add_argument("--k", type=int, default=10)
    a = p.parse_args(argv)
    with open(a.runs) as f:
        data = json.load(f)
    with open(os.path.join(a.relevance, "pool.json")) as f:
        chosen = json.load(f)["resumes"]
    qrels, _ = tr.load_qrels(a.relevance)
    tuned = os.path.join(a.out, "tuned_scoring.json")
    if os.path.exists(tuned):
        with open(tuned) as f:
            setting = json.load(f)["best"]
    else:
        print("warning: no tuned_scoring.json; using default weights")
        setting = {"betas": dict(gs.DEFAULT_BETAS), "lambdas": dict(gs.DEFAULT_LAMBDAS)}
    jobs, resumes = tr._load_linked(a.postings), tr._load_linked(a.resumes)
    cases = analyze(data, jobs, resumes, qrels, chosen[a.split], setting, a.k)
    summary = summarize(cases)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"error_analysis_{a.split}")
    with open(path + ".json", "w") as f:
        json.dump(
            {"split": a.split, "setting": setting, "summary": summary, "cases": cases},
            f,
            indent=1,
        )
    md = markdown(summary, a.split, a.k)
    with open(path + ".md", "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
