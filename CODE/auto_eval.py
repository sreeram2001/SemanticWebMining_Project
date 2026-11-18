import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrieval as rt

CATEGORY_TITLES = {
    "INFORMATION-TECHNOLOGY": r"\b(software|developer|programmer|devops|cloud|it (support|specialist|manager|analyst|"
                              r"technician|director)|information technology|help ?desk|network (engineer|administrator|"
                              r"analyst)|systems? (administrator|engineer|analyst)|database|web developer|full[- ]stack|"
                              r"front[- ]end|back[- ]end|cyber ?security|data (engineer|scientist)|machine learning)\b",
    "ENGINEERING": r"\b(mechanical|electrical|civil|chemical|industrial|manufacturing|process|structural|aerospace|"
                   r"controls?|hardware|quality|design|project|field|maintenance|environmental) engineer(ing)?\b",
    "FINANCE": r"\b(financial analyst|finance (manager|director|analyst)|fp&a|controller|treasury|investment|"
               r"financial planning|financial advisor|cfo|chief financial)\b",
    "BANKING": r"\b(bank(er|ing)?|teller|loan|mortgage|underwriter|credit analyst|branch manager|relationship banker)\b",
    "BUSINESS-DEVELOPMENT": r"\b(business development|partnerships?|growth manager|bdr|bdm|new business)\b",
    "ACCOUNTANT": r"\b(accountant|accounting|bookkeeper|staff accountant|cpa|accounts (payable|receivable)|"
                  r"tax (accountant|preparer|associate|manager)|auditor|payroll)\b",
    "SALES": r"\b(sales|account executive|account manager|inside sales|retail sales|sales associate|"
             r"sales representative)\b",
    "CONSULTANT": r"\b(consultant|consulting|advisory|advisor)\b",
    "HR": r"\b(human resources|hr (generalist|manager|coordinator|specialist|business partner|director|assistant)|"
          r"recruit(er|ing|ment)|talent acquisition|people operations|benefits (specialist|administrator)|"
          r"compensation)\b",
    "DIGITAL-MEDIA": r"\b(digital (marketing|media|content)|social media|content (creator|strategist|manager|writer)|"
                     r"seo|video (editor|producer)|multimedia|media (planner|buyer|specialist))\b",
    "DESIGNER": r"\b(designer|graphic design|ux|ui|user experience|art director|interior design|creative director)\b",
    "HEALTHCARE": r"\b(nurse|rn|lpn|cna|medical|clinical|physician|therapist|pharmac(ist|y)|health ?care|"
                  r"patient|dental|caregiver|healthcare)\b",
}
METHODS = ("bm25", "dense", "hybrid", "rrf", "rerank")


def relevant_jobs(titles, patterns=CATEGORY_TITLES):
    compiled = {c: re.compile(p, re.I) for c, p in patterns.items()}
    out = {c: set() for c in patterns}
    for jid, title in titles.items():
        t = str(title or "")
        for c, rx in compiled.items():
            if rx.search(t):
                out[c].add(str(jid))
    return out


def qrels_for(resume_categories, relevant):
    return {rid: {j: 1 for j in relevant.get(cat, ())} for rid, cat in resume_categories.items()}


def method_runs(data, methods=METHODS):
    runs = {m: {str(r): [str(x[0]) for x in v] for r, v in data["runs"][m].items()}
            for m in methods if m in data.get("runs", {})}
    if "rerank" in methods and "rerank" in data:
        runs["rerank"] = {str(r): [str(x[0]) for x in v] for r, v in data["rerank"].items()}
    return runs


def evaluate_all(runs, qrels, k=10):
    return {m: rt.evaluate(run, qrels, k) for m, run in runs.items()}


def per_category(runs, qrels, resume_categories, k=10, metric="nDCG"):
    cats = sorted(set(resume_categories.values()))
    out = {}
    for c in cats:
        ids = {r for r, rc in resume_categories.items() if rc == c}
        out[c] = {m: rt.evaluate({r: v for r, v in run.items() if r in ids}, qrels, k)[f"{metric}@{k}"]
                  for m, run in runs.items()}
    return out


def markdown(table, cats, sizes, n_resumes, k=10):
    lines = ["# Automatic-label evaluation", "",
             f"{n_resumes} resumes. A job counts as relevant when its title matches the resume's category "
             f"(`CATEGORY_TITLES` in `auto_eval.py`). These are silver labels: they test whether a ranker "
             f"finds jobs in the right field, not how good the fit is.", "",
             f"| Method | P@{k} | nDCG@{k} |", "|---|---|---|"]
    for m, row in table.items():
        lines.append(f"| {m} | {row[f'P@{k}']:.3f} | {row[f'nDCG@{k}']:.3f} |")
    methods = list(table)
    lines += ["", f"## nDCG@{k} by resume category", "",
              "| Category | relevant jobs | " + " | ".join(methods) + " |",
              "|---|---|" + "---|" * len(methods)]
    for c, row in cats.items():
        lines.append(f"| {c} | {sizes.get(c, 0):,} | " + " | ".join(f"{row[m]:.3f}" for m in methods) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="Score the rankers against category-to-title labels.")
    p.add_argument("--runs", default="EVALUATIONS/runs/baselines.json")
    p.add_argument("--processed", default="DATA/processed")
    p.add_argument("--out", default="EVALUATIONS/runs")
    p.add_argument("--k", type=int, default=10)
    a = p.parse_args(argv)
    import pandas as pd
    with open(a.runs) as f:
        data = json.load(f)
    posts = pd.read_csv(os.path.join(a.processed, "postings_cleaned.csv"), usecols=["job_id", "title"])
    titles = dict(zip(posts["job_id"].astype(str), posts["title"]))
    res = pd.read_csv(os.path.join(a.processed, "resumes_cleaned.csv"), usecols=["ID", "Category"])
    runs = method_runs(data)
    cats = {str(r): c for r, c in zip(res["ID"], res["Category"]) if str(r) in runs["rrf"]}
    relevant = relevant_jobs(titles)
    qrels = qrels_for(cats, relevant)
    table = evaluate_all(runs, qrels, a.k)
    by_cat = per_category(runs, qrels, cats, a.k)
    sizes = {c: len(v) for c, v in relevant.items()}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "auto_eval.json"), "w") as f:
        json.dump({"results": table, "by_category": by_cat, "relevant_jobs": sizes,
                   "n_resumes": len(cats), "config": data.get("config")}, f, indent=1)
    md = markdown(table, by_cat, sizes, len(cats), a.k)
    with open(os.path.join(a.out, "auto_eval.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
