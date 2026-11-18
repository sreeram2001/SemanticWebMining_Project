"""Relevance pooling and grading sheets (Step 6b, owner: Sreeram).

Builds the graded relevance judgments (0 = not relevant, 1 = partly, 2 = relevant)
for the 30 test resumes:

  1. build_pool       take the top `depth` jobs from every method's run, merge
                      them, and shuffle so annotators cannot tell which method
                      ranked a job first
  2. write_sheets     write a jobs sheet and a resumes file for the annotators
  3. read_grades      read each annotator's filled-in sheet
  4. merge_grades     one grade per (resume, job) for the evaluation harness
  5. weighted_kappa   inter-annotator agreement on the shared items

The merged grades have the shape {resume id: {job id: grade}} that
retrieval.evaluate() expects.
"""

import csv
import os
import random
from collections import defaultdict
from itertools import combinations

GRADES = (0, 1, 2)
DESC_CHARS = 600


# ------------------------------------------------------------------- pooling
def build_pool(runs, resume_ids, depth=10, seed=573, methods=None):
    """Pool the top `depth` jobs of each method for each resume.

    `runs`: {method: {resume id: [(job id, score), ...]}}. Returns
    {resume id: [{"job_id", "methods": [...]}, ...]} in a seeded shuffled
    order, where "methods" lists every method that put the job in its top
    `depth`. The methods are kept for analysis and left out of the sheets.
    """
    names = list(methods or runs)
    pool = {}
    for rid in resume_ids:
        found = defaultdict(list)
        for m in names:
            for item in runs[m].get(rid, [])[:depth]:
                found[item[0]].append(m)
        rows = [{"job_id": j, "methods": ms} for j, ms in found.items()]
        random.Random(f"{seed}:{rid}").shuffle(rows)
        pool[rid] = rows
    return pool


def pool_stats(pool):
    sizes = [len(v) for v in pool.values()]
    return {
        "resumes": len(sizes),
        "pairs": sum(sizes),
        "mean_per_resume": (sum(sizes) / len(sizes)) if sizes else 0.0,
    }


# -------------------------------------------------------------------- sheets
def select_relevance_resumes(split, categories, n_dev=15, n_test=15, seed=573):
    """Pick the graded resumes from the frozen gold split, stratified by category.

    `split`: the "docs" map of EVALUATIONS/gold_split.json ("resume:<id>" ->
    "dev" or "test"). `categories`: {resume id: category}. Within each split,
    resumes are taken round-robin over categories in a seeded order, so every
    category is covered before any repeats. Returns {"dev": [...], "test": [...]}.
    """
    out = {}
    for name, n in (("dev", n_dev), ("test", n_test)):
        by_cat = defaultdict(list)
        for key, part in sorted(split.items()):
            if part == name and key.startswith("resume:"):
                rid = key.split(":", 1)[1]
                by_cat[categories.get(rid, "")].append(rid)
        rng = random.Random(f"{seed}:{name}")
        for ids in by_cat.values():
            rng.shuffle(ids)
        cats = sorted(by_cat)
        rng.shuffle(cats)
        chosen, i = [], 0
        while len(chosen) < n and any(by_cat.values()):
            ids = by_cat[cats[i % len(cats)]]
            if ids:
                chosen.append(ids.pop())
            i += 1
        out[name] = chosen
    return out


def assign_resumes(resume_ids, annotators, shared=3):
    """{annotator: [resume ids]}. The first `shared` resumes go to everyone
    (for agreement); the rest are dealt out round-robin, one owner each."""
    ids = list(resume_ids)
    common, rest = ids[:shared], ids[shared:]
    out = {a: list(common) for a in annotators}
    for i, rid in enumerate(rest):
        out[annotators[i % len(annotators)]].append(rid)
    return out


def write_sheets(
    pool,
    resume_texts,
    job_info,
    out_dir,
    annotators,
    resume_categories=None,
    assignment=None,
):
    """Write the files annotators fill in.

    One sheet per annotator: resume_id, job_id, title, company, description
    (excerpt), grade (blank). resumes.csv has the full resume text. With an
    `assignment` ({annotator: [resume ids]}) each sheet only holds that
    person's resumes; otherwise everyone gets every pair. Returns the paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    resumes_path = os.path.join(out_dir, "resumes.csv")
    with open(resumes_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["resume_id", "category", "resume_text"])
        for rid in pool:
            w.writerow([rid, (resume_categories or {}).get(rid, ""), resume_texts[rid]])
    paths.append(resumes_path)
    for name in annotators:
        path = os.path.join(out_dir, f"grades_{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                ["resume_id", "job_id", "title", "company", "description", "grade"]
            )
            mine = set(assignment[name]) if assignment else set(pool)
            for rid, rows in pool.items():
                if rid not in mine:
                    continue
                for row in rows:
                    info = job_info.get(row["job_id"], {})
                    w.writerow(
                        [
                            rid,
                            row["job_id"],
                            info.get("title", ""),
                            info.get("company_name", ""),
                            str(info.get("description", ""))[:DESC_CHARS],
                            "",
                        ]
                    )
        paths.append(path)
    return paths


def read_grades(path):
    """Read one annotator's sheet into {(resume id, job id): grade}.

    Blank grades are skipped. A grade outside 0, 1, 2 raises ValueError with
    the row, so typos are caught before they reach the metrics.
    """
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            raw = (row.get("grade") or "").strip()
            if not raw:
                continue
            try:
                grade = int(raw)
            except ValueError:
                raise ValueError(f"{path} line {n}: grade {raw!r} is not 0, 1 or 2")
            if grade not in GRADES:
                raise ValueError(f"{path} line {n}: grade {grade} is not 0, 1 or 2")
            out[(str(row["resume_id"]), str(row["job_id"]))] = grade
    return out


def merge_grades(sheets):
    """Combine annotators into {resume id: {job id: grade}}.

    `sheets`: list of {(resume id, job id): grade}. The final grade is the mean
    of the available grades, rounded half up, so one annotator's grade stands
    when only one graded the pair.
    """
    collected = defaultdict(list)
    for sheet in sheets:
        for key, g in sheet.items():
            collected[key].append(g)
    qrels = defaultdict(dict)
    for (rid, jid), gs in collected.items():
        qrels[rid][jid] = int(sum(gs) / len(gs) + 0.5)
    return dict(qrels)


# ----------------------------------------------------------------- agreement
def weighted_kappa(a, b, categories=len(GRADES)):
    """Quadratic-weighted Cohen's kappa for two annotators' ordinal grades.

    1.0 is perfect agreement, 0 is chance level. If both annotators gave one
    identical grade throughout, chance agreement is total and the result is
    1.0 when they agree and 0.0 otherwise.
    """
    if len(a) != len(b):
        raise ValueError("the two grade lists must have the same length")
    n = len(a)
    if n == 0:
        raise ValueError("no shared items to compare")
    k = categories
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            weight = ((i - j) / (k - 1)) ** 2
            num += weight * obs[i][j]
            den += weight * row[i] * col[j] / n
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return 1 - num / den


def agreement(sheets):
    """Mean pairwise weighted kappa over annotators, with the item counts.

    Only pairs graded by both annotators are compared. Returns
    {"pairs": {(i, j): (kappa, n)}, "mean_kappa": float or None}.
    """
    pairs = {}
    for i, j in combinations(range(len(sheets)), 2):
        shared = sorted(set(sheets[i]) & set(sheets[j]))
        if shared:
            pairs[(i, j)] = (
                weighted_kappa(
                    [sheets[i][k] for k in shared], [sheets[j][k] for k in shared]
                ),
                len(shared),
            )
    mean = sum(k for k, _ in pairs.values()) / len(pairs) if pairs else None
    return {"pairs": pairs, "mean_kappa": mean}


# ----------------------------------------------------------------------- CLI
ANNOTATORS = ["aakash", "chandana", "chirag", "kaushal", "richitha", "sreeram"]
POOL_METHODS = ("bm25", "dense", "hybrid", "rrf", "rerank")


def main(argv=None):
    import argparse
    import json

    import pandas as pd

    p = argparse.ArgumentParser(description="Build the relevance pool and grading sheets.")
    p.add_argument("--runs", default="DATA/runs/baselines.json")
    p.add_argument("--split", default="EVALUATIONS/gold_split.json")
    p.add_argument("--processed", default="DATA/processed")
    p.add_argument("--out", default="EVALUATIONS/relevance")
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--shared", type=int, default=3, help="resumes everyone grades")
    p.add_argument("--seed", type=int, default=573)
    a = p.parse_args(argv)

    with open(a.runs) as f:
        data = json.load(f)
    runs = dict(data["runs"])
    if "rerank" in data:
        runs["rerank"] = data["rerank"]
    with open(a.split) as f:
        split = json.load(f)["docs"]
    res = pd.read_csv(os.path.join(a.processed, "resumes_cleaned.csv"),
                      usecols=["ID", "Category", "Resume_clean"])
    res["ID"] = res["ID"].astype(str)
    cats = dict(zip(res["ID"], res["Category"]))
    texts = dict(zip(res["ID"], res["Resume_clean"]))

    chosen = select_relevance_resumes(split, cats, seed=a.seed)
    ids = chosen["dev"] + chosen["test"]
    missing = [r for r in ids if r not in runs["rrf"]]
    if missing:
        raise SystemExit(f"error: no runs for resumes {missing}; rerun run_baselines.py")
    methods = [m for m in POOL_METHODS if m in runs]
    pool = build_pool(runs, ids, depth=a.depth, seed=a.seed, methods=methods)

    job_ids = {row["job_id"] for rows in pool.values() for row in rows}
    posts = pd.read_csv(os.path.join(a.processed, "postings_cleaned.csv"),
                        usecols=["job_id", "title", "company_name", "description"])
    posts["job_id"] = posts["job_id"].astype(str)
    info = posts[posts["job_id"].isin(job_ids)].set_index("job_id").to_dict("index")

    order = chosen["dev"][:a.shared] + [r for r in ids if r not in chosen["dev"][:a.shared]]
    assignment = assign_resumes(order, ANNOTATORS, a.shared)
    write_sheets({r: pool[r] for r in order}, texts, info, a.out, ANNOTATORS, cats, assignment)
    meta = {"seed": a.seed, "depth": a.depth, "methods": methods, "resumes": chosen,
            "assignment": assignment, "stats": pool_stats(pool),
            "pool": {r: pool[r] for r in ids}}
    with open(os.path.join(a.out, "pool.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{meta['stats']['pairs']} pairs for {len(ids)} resumes "
          f"({meta['stats']['mean_per_resume']:.1f} per resume) -> {a.out}")


if __name__ == "__main__":
    main()
