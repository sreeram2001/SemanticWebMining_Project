import argparse
import glob
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_score as gs
import rerank as rr
import pooling as pl
import retrieval as rt

BASELINES = ("bm25", "dense", "hybrid", "rrf", "rerank")
GRID = {
    "b2": [0.0, 0.25, 0.5, 1.0],
    "b3": [0.0, 0.25, 0.5],
    "lambdas": [(0.5, 0.25), (0.6, 0.3), (0.8, 0.4)],
    "w_required": [1.0, 2.0, 3.0],
    "top": [20, 30, 50],
}


def ids_of(ranking):
    return [str(item[0]) for item in ranking]


def shortlist(data, rid, top=None):
    rows = data.get("rerank", {}).get(rid, [])
    if top is None:
        return [(str(r[0]), r[2]) for r in rows]
    keep = set(ids_of(data["runs"]["rrf"].get(rid, []))[:top])
    rows = [r for r in rows if str(r[0]) in keep]
    return list(zip([str(r[0]) for r in rows], rr.minmax([r[1] for r in rows])))


def full_run(data, jobs, resumes, betas, lambdas=None, weights=None, top=None, ids=None):
    out = {}
    for rid in data.get("rerank", {}):
        if ids is not None and str(rid) not in ids:
            continue
        cands = shortlist(data, rid, top)
        scored = gs.rank_candidates(cands, jobs, resumes.get(str(rid)), betas, lambdas, weights)
        out[str(rid)] = [r["job_id"] for r in scored]
    return out


def setting_args(setting):
    w = setting.get("w_required")
    return {"lambdas": setting.get("lambdas"),
            "weights": {"required": w, "preferred": 1.0} if w else None,
            "top": setting.get("top")}


def method_runs(data, jobs, resumes, betas, lambdas=None, weights=None, top=None):
    runs = {m: {str(r): ids_of(v) for r, v in data["runs"][m].items()}
            for m in ("bm25", "dense", "hybrid", "rrf")}
    rerank = data.get("rerank", {})
    runs["rerank"] = {str(r): ids_of(v) for r, v in rerank.items()}
    runs["full"] = full_run(data, jobs, resumes, betas, lambdas, weights, top)
    runs["full_no_graph"] = full_run(data, jobs, resumes, gs.without_graph(betas), lambdas, weights, top)
    return runs


def subset(run, ids):
    keep = {str(i) for i in ids}
    return {r: v for r, v in run.items() if r in keep}


def tune(data, jobs, resumes, qrels, dev_ids, grid=None, k=10):
    grid = grid or GRID
    dev = {str(i) for i in dev_ids}
    tried, best = [], None
    combos = itertools.product(grid["b2"], grid["b3"], grid["lambdas"],
                               grid.get("w_required", [2.0]), grid.get("top", [None]))
    for b2, b3, (ls, lc), w, top in combos:
        setting = {"betas": {"b1": 1.0, "b2": b2, "b3": b3},
                   "lambdas": {"lam_sub": ls, "lam_cat": lc}, "w_required": w, "top": top}
        run = full_run(data, jobs, resumes, setting["betas"], ids=dev, **setting_args(setting))
        setting[f"nDCG@{k}"] = rt.evaluate(run, qrels, k)[f"nDCG@{k}"]
        tried.append(setting)
        score = setting[f"nDCG@{k}"]
        if best is None or score > best[f"nDCG@{k}"]:
            best = tried[-1]
    return best, tried


def pool_runs(data, jobs, resumes, ids, grid=None, k=10):
    grid = grid or GRID
    union = {}
    combos = itertools.product(grid["b2"], grid["b3"], grid["lambdas"],
                               grid.get("w_required", [2.0]), grid.get("top", [None]))
    for b2, b3, (ls, lc), w, top in combos:
        setting = {"betas": {"b1": 1.0, "b2": b2, "b3": b3},
                   "lambdas": {"lam_sub": ls, "lam_cat": lc}, "w_required": w, "top": top}
        run = full_run(data, jobs, resumes, setting["betas"], ids=ids, **setting_args(setting))
        for rid, ranked in run.items():
            union.setdefault(rid, [])
            union[rid].extend(j for j in ranked[:k] if j not in union[rid])
    return union


def fact_coverage(data, jobs, resumes, ids, top=None):
    per = {}
    for rid in ids:
        rid = str(rid)
        cands = [j for j, _ in shortlist(data, rid, top)]
        have = sum(1 for j in cands if j in jobs)
        per[rid] = {"resume": rid in resumes, "jobs": have, "shortlist": len(cands)}
    resumes_ok = sum(v["resume"] for v in per.values())
    jobs_ok = sum(v["jobs"] for v in per.values())
    total = sum(v["shortlist"] for v in per.values())
    return {"resumes": f"{resumes_ok}/{len(per)}", "jobs": f"{jobs_ok}/{total}",
            "rate": (jobs_ok / total if total else 0.0) * (resumes_ok / len(per) if per else 0.0),
            "per_resume": per}


def check_coverage(data, jobs, resumes, ids, tops, minimum, label):
    worst = None
    for top in tops:
        cov = fact_coverage(data, jobs, resumes, ids, top)
        print(f"{label}: facts for {cov['resumes']} resumes and {cov['jobs']} shortlist jobs "
              f"(shortlist {top or 'all'})")
        worst = cov["rate"] if worst is None else min(worst, cov["rate"])
    if worst < minimum:
        raise SystemExit(f"error: only {worst:.0%} of {label} pairs have facts (need {minimum:.0%}); "
                         "G and P would reward extracted jobs, not matching ones. "
                         "Extract the rest, lower the shortlist (--top), or pass --min-coverage")
    return worst


def compare(runs, qrels, ids, k=10):
    table = {m: rt.evaluate(subset(run, ids), qrels, k) for m, run in runs.items()}
    key = f"nDCG@{k}"
    table["delta_graph"] = {key: table["full"][key] - table["full_no_graph"][key]}
    return table


def markdown(table, title, k=10, setting=None):
    lines = [f"# {title}", ""]
    if setting:
        lines += [f"Setting: betas {setting['betas']}, lambdas {setting['lambdas']}, "
                  f"w_required {setting.get('w_required')}, shortlist {setting.get('top')}", ""]
    lines += [f"| Method | n | P@{k} | nDCG@{k} |", "|---|---|---|---|"]
    for m, row in table.items():
        if m != "delta_graph":
            lines.append(f"| {m} | {row['n']} | {row[f'P@{k}']:.3f} | {row[f'nDCG@{k}']:.3f} |")
    lines += ["", f"Graph contribution (full - full_no_graph): "
                  f"ΔnDCG@{k} = {table['delta_graph'][f'nDCG@{k}']:+.4f}"]
    return "\n".join(lines) + "\n"


def load_qrels(relevance_dir):
    sheets = [pl.read_grades(p) for p in sorted(glob.glob(os.path.join(relevance_dir, "grades_*.csv")))]
    return pl.merge_grades(sheets), pl.agreement(sheets)


def _load_linked(path):
    if not os.path.exists(path):
        print(f"warning: {path} not found; G and P will be 0 for every job")
        return {}
    return gs.load_linked(path)


def main(argv=None):
    p = argparse.ArgumentParser(description="Tune and evaluate graph-aware ranking.")
    p.add_argument("command", choices=["tune", "evaluate", "pool-runs"])
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--runs", default="EVALUATIONS/runs/baselines.json")
    p.add_argument("--resumes", default="DATA/linked/resumes.jsonl")
    p.add_argument("--postings", default="DATA/linked/postings.jsonl")
    p.add_argument("--relevance", default="EVALUATIONS/relevance")
    p.add_argument("--out", default="EVALUATIONS/runs")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--force", action="store_true", help="overwrite an existing test result")
    p.add_argument("--min-coverage", type=float, default=0.95,
                   help="share of resume-job pairs that must have extracted facts")
    p.add_argument("--top", type=int, nargs="+", help="shortlist sizes to tune (default 20 30 50)")
    a = p.parse_args(argv)

    with open(a.runs) as f:
        data = json.load(f)
    with open(os.path.join(a.relevance, "pool.json")) as f:
        chosen = json.load(f)["resumes"]
    if a.command == "pool-runs":
        resumes, jobs = _load_linked(a.resumes), _load_linked(a.postings)
        out = pool_runs(data, jobs, resumes, set(chosen["dev"] + chosen["test"]), k=a.k)
        path = os.path.join(a.out, "full_pool_run.json")
        os.makedirs(a.out, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f)
        print(f"wrote {path}; next: python CODE/pooling.py --extend {path}")
        return

    qrels, agree = load_qrels(a.relevance)
    if not qrels:
        raise SystemExit(f"error: no grades in {a.relevance}/grades_*.csv yet")
    resumes, jobs = _load_linked(a.resumes), _load_linked(a.postings)
    os.makedirs(a.out, exist_ok=True)
    tuned_path = os.path.join(a.out, "tuned_scoring.json")

    if a.command == "tune":
        grid = dict(GRID, top=a.top) if a.top else GRID
        check_coverage(data, jobs, resumes, chosen["dev"], grid["top"], a.min_coverage, "dev")
        best, tried = tune(data, jobs, resumes, qrels, chosen["dev"], grid, k=a.k)
        with open(tuned_path, "w") as f:
            json.dump({"best": best, "tried": tried, "dev_resumes": chosen["dev"]}, f, indent=1)
        print(f"best on dev: {best} -> {tuned_path}")
        return

    out_json = os.path.join(a.out, f"ranking_{a.split}.json")
    if a.split == "test" and os.path.exists(out_json) and not a.force:
        raise SystemExit(f"error: {out_json} exists; the test set is evaluated once (--force to redo)")
    if os.path.exists(tuned_path):
        with open(tuned_path) as f:
            setting = json.load(f)["best"]
    else:
        print("warning: no tuned_scoring.json; using default betas and lambdas")
        setting = {"betas": dict(gs.DEFAULT_BETAS), "lambdas": dict(gs.DEFAULT_LAMBDAS)}
    check_coverage(data, jobs, resumes, chosen[a.split], [setting.get("top")], a.min_coverage, a.split)
    runs = method_runs(data, jobs, resumes, setting["betas"], **setting_args(setting))
    table = compare(runs, qrels, chosen[a.split], a.k)
    with open(out_json, "w") as f:
        json.dump({"split": a.split, "setting": setting, "results": table,
                   "agreement_mean_kappa": agree["mean_kappa"],
                   "config": data.get("config")}, f, indent=1)
    md = markdown(table, f"Ranking results ({a.split})", a.k, setting)
    with open(out_json[:-5] + ".md", "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
