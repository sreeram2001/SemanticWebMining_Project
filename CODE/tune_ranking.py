import argparse
import glob
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_score as gs
import pooling as pl
import retrieval as rt

BASELINES = ("bm25", "dense", "hybrid", "rrf", "rerank")
GRID = {
    "b2": [0.0, 0.25, 0.5, 1.0],
    "b3": [0.0, 0.25, 0.5],
    "lambdas": [(0.5, 0.25), (0.6, 0.3), (0.8, 0.4)],
}


def ids_of(ranking):
    return [str(item[0]) for item in ranking]


def full_run(rerank, jobs, resumes, betas, lambdas=None):
    out = {}
    for rid, rows in rerank.items():
        cands = [(str(r[0]), r[2]) for r in rows]
        scored = gs.rank_candidates(cands, jobs, resumes.get(str(rid)), betas, lambdas)
        out[str(rid)] = [r["job_id"] for r in scored]
    return out


def method_runs(data, jobs, resumes, betas, lambdas=None):
    runs = {m: {str(r): ids_of(v) for r, v in data["runs"][m].items()}
            for m in ("bm25", "dense", "hybrid", "rrf")}
    rerank = data.get("rerank", {})
    runs["rerank"] = {str(r): ids_of(v) for r, v in rerank.items()}
    runs["full"] = full_run(rerank, jobs, resumes, betas, lambdas)
    runs["full_no_graph"] = full_run(rerank, jobs, resumes, gs.without_graph(betas), lambdas)
    return runs


def subset(run, ids):
    keep = {str(i) for i in ids}
    return {r: v for r, v in run.items() if r in keep}


def tune(data, jobs, resumes, qrels, dev_ids, grid=None, k=10):
    grid = grid or GRID
    rerank = {r: v for r, v in data.get("rerank", {}).items() if str(r) in set(dev_ids)}
    tried, best = [], None
    for b2, b3, (ls, lc) in itertools.product(grid["b2"], grid["b3"], grid["lambdas"]):
        betas = {"b1": 1.0, "b2": b2, "b3": b3}
        lambdas = {"lam_sub": ls, "lam_cat": lc}
        score = rt.evaluate(full_run(rerank, jobs, resumes, betas, lambdas), qrels, k)[f"nDCG@{k}"]
        tried.append({"betas": betas, "lambdas": lambdas, f"nDCG@{k}": score})
        if best is None or score > best[f"nDCG@{k}"]:
            best = tried[-1]
    return best, tried


def compare(runs, qrels, ids, k=10):
    table = {m: rt.evaluate(subset(run, ids), qrels, k) for m, run in runs.items()}
    key = f"nDCG@{k}"
    table["delta_graph"] = {key: table["full"][key] - table["full_no_graph"][key]}
    return table


def markdown(table, title, k=10, setting=None):
    lines = [f"# {title}", ""]
    if setting:
        lines += [f"Setting: betas {setting['betas']}, lambdas {setting['lambdas']}", ""]
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
    p.add_argument("command", choices=["tune", "evaluate"])
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--runs", default="DATA/runs/baselines.json")
    p.add_argument("--resumes", default="DATA/linked/resumes.jsonl")
    p.add_argument("--postings", default="DATA/linked/postings.jsonl")
    p.add_argument("--relevance", default="EVALUATIONS/relevance")
    p.add_argument("--out", default="EVALUATIONS/runs")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--force", action="store_true", help="overwrite an existing test result")
    a = p.parse_args(argv)

    with open(a.runs) as f:
        data = json.load(f)
    with open(os.path.join(a.relevance, "pool.json")) as f:
        chosen = json.load(f)["resumes"]
    qrels, agree = load_qrels(a.relevance)
    if not qrels:
        raise SystemExit(f"error: no grades in {a.relevance}/grades_*.csv yet")
    resumes, jobs = _load_linked(a.resumes), _load_linked(a.postings)
    os.makedirs(a.out, exist_ok=True)
    tuned_path = os.path.join(a.out, "tuned_scoring.json")

    if a.command == "tune":
        best, tried = tune(data, jobs, resumes, qrels, chosen["dev"], k=a.k)
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
    runs = method_runs(data, jobs, resumes, setting["betas"], setting["lambdas"])
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
