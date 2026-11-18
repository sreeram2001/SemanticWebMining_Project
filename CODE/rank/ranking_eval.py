"""Ranking evaluation: P@k, nDCG@k, bootstrap CIs, paired tests (Tier 1 auto, Tier 2 graded)."""

import csv
import itertools
import math
import os
import random
import re
import time
from collections import defaultdict

from core.config import write_json_atomic
from rank.ranking import BASELINES, METHODS

# Title patterns per resume category.
CATEGORY_TITLES = {
    "INFORMATION-TECHNOLOGY": r"\b(software|developer|programmer|data (engineer|scientist|analyst)|devops|cloud|"
                              r"it (support|specialist|manager|analyst|technician)|information technology|"
                              r"help ?desk|network|system(s)? (administrator|engineer|analyst)|machine learning|"
                              r"database|web|frontend|front-end|backend|back-end|full[- ]stack|cyber ?security)\b",
    "ENGINEERING": r"\b(mechanical|electrical|civil|chemical|industrial|manufacturing|process|structural|"
                   r"aerospace|controls?|hardware|quality|design) engineer(ing)?\b",
    "FINANCE": r"\b(financial analyst|finance|fp&a|controller|treasury|investment|financial planning)\b",
    "BANKING": r"\b(bank(er|ing)?|teller|loan|mortgage|underwriter|credit analyst|branch manager)\b",
    "BUSINESS-DEVELOPMENT": r"\b(business development|partnerships?|growth manager|bdr)\b",
    "ACCOUNTANT": r"\b(accountant|accounting|bookkeeper|auditor|tax|accounts (payable|receivable)|cpa)\b",
    "SALES": r"\b(sales|account executive|account manager|sdr|bdr)\b",
    "CONSULTANT": r"\b(consultant|consulting|advisory)\b",
    "HR": r"\b(human resources|hr|recruit(er|ing)|talent|payroll|benefits|people (operations|partner))\b",
    "DIGITAL-MEDIA": r"\b(digital marketing|social media|content|seo|media|video|multimedia|marketing)\b",
    "DESIGNER": r"\b(designer|ux|ui|graphic|creative director|illustrator)\b",
    "HEALTHCARE": r"\b(nurse|rn|lpn|physician|medical|clinical|health ?care|therapist|pharmac\w*|patient|"
                  r"caregiver|dental)\b",
}


# ------------------------------------------------------------------ labels

def auto_qrels(resumes, jobs):
    """Auto labels: {rid: {jid: 1}} where title fits category."""
    pats = {c: re.compile(p, re.I) for c, p in CATEGORY_TITLES.items()}
    out = {}
    for rid, cat in resumes.items():
        pat = pats.get(cat)
        if pat is None:
            continue
        rel = {j: 1 for j, t in jobs.items() if t and pat.search(t)}
        if rel:
            out[rid] = rel
    return out


def load_graded_qrels(path):
    out = defaultdict(dict)
    if not path or not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("grade", "")).strip() != "":
                out[str(r["resume_id"])][str(r["job_id"])] = int(r["grade"])
    return dict(out)


def pool_rows(ranker, resume_ids, methods=("bm25", "dense", "rrf", "full"), depth=10):
    """Shuffled pool of each system's top `depth` jobs."""
    rows = []
    for rid in resume_ids:
        pool = []
        for m in methods:
            for r in ranker.rank(rid, m)[:depth]:
                if r["job_id"] not in pool:
                    pool.append(r["job_id"])
        random.Random(f"{rid}").shuffle(pool)
        for j in pool:
            meta = ranker.g.meta.get(j, {})
            rows.append({"resume_id": rid, "category": ranker.g.meta.get(rid, {}).get("category"),
                         "job_id": j, "title": meta.get("title"), "company": meta.get("company_name"),
                         "grade": "", "annotator": ""})
    return rows


# ------------------------------------------------------------------ metrics

def precision_at_k(ranked, grades, k):
    return sum(1 for j in ranked[:k] if grades.get(j, 0) >= 1) / k


def ndcg_at_k(ranked, grades, k):
    dcg = sum((2 ** grades.get(j, 0) - 1) / math.log2(i + 2) for i, j in enumerate(ranked[:k]))
    ideal = sorted((2 ** g - 1 for g in grades.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def bootstrap_ci(values, n=1000, seed=573, alpha=0.05):
    """Percentile bootstrap CI of the mean, resampling resumes."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
    return means[int(alpha / 2 * n)], means[min(n - 1, int((1 - alpha / 2) * n))]


def randomization_test(a, b, n=10000, seed=573):
    """Paired sign-flip test p-value."""
    diffs = [x - y for x, y in zip(a, b)]
    if not diffs:
        return None
    observed = abs(sum(diffs))
    if observed == 0:
        return 1.0
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n + 1)


# ------------------------------------------------------------------ evaluation

def evaluate_ranking(ranker, qrels, resume_ids, methods=METHODS, k=10, scoring=None,
                     n_boot=1000, n_perm=10000, seed=573):
    rids = [r for r in resume_ids if r in qrels and any(g > 0 for g in qrels[r].values())]
    per = {m: {"P": [], "nDCG": []} for m in methods}
    for rid in rids:
        for m in methods:
            ranked = [x["job_id"] for x in ranker.rank(rid, m, scoring)]
            per[m]["P"].append(precision_at_k(ranked, qrels[rid], k))
            per[m]["nDCG"].append(ndcg_at_k(ranked, qrels[rid], k))
    summary = {}
    for m in methods:
        summary[m] = {}
        for metric, vals in per[m].items():
            mean = sum(vals) / len(vals) if vals else None
            summary[m][f"{metric}@{k}"] = mean
            summary[m][f"{metric}@{k}_ci95"] = bootstrap_ci(vals, n_boot, seed)
    comparisons = []
    pairs = [("full", b) for b in BASELINES if b in methods and "full" in methods]
    if "full" in methods and "full_no_graph" in methods:
        pairs.append(("full", "full_no_graph"))
    for a, b in pairs:
        for metric in ("nDCG", "P"):
            diffs = [x - y for x, y in zip(per[a][metric], per[b][metric])]
            comparisons.append({
                "system": a, "baseline": b, "metric": f"{metric}@{k}",
                "delta": sum(diffs) / len(diffs) if diffs else None,
                "delta_ci95": bootstrap_ci(diffs, n_boot, seed),
                "p_value": randomization_test(per[a][metric], per[b][metric], n_perm, seed),
                "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs)})
    return {"n_resumes": len(rids), "resume_ids": rids, "summary": summary,
            "comparisons": comparisons, "per_resume": per}


def tune(ranker, qrels, dev_ids, grid, k=10):
    """Dev grid search -> (best_scoring, best_nDCG, n_tried)."""
    keys = list(grid)
    best, best_sc, tried = -1.0, None, 0
    for values in itertools.product(*grid.values()):
        sc = dict(zip(keys, values))
        if not 0 < sc.get("lambda_cat", 0.3) < sc.get("lambda_sub", 0.6) < 1:
            continue
        tried += 1
        res = evaluate_ranking(ranker, qrels, dev_ids, ["full"], k, sc, n_boot=10, n_perm=10)
        v = res["summary"]["full"][f"nDCG@{k}"] or 0.0
        if v > best:
            best, best_sc = v, sc
    return best_sc, best, tried


DEFAULT_GRID = {
    "beta_ce": [0.4, 0.6, 0.8],
    "beta_graph": [0.2, 0.4, 0.6],
    "beta_penalty": [0.0, 0.3, 0.6],
    "lambda_sub": [0.5, 0.7],
    "lambda_cat": [0.2, 0.3],
}


# ------------------------------------------------------------------ reporting

def _fmt(x, ci=None):
    if x is None:
        return "n/a"
    s = f"{x:.3f}"
    return s + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def report_markdown(result, title, k=10):
    lines = [f"### {title} ({result['n_resumes']} resumes)", "",
             f"| Method | P@{k} [95% CI] | nDCG@{k} [95% CI] |", "|---|---|---|"]
    for m, s in result["summary"].items():
        lines.append(f"| {m} | {_fmt(s[f'P@{k}'], s[f'P@{k}_ci95'])} | {_fmt(s[f'nDCG@{k}'], s[f'nDCG@{k}_ci95'])} |")
    if result["comparisons"]:
        lines += ["", "| Comparison | Metric | Δ [95% CI] | p (randomization) | wins/losses |", "|---|---|---|---|---|"]
        for c in result["comparisons"]:
            lines.append(f"| {c['system']} vs {c['baseline']} | {c['metric']} | {_fmt(c['delta'], c['delta_ci95'])} | "
                         f"{c['p_value']:.4f} | {c['wins']}/{c['losses']} |" if c["p_value"] is not None else
                         f"| {c['system']} vs {c['baseline']} | {c['metric']} | n/a | n/a | n/a |")
    return "\n".join(lines)


def run_log(cfg, backends, data_hash, split_name, extra=None):
    return {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "split": split_name, "data_hash": data_hash,
            "seed": cfg["evaluation"]["seed"], "extraction_model": cfg["extraction"].get("model")
            if cfg["extraction"].get("mode") == "llm" else "rules", "backends": backends,
            "config": {k: cfg[k] for k in ("retrieval", "rerank", "scoring", "linking")}, **(extra or {})}


def save_json(obj, path):
    write_json_atomic(path, obj)
