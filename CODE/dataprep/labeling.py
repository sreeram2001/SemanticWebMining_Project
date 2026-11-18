"""Team labeling sheets and merging. Shared items are majority-voted; agreement via Cohen's kappa."""

import csv
import glob
import os
import random
from collections import Counter, defaultdict
from itertools import combinations

from core.config import write_json_atomic


# ------------------------------------------------------------------ assignment

def choose_resumes(split, categories, n=30, seed=573):
    """n resumes, half dev/half test, round-robin over categories."""
    rng = random.Random(seed)
    out = []
    for part, k in (("dev", n // 2), ("test", n - n // 2)):
        by_cat = defaultdict(list)
        for rid, s in sorted(split.items()):
            if s == part and rid in categories:
                by_cat[categories[rid]].append(rid)
        for ids in by_cat.values():
            rng.shuffle(ids)
        picked, cats = [], sorted(by_cat)
        while len(picked) < k and any(by_cat.values()):
            for c in cats:
                if by_cat[c] and len(picked) < k:
                    picked.append(by_cat[c].pop())
        out += picked
    return out


def assign(units, annotators, shared, seed=573):
    """{annotator: [units]}: `shared` to all, the rest split evenly."""
    units = list(dict.fromkeys(units))
    random.Random(seed).shuffle(units)
    common, rest = units[:shared], units[shared:]
    out = {a: list(common) for a in annotators}
    for i, u in enumerate(rest):
        out[annotators[i % len(annotators)]].append(u)
    return out, common


def write_sheets(rows, unit_key, annotators, shared, out_dir, prefix, seed=573):
    """One CSV per annotator; existing sheets are kept."""
    os.makedirs(out_dir, exist_ok=True)
    units = [r[unit_key] for r in rows]
    plan, common = assign(units, annotators, shared, seed)
    by_unit = defaultdict(list)
    for r in rows:
        by_unit[r[unit_key]].append(r)
    written = []
    for a, mine in plan.items():
        path = os.path.join(out_dir, f"{prefix}_{a}.csv")
        if os.path.exists(path):
            continue
        sheet = [{**r, "annotator": a, "shared": int(u in common)} for u in mine for r in by_unit[u]]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sheet[0]) if sheet else [unit_key])
            w.writeheader()
            w.writerows(sheet)
        written.append((path, len(sheet)))
    write_json_atomic(os.path.join(out_dir, f"{prefix}_assignment.json"), {"shared": common, "plan": plan})
    return written, common


# ------------------------------------------------------------------ agreement

def cohen_kappa(a, b, weights=None):
    """Cohen's kappa for two label lists; weights='linear' for ordinal grades."""
    if not a:
        return None
    cats = sorted(set(a) | set(b), key=str)
    if len(cats) == 1:
        return 1.0
    idx = {c: i for i, c in enumerate(cats)}
    n, k = len(a), len(cats)

    def w(i, j):
        if weights == "linear":
            return abs(i - j) / (k - 1)
        return 0.0 if i == j else 1.0

    obs = sum(w(idx[x], idx[y]) for x, y in zip(a, b)) / n
    pa, pb = Counter(a), Counter(b)
    exp = sum(pa[x] * pb[y] * w(idx[x], idx[y]) for x in cats for y in cats) / (n * n)
    return 1.0 - obs / exp if exp else 1.0


def merge(sheet_dir, prefix, item_keys, label_key, out_path, annotators, ordinal=False):
    """Merge sheets into one gold CSV and report agreement."""
    labels = defaultdict(dict)                 # item -> {annotator: label}
    rows = {}
    for path in sorted(glob.glob(os.path.join(sheet_dir, f"{prefix}_*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lab = (r.get(label_key) or "").strip()
                if not lab:
                    continue
                item = tuple(r[k] for k in item_keys)
                labels[item][r["annotator"]] = int(lab) if ordinal else lab
                rows.setdefault(item, r)
    order = {a: i for i, a in enumerate(annotators)}
    merged = []
    for item, by in labels.items():
        counts = Counter(by.values())
        top = max(counts.values())
        winners = [lab for lab, c in counts.items() if c == top]
        first = min(by, key=lambda a: order.get(a, 99))
        final = by[first] if by[first] in winners else winners[0]
        r = {k: v for k, v in rows[item].items() if k not in ("annotator", "shared", label_key)}
        merged.append({**r, label_key: final, "annotator": ";".join(sorted(by)), "n_labels": len(by)})
    if merged:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(merged[0]))
            w.writeheader()
            w.writerows(merged)
    multi = {i: by for i, by in labels.items() if len(by) > 1}
    kappas, agree = [], []
    for a, b in combinations(sorted({x for by in multi.values() for x in by}), 2):
        both = [(by[a], by[b]) for by in multi.values() if a in by and b in by]
        if both:
            kappas.append(cohen_kappa([x for x, _ in both], [y for _, y in both], "linear" if ordinal else None))
            agree.append(sum(x == y for x, y in both) / len(both))
    return {"items": len(labels), "merged_rows": len(merged), "shared_items": len(multi),
            "mean_pairwise_kappa": round(sum(kappas) / len(kappas), 4) if kappas else None,
            "mean_pairwise_agreement": round(sum(agree) / len(agree), 4) if agree else None,
            "kappa_type": "linear-weighted Cohen" if ordinal else "Cohen", "out": out_path}


# ------------------------------------------------------------------ extraction gold set

def write_extraction_templates(postings_csv, resumes_csv, gold_dir, second_dir, annotators,
                               n=60, shared=10, seed=573):
    """Gold templates (as in notebook Step 3c) plus annotator assignment."""
    import pandas as pd
    from extract.evaluation import write_templates
    df_res, df_posts = pd.read_csv(resumes_csv), pd.read_csv(postings_csv)
    per_category = n // df_res["Category"].nunique()
    gold_res = df_res.groupby("Category").sample(n=per_category, random_state=seed)
    gold_posts = df_posts.sample(n=n, random_state=seed)
    out = {"resumes": write_templates("resume", zip(gold_res["ID"], gold_res["Resume_clean"], gold_res["Category"]),
                                      gold_dir),
           "postings": write_templates("posting", zip(gold_posts["job_id"], gold_posts["description"],
                                                      [None] * len(gold_posts)), gold_dir)}
    shared_res = gold_res.groupby("Category").head(1).head(shared)
    shared_posts = gold_posts.head(shared)
    write_templates("resume", zip(shared_res["ID"], shared_res["Resume_clean"], shared_res["Category"]), second_dir)
    write_templates("posting", zip(shared_posts["job_id"], shared_posts["description"], [None] * len(shared_posts)),
                    second_dir)
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(gold_dir, "*.json")))
    second = sorted(os.path.basename(p) for p in glob.glob(os.path.join(second_dir, "*.json")))
    plan = {a: files[i::len(annotators)] for i, a in enumerate(annotators)}
    # second annotator differs from the first
    owner = {f: a for a, fs in plan.items() for f in fs}
    second_plan = defaultdict(list)
    for i, f in enumerate(second):
        a = annotators[(annotators.index(owner.get(f, annotators[0])) + 1 + i % (len(annotators) - 1))
                       % len(annotators)]
        second_plan[a].append(f)
    write_json_atomic(os.path.join(gold_dir, "assignment.json"), {"gold": plan, "gold_second": dict(second_plan)})
    out["shared_copies"] = len(second)
    return out
