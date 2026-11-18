"""Skill-linking gold set and scoring (proposal: 300 mentions, Accuracy@1 and NIL accuracy).

1. sample_mentions picks mentions from the extraction output: half from
   resumes, half from postings, one occurrence per distinct mention, fixed seed.
2. write_gold_template writes them to a CSV sheet with a shuffled list of
   candidate skills; annotators fill in gold_id with a Lightcast ID or NIL.
   Each mention is assigned to dev or test by its document, so all mentions
   of a document fall on the same side.
3. load_gold reads the labeled rows and checks every ID against the taxonomy.
4. evaluate scores one linker on the labeled rows; compare scores several
   linker configurations (the alias-only baseline first) with bootstrap CIs
   and a paired randomization test against the baseline.
"""

import csv
import hashlib
import os
import random

import skill_linking as sl

CANDIDATES_SHOWN = 6
COLUMNS = (["mention_id", "split", "kind", "doc_id", "mention", "context"]
           + [f"candidate_{i}" for i in range(1, CANDIDATES_SHOWN + 1)]
           + ["gold_id", "annotator", "notes"])
NIL_WORDS = {"NIL", "NONE", "N/A"}


# ---------------------------------------------------------------- sampling

def mention_pool(records):
    """(kind, doc_id, mention, context) for every skill fact in extraction records."""
    for rec in records:
        if rec.get("status") != "ok":
            continue
        field = sl.SKILL_FIELDS.get(rec.get("kind"))
        for fact in (rec.get("facts") or {}).get(field) or []:
            name = fact.get("name")
            if sl.normalize_name(name):
                yield rec["kind"], str(rec["doc_id"]), name, fact.get("source_span") or name


def sample_mentions(records, n=300, seed=573, resume_share=0.5):
    """Up to n mentions, split between resumes and postings, one per distinct mention.

    If one kind has too few distinct mentions, the other kind makes up the rest.
    """
    pools = {"resume": [], "posting": []}
    for item in mention_pool(records):
        pools[item[0]].append(item)
    rng = random.Random(seed)
    unique = {}
    for kind in ("resume", "posting"):
        items = list(pools[kind])
        rng.shuffle(items)
        seen, keep = set(), []
        for item in items:
            key = sl.normalize_name(item[2])
            if key not in seen:
                seen.add(key)
                keep.append(item)
        unique[kind] = keep
    want_resume = round(n * resume_share)
    take = {"resume": min(len(unique["resume"]), want_resume),
            "posting": min(len(unique["posting"]), n - want_resume)}
    short = n - sum(take.values())
    for kind in ("resume", "posting"):
        extra = min(short, len(unique[kind]) - take[kind])
        take[kind] += extra
        short -= extra
    return unique["resume"][:take["resume"]] + unique["posting"][:take["posting"]]


def split_of(kind, doc_id, dev_share=1 / 3):
    """dev or test, decided by a hash of the document so it never changes."""
    h = int(hashlib.md5(f"{kind}:{doc_id}".encode("utf-8")).hexdigest(), 16)
    return "dev" if (h % 1000) / 1000 < dev_share else "test"


# ---------------------------------------------------------------- the labeling sheet

def write_gold_template(linker, sample, path, seed=573):
    """Write the labeling sheet. An existing sheet is never overwritten.

    Candidates come from every stage of the linker (and the retriever, if it
    has one) and are shuffled, so the sheet does not show which one the
    system would pick. Returns the number of rows written (0 if kept).
    """
    if os.path.exists(path):
        return 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rng = random.Random(seed)
    counters = {"resume": 0, "posting": 0}
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for kind, doc_id, mention, context in sample:
            counters[kind] += 1
            ids = linker.candidates_for(mention, context, k=CANDIDATES_SHOWN)
            rng.shuffle(ids)
            row = {"mention_id": f"{kind[0]}{counters[kind]:03d}", "split": split_of(kind, doc_id),
                   "kind": kind, "doc_id": doc_id, "mention": mention, "context": context,
                   "gold_id": "", "annotator": "", "notes": ""}
            for i in range(CANDIDATES_SHOWN):
                if i < len(ids):
                    row[f"candidate_{i + 1}"] = f"{ids[i]} | {linker.skills[ids[i]]['name']}"
                else:
                    row[f"candidate_{i + 1}"] = ""
            writer.writerow(row)
    return sum(counters.values())


def search_skills(skills, query, n=10):
    """Find Lightcast skills by name while labeling: every word of the query must appear."""
    words = sl.normalize_name(query).split()
    if not words:
        return []
    hits = [s for s in skills if all(w in sl.normalize_name(s["name"]) for w in words)]
    hits.sort(key=lambda s: (len(s["name"]), s["name"]))
    return [(s["id"], s["name"], s.get("category"), s.get("subcategory")) for s in hits[:n]]


def load_gold(path, skills):
    """Labeled rows of the sheet. gold_id may be an ID, 'ID | name' as copied, or NIL.

    Returns (rows, problems, unlabeled); rows with an unknown ID are reported
    in problems and left out, so a typo is caught instead of scored.
    """
    known = {s["id"] for s in skills}
    rows, problems, unlabeled = [], [], 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("gold_id") or "").strip()
            if not raw:
                unlabeled += 1
                continue
            gold = raw.split("|")[0].strip()
            if gold.upper() in NIL_WORDS:
                rows.append({**row, "gold_id": None})
            elif gold in known:
                rows.append({**row, "gold_id": gold})
            else:
                problems.append(f"{row.get('mention_id')}: unknown skill ID {gold!r}")
    return rows, problems, unlabeled


# ---------------------------------------------------------------- scoring

def _rate(items, key="correct"):
    items = [x for x in items if x[key] is not None]
    return round(sum(1 for x in items if x[key]) / len(items), 4) if items else None


def metrics(per_mention):
    """Accuracy@1 over mentions that have a Lightcast skill; NIL accuracy over those that do not."""
    linkable = [p for p in per_mention if not p["gold_nil"]]
    nil = [p for p in per_mention if p["gold_nil"]]
    n = len(per_mention)
    return {
        "mentions": n,
        "linkable": len(linkable),
        "gold_nil": len(nil),
        "accuracy@1": _rate(linkable),
        "nil_accuracy": _rate(nil),
        "overall_accuracy": _rate(per_mention),
        "candidate_recall": _rate(linkable, "in_candidates"),
        "predicted_nil_rate": round(sum(1 for p in per_mention if p["pred"] is None) / n, 4) if n else None,
    }


def bootstrap(per_mention, key, n=1000, seed=573, alpha=0.05):
    """Percentile bootstrap CI for one metric, resampling mentions."""
    if len(per_mention) < 2:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(n):
        sample = [rng.choice(per_mention) for _ in per_mention]
        value = metrics(sample)[key]
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    return values[int(alpha / 2 * len(values))], values[min(len(values) - 1, int((1 - alpha / 2) * len(values)))]


def evaluate(linker, rows, n_boot=1000, seed=573):
    """Score a linker on labeled rows."""
    per = []
    for row in rows:
        link = linker.link(row["mention"], context=row.get("context"))
        gold = row["gold_id"]
        pool = {link["id"], *link["candidates"], *(s["id"] for s in link["suggestions"])}
        pool.discard(None)
        per.append({"mention_id": row.get("mention_id"), "mention": row["mention"], "gold": gold,
                    "pred": link["id"], "method": link["method"], "correct": link["id"] == gold,
                    "gold_nil": gold is None, "in_candidates": (gold in pool) if gold else None})
    return {
        "metrics": metrics(per),
        "ci95": {key: bootstrap(per, key, n_boot, seed) for key in ("accuracy@1", "nil_accuracy")},
        "per_mention": per,
    }


def paired_randomization(a, b, n=10000, seed=573):
    """Two-sided p-value for the accuracy difference of two systems on the same mentions.

    a and b are lists of booleans (correct or not), aligned by mention.
    """
    diffs = [int(x) - int(y) for x, y in zip(a, b)]
    observed = abs(sum(diffs))
    if observed == 0:
        return 1.0
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        total = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(total) >= observed:
            hits += 1
    return (hits + 1) / (n + 1)


def compare(linkers, rows, n_boot=1000, seed=573):
    """Score each configuration; the first one is the baseline for the p-values.

    linkers maps a name to a Linker. Returns (table_rows, results).
    """
    results = {name: evaluate(linker, rows, n_boot, seed) for name, linker in linkers.items()}
    names = list(linkers)
    base = results[names[0]]["per_mention"]
    base_linkable = [p["correct"] for p in base if not p["gold_nil"]]
    table = []
    for name in names:
        res = results[name]
        per = res["per_mention"]
        m = res["metrics"]
        row = {"config": name, "accuracy@1": m["accuracy@1"], "acc@1_ci95": res["ci95"]["accuracy@1"],
               "nil_accuracy": m["nil_accuracy"], "nil_ci95": res["ci95"]["nil_accuracy"],
               "overall_accuracy": m["overall_accuracy"], "candidate_recall": m["candidate_recall"],
               "mentions": m["mentions"]}
        if name != names[0]:
            linkable = [p["correct"] for p in per if not p["gold_nil"]]
            row["p_acc@1_vs_baseline"] = paired_randomization(linkable, base_linkable, seed=seed)
            row["p_overall_vs_baseline"] = paired_randomization(
                [p["correct"] for p in per], [p["correct"] for p in base], seed=seed)
        table.append(row)
    return table, results
