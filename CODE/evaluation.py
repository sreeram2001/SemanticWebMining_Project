"""Extraction gold set, scoring and model comparison."""

import glob
import json
import os
import random
import re
from collections import Counter

from extraction import FIELDS, SINGLE_FIELDS, Extractor, ground, load_records, run_batch, summarize

_PUNCT = " \t\n.,;:()[]{}\"'*-•"
_CHAR_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                           "–": "-", "—": "-"})


def _norm(value):
    if value is None:
        return ""
    value = re.sub(r"\s+", " ", str(value).translate(_CHAR_MAP)).strip(_PUNCT)
    return value.lower()


def fact_key(field, fact):
    # Spans aren't part of the key: a skill mentioned twice may be labeled at
    # either mention, and span quality is measured by the unsupported rate.
    if field in ("skills", "required_skills"):
        return _norm(fact.get("name"))
    if field in ("degrees", "required_degrees"):
        return f"{fact.get('level')}|{_norm(fact.get('field'))}"
    if field == "experience":
        return _norm(fact.get("title"))
    if field == "experience_required":
        years = fact.get("min_years")
        if isinstance(years, (int, float)) and not isinstance(years, bool):
            return str(float(years))
        return str(years)
    raise ValueError(f"unknown field: {field}")


def _as_list(field, value):
    if field in SINGLE_FIELDS:
        return [] if value is None else [value]
    return value or []


def _keys(kind, facts):
    out = {}
    for field in FIELDS[kind]:
        keys = [fact_key(field, f) for f in _as_list(field, facts.get(field))]
        # a skill listed twice counts once
        out[field] = Counter(set(keys)) if field in ("skills", "required_skills") else Counter(keys)
    return out


def ungrounded(record):
    """Baseline: the same model output with no span or type checks."""
    facts = {f: list(_as_list(f, v)) for f, v in record["facts"].items()}
    for r in record["rejected"]:
        if isinstance(r["fact"], dict) and r["reason"] != "expected a list":
            facts[r["field"]].append(r["fact"])
    for field in SINGLE_FIELDS & facts.keys():
        facts[field] = facts[field][0] if facts[field] else None
    return facts


# --- gold set files ---

# Same patterns as Step 1's mask_pii. Step 1 only masks resumes, and the gold
# files are public.
_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE = re.compile(r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
_URL = re.compile(r"https?://\S+")


def mask_contacts(text):
    if not isinstance(text, str):
        return text
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PHONE.sub("[PHONE]", text)
    return _URL.sub("[URL]", text)


def _filename(kind, doc_id):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(doc_id))
    return f"{kind}_{safe}.json"


def write_templates(kind, docs, out_dir):
    """Blank gold file per (doc_id, text, category); never overwrites. Returns (written, kept)."""
    if kind not in FIELDS:
        raise ValueError(f"unknown kind: {kind}")
    os.makedirs(out_dir, exist_ok=True)
    written = kept = 0
    for doc_id, text, category in docs:
        path = os.path.join(out_dir, _filename(kind, doc_id))
        if os.path.exists(path):
            kept += 1
            continue
        blank = {f: (None if f in SINGLE_FIELDS else []) for f in FIELDS[kind]}
        doc = {"doc_id": str(doc_id), "kind": kind, "category": category,
               "annotator": "", "facts": blank, "text": text}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
            f.write("\n")
        written += 1
    return written, kept


def _read_docs(gold_dir, pattern="*.json"):
    for path in sorted(glob.glob(os.path.join(gold_dir, pattern))):
        try:
            with open(path, encoding="utf-8") as f:
                yield os.path.basename(path), json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            yield os.path.basename(path), exc


def load_gold(gold_dir):
    """Labeled gold files -> ({(kind, doc_id): facts}, problems, unlabeled count).

    Labels go through the same span and type checks as model output, so a
    mistyped span is reported instead of scored.
    """
    gold, problems, unlabeled = {}, [], 0
    for name, doc in _read_docs(gold_dir):
        if isinstance(doc, Exception):
            problems.append(f"{name}: unreadable ({doc})")
            continue
        if not str(doc.get("annotator", "")).strip():
            unlabeled += 1
            continue
        kind, text = doc.get("kind"), doc.get("text")
        if kind not in FIELDS or not isinstance(text, str):
            problems.append(f"{name}: missing kind or text")
            continue
        facts, rejected = ground(kind, doc.get("facts") or {}, text)
        for r in rejected:
            problems.append(f"{name}: {r['field']} {json.dumps(r['fact'], ensure_ascii=False)}: {r['reason']}")
        gold[(kind, str(doc["doc_id"]))] = facts
    return gold, problems, unlabeled


def gold_docs(gold_dir, kind):
    """(doc_id, text) for labeled gold documents of one kind."""
    out = []
    for _, doc in _read_docs(gold_dir, f"{kind}_*.json"):
        # half-saved files are reported by load_gold
        if isinstance(doc, Exception):
            continue
        if str(doc.get("annotator", "")).strip() and isinstance(doc.get("text"), str):
            out.append((str(doc["doc_id"]), doc["text"]))
    return out


def freeze_split(gold_dir, path, seed=573):
    """Half dev, half test, stratified by kind and category. Written once, then reused."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        return {tuple(k.split(":", 1)): v for k, v in saved["docs"].items()}

    groups = {}
    for name, doc in _read_docs(gold_dir):
        if isinstance(doc, Exception):
            raise ValueError(f"{name}: unreadable, fix it before freezing the split")
        groups.setdefault((doc["kind"], doc.get("category") or ""), []).append(str(doc["doc_id"]))

    rng = random.Random(seed)
    order = []
    for group in sorted(groups):
        ids = sorted(groups[group])
        rng.shuffle(ids)
        order += [(group[0], d) for d in ids]
    # Alternating along the grouped order keeps each category within one
    # document of half, and the totals exactly half.
    split = {key: ("dev" if i % 2 == 0 else "test") for i, key in enumerate(order)}

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "docs": {f"{k}:{d}": v for (k, d), v in split.items()}}, f, indent=1)
        f.write("\n")
    return split


# --- scoring ---

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    # in counts so predicting nothing against real gold scores 0, not undefined
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return p, r, f1


def _counts(kind, predicted, gold):
    per_doc = {}
    for doc_id, g in gold.items():
        gk = _keys(kind, g)
        pk = _keys(kind, predicted.get(doc_id) or {})
        per_doc[doc_id] = {
            field: (sum((pk[field] & gk[field]).values()),
                    sum((pk[field] - gk[field]).values()),
                    sum((gk[field] - pk[field]).values()))
            for field in FIELDS[kind]
        }
    return per_doc


def _aggregate(kind, per_doc, doc_ids):
    fields, f1s = {}, []
    for field in FIELDS[kind]:
        tp = sum(per_doc[d][field][0] for d in doc_ids)
        fp = sum(per_doc[d][field][1] for d in doc_ids)
        fn = sum(per_doc[d][field][2] for d in doc_ids)
        p, r, f1 = _prf(tp, fp, fn)
        fields[field] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}
        # fields with no gold facts stay out of the macro average
        if tp + fn:
            f1s.append(f1)
    macro = sum(f1s) / len(f1s) if f1s else None
    return fields, macro


def bootstrap_ci(kind, per_doc, n=1000, seed=573, alpha=0.05):
    """95% percentile bootstrap CI for macro-F1, resampling documents."""
    ids = list(per_doc)
    if len(ids) < 2:
        return None
    rng = random.Random(seed)
    scores = []
    for _ in range(n):
        _, macro = _aggregate(kind, per_doc, [rng.choice(ids) for _ in ids])
        if macro is not None:
            scores.append(macro)
    if not scores:
        return None
    scores.sort()
    lo = scores[int(alpha / 2 * len(scores))]
    hi = scores[min(len(scores) - 1, int((1 - alpha / 2) * len(scores)))]
    return lo, hi


def for_kind(gold, kind):
    return {d: f for (k, d), f in gold.items() if k == kind}


def score(kind, predicted, gold):
    """Score {doc_id: facts} against gold. Documents with no prediction count as empty."""
    if not gold:
        return None
    per_doc = _counts(kind, predicted, gold)
    fields, macro = _aggregate(kind, per_doc, list(per_doc))
    missing = sum(1 for d in gold if d not in predicted)
    return {"documents": len(gold), "missing_predictions": missing, "fields": fields,
            "macro_f1": macro, "macro_f1_ci95": bootstrap_ci(kind, per_doc), "per_doc": per_doc}


def evaluate_run(path, gold, kind):
    """Score one JSONL run with and without the checks. gold is load_gold's mapping."""
    gold = for_kind(gold, kind)
    records = [r for r in load_records(path) if r["kind"] == kind and r["status"] == "ok"]
    checked = {r["doc_id"]: r["facts"] for r in records}
    plain = {r["doc_id"]: ungrounded(r) for r in records}
    return {"grounded": score(kind, checked, gold), "ungrounded": score(kind, plain, gold)}


def agreement(dir_a, dir_b):
    """Pairwise F1 between two annotators on the documents both labeled."""
    a, _, _ = load_gold(dir_a)
    b, _, _ = load_gold(dir_b)
    shared = a.keys() & b.keys()
    out = {}
    for kind in FIELDS:
        result = score(kind, for_kind({k: a[k] for k in shared}, kind), for_kind({k: b[k] for k in shared}, kind))
        if result:
            result.pop("per_doc")
            out[kind] = result
    return out


# --- model comparison ---

def run_slug(model):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model)


def compare_models(client, models, gold_dir, out_dir, split=None, part="dev", **extractor_args):
    """Run and score each model on the labeled gold documents in one part of the split.

    Documents are extracted whole so facts late in a long posting aren't
    counted as misses. The prompt hash is in the file name, so editing the
    prompt starts a fresh run instead of resuming an old one.
    """
    gold, _, _ = load_gold(gold_dir)
    if split is not None:
        gold = {k: v for k, v in gold.items() if split.get(k) == part}
    results = {}
    for model in models:
        extractor = Extractor(client, model, max_chars=None, **extractor_args)
        for kind in FIELDS:
            docs = [(d, t) for d, t in gold_docs(gold_dir, kind) if (kind, d) in gold]
            if not docs:
                continue
            config = extractor.config(kind)
            path = os.path.join(out_dir, f"{run_slug(model)}_{kind}_{config['prompt']}.jsonl")
            counts = run_batch(extractor, kind, docs, path, log_every=0)
            entry = {"run": counts, "part": part if split is not None else "all",
                     "config": dict(config, constrained=extractor.constrained),
                     "facts": summarize(load_records(path))}
            entry.update(evaluate_run(path, gold, kind))
            results.setdefault(model, {})[kind] = entry
    return results


def comparison_rows(results):
    """One row per model, kind and mode."""
    rows = []
    for model, kinds in results.items():
        for kind, entry in kinds.items():
            fields = entry["facts"]["fields"].values()
            unsupported = sum(c["unsupported"] for c in fields)
            total = sum(c["kept"] + c["unsupported"] + c["invalid"] for c in fields)
            for mode in ("grounded", "ungrounded"):
                s = entry[mode]
                if s is None:
                    continue
                row = {"model": model, "kind": kind, "part": entry["part"], "mode": mode,
                       "macro_f1": s["macro_f1"], "ci95": s["macro_f1_ci95"],
                       "failed_docs": s["missing_predictions"]}
                for field, c in s["fields"].items():
                    row[f"{field}_f1"] = c["f1"]
                # grounding drops every unsupported fact; only the baseline keeps them
                row["unsupported_rate"] = (round(unsupported / total, 4) if total else None) \
                    if mode == "ungrounded" else 0.0
                rows.append(row)
    return rows
