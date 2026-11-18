"""Extraction gold set and scoring.

Gold documents are JSON files that annotators fill in by hand, one file per
document. Model output (the JSONL records written by extraction.run_batch) is
scored against them per field with precision, recall and F1, for the grounded
pipeline and for the same output with the grounding check switched off.
"""

import glob
import json
import os
import random
import re
from collections import Counter

from extraction import (
    FIELDS, SINGLE_FIELDS, UNSUPPORTED, Extractor, ground, load_records, run_batch, summarize,
)

_PUNCT = " \t\n.,;:()[]{}\"'*-•"
_CHAR_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                           "–": "-", "—": "-"})


def _norm(value):
    if value is None:
        return ""
    value = re.sub(r"\s+", " ", str(value).translate(_CHAR_MAP)).strip(_PUNCT)
    return value.lower()


def fact_key(field, fact):
    """What has to agree for a predicted fact to count as the gold fact.

    Spans are not part of the key: where a fact is quoted from is scored by
    the unsupported-fact rate, and a skill mentioned twice may be labeled at
    either mention.
    """
    if field in ("skills", "required_skills"):
        return _norm(fact.get("name"))
    if field in ("degrees", "required_degrees"):
        return f"{fact.get('level')}|{_norm(fact.get('field'))}"
    if field == "experience":
        return _norm(fact.get("title"))
    if field == "experience_required":
        return str(float(fact["min_years"]))
    raise ValueError(f"unknown field: {field}")


def _as_list(field, value):
    if field in SINGLE_FIELDS:
        return [] if value is None else [value]
    return value or []


def _keys(kind, facts):
    out = {}
    for field in FIELDS[kind]:
        keys = [fact_key(field, f) for f in _as_list(field, facts.get(field))]
        # one gold fact per distinct key, so a skill listed twice counts once
        out[field] = Counter(set(keys)) if field in ("skills", "required_skills") else Counter(keys)
    return out


def ungrounded(record):
    """The record's facts as they would be without the span check.

    Facts rejected only because their span was missing are put back; facts
    that failed the type checks stay out, so the ablation isolates grounding.
    """
    kind = record["kind"]
    facts = {f: list(_as_list(f, v)) for f, v in record["facts"].items()}
    for r in record["rejected"]:
        if r["reason"] != UNSUPPORTED:
            continue
        fact, _ = FIELDS[kind][r["field"]](r["fact"])
        facts[r["field"]].append(fact)
    for field in SINGLE_FIELDS & facts.keys():
        facts[field] = facts[field][0] if facts[field] else None
    return facts


# --- gold set files -----------------------------------------------------------

# Same patterns as Step 1's mask_pii. Step 1 masks only resumes, but gold
# files are committed to a public repo and postings carry recruiters'
# emails and phone numbers.
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
    """Write one blank gold file per (doc_id, text, category) tuple.

    Existing files are never overwritten, because they may already hold an
    annotator's work. Returns (written, kept) counts.
    """
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


def load_gold(gold_dir):
    """Read labeled gold files.

    A file counts as labeled once its "annotator" field is filled in. Every
    labeled fact goes through the same span and type checks as model output,
    so a typo in a copied span is reported instead of silently scored.
    Returns (gold, problems, unlabeled): gold maps (kind, doc_id) to facts.
    """
    gold, problems, unlabeled = {}, [], 0
    for path in sorted(glob.glob(os.path.join(gold_dir, "*.json"))):
        name = os.path.basename(path)
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{name}: unreadable ({exc})")
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
    """(doc_id, text) pairs for labeled gold documents, to run models on."""
    out = []
    for path in sorted(glob.glob(os.path.join(gold_dir, f"{kind}_*.json"))):
        # a file an annotator is halfway through saving is skipped here;
        # load_gold is what reports it
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if str(doc.get("annotator", "")).strip() and isinstance(doc.get("text"), str):
            out.append((str(doc["doc_id"]), doc["text"]))
    return out


# --- scoring ------------------------------------------------------------------

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    # written in counts so predicting nothing against real gold scores 0, not undefined
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return p, r, f1


def _counts(kind, predicted, gold):
    """Per-document tp/fp/fn per field. predicted and gold map doc_id -> facts."""
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
        # a field with no gold facts says nothing about recall, so leave it
        # out of the macro average rather than scoring it 0 or 1
        if tp + fn:
            f1s.append(f1)
    macro = sum(f1s) / len(f1s) if f1s else None
    return fields, macro


def bootstrap_ci(kind, per_doc, n=1000, seed=573, alpha=0.05):
    """Percentile bootstrap CI for macro-F1, resampling documents."""
    ids = list(per_doc)
    if len(ids) < 2:
        return None
    rng = random.Random(seed)
    scores = []
    for _ in range(n):
        sample = [rng.choice(ids) for _ in ids]
        _, macro = _aggregate(kind, per_doc, sample)
        if macro is not None:
            scores.append(macro)
    if not scores:
        return None
    scores.sort()
    lo = scores[int(alpha / 2 * len(scores))]
    hi = scores[min(len(scores) - 1, int((1 - alpha / 2) * len(scores)))]
    return lo, hi


def for_kind(gold, kind):
    """Narrow load_gold's (kind, doc_id) mapping to doc_id -> facts."""
    return {d: f for (k, d), f in gold.items() if k == kind}


def score(kind, predicted, gold):
    """Score predicted facts against gold; both map doc_id -> facts.

    Documents in the gold set with no prediction (the model errored on them)
    count as predicting nothing, so failures lower recall instead of
    disappearing from the score.
    """
    if not gold:
        return None
    per_doc = _counts(kind, predicted, gold)
    fields, macro = _aggregate(kind, per_doc, list(per_doc))
    missing = sum(1 for d in gold if d not in predicted)
    return {"documents": len(gold), "missing_predictions": missing, "fields": fields,
            "macro_f1": macro, "macro_f1_ci95": bootstrap_ci(kind, per_doc), "per_doc": per_doc}


def evaluate_run(path, gold, kind):
    """Score one model's JSONL output, with and without grounding.

    gold is load_gold's mapping; only documents of this kind are scored.
    """
    gold = for_kind(gold, kind)
    records = [r for r in load_records(path) if r["kind"] == kind and r["status"] == "ok"]
    grounded = {r["doc_id"]: r["facts"] for r in records}
    plain = {r["doc_id"]: ungrounded(r) for r in records}
    return {"grounded": score(kind, grounded, gold), "ungrounded": score(kind, plain, gold)}


def agreement(dir_a, dir_b):
    """Agreement between two annotators on the documents both labeled.

    F1 is symmetric, so scoring one annotator against the other gives the
    usual pairwise F1 agreement for span extraction.
    """
    a, _, _ = load_gold(dir_a)
    b, _, _ = load_gold(dir_b)
    out = {}
    for kind in FIELDS:
        shared = a.keys() & b.keys()
        result = score(kind, for_kind({k: a[k] for k in shared}, kind), for_kind({k: b[k] for k in shared}, kind))
        if result:
            result.pop("per_doc")
            out[kind] = result
    return out


# --- model comparison ----------------------------------------------------------

def run_slug(model):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model)


def compare_models(client, models, gold_dir, out_dir, **extractor_args):
    """Run each model on the labeled gold documents and score it.

    Documents are extracted whole (no truncation) so a fact the annotator
    labeled late in a long posting is not counted as a miss. Runs append to
    one JSONL file per model and kind, so an interrupted comparison resumes.
    """
    gold, _, _ = load_gold(gold_dir)
    results = {}
    for model in models:
        extractor = Extractor(client, model, max_chars=None, **extractor_args)
        for kind in FIELDS:
            docs = gold_docs(gold_dir, kind)
            if not docs:
                continue
            path = os.path.join(out_dir, f"{run_slug(model)}_{kind}.jsonl")
            counts = run_batch(extractor, kind, docs, path, log_every=0)
            entry = {"run": counts, "constrained": extractor.constrained,
                     "facts": summarize(load_records(path))}
            entry.update(evaluate_run(path, gold, kind))
            results.setdefault(model, {})[kind] = entry
    return results


def comparison_rows(results):
    """Flatten compare_models output into one row per model, kind and mode."""
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
                row = {"model": model, "kind": kind, "mode": mode,
                       "macro_f1": s["macro_f1"], "ci95": s["macro_f1_ci95"],
                       "failed_docs": s["missing_predictions"]}
                for field, c in s["fields"].items():
                    row[f"{field}_f1"] = c["f1"]
                # the grounded pipeline drops every unsupported fact, so only
                # the ablation keeps any in its output
                row["unsupported_rate"] = (round(unsupported / total, 4) if total else None) \
                    if mode == "ungrounded" else 0.0
                rows.append(row)
    return rows
