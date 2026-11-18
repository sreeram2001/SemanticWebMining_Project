"""Step 1: clean Kaggle data, mask PII, dedupe reposts, freeze the dev/test split."""

import hashlib
import json
import os
import random
import re
from collections import defaultdict

from core.config import write_json_atomic

POSTING_COLUMNS = ["job_id", "company_name", "title", "description", "skills_desc",
                   "max_salary", "min_salary", "med_salary", "pay_period", "location",
                   "company_id", "formatted_work_type", "formatted_experience_level",
                   "remote_allowed"]

RELEVANT_CATEGORIES = [
    "INFORMATION-TECHNOLOGY", "ENGINEERING", "FINANCE", "BANKING",
    "BUSINESS-DEVELOPMENT", "ACCOUNTANT", "SALES", "CONSULTANT",
    "HR", "DIGITAL-MEDIA", "DESIGNER", "HEALTHCARE",
]

MIN_DESCRIPTION = 50

# ------------------------------------------------------------------ PII

_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE = re.compile(r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
_URL = re.compile(r"https?://\S+|www\.\S+")
_STREET = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][a-z]+\s){1,3}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl)\b\.?"
)


def mask_pii(text):
    """Mask emails, phones, URLs and addresses."""
    if not isinstance(text, str):
        return text
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _URL.sub("[URL]", text)
    text = _STREET.sub("[ADDRESS]", text)
    return text


# ------------------------------------------------------------------ cleaning

def _fingerprint(company, title, description):
    """Repost key: company, title, text minus digits/whitespace."""
    body = re.sub(r"[\d\s]+", "", str(description).lower())
    key = f"{str(company).lower()}|{str(title).lower()}|{body}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def posting_problems(row):
    """List of schema problems for a posting (empty = ok)."""
    problems = []
    if not str(row.get("job_id", "")).strip():
        problems.append("missing job_id")
    if not isinstance(row.get("title"), str) or not row["title"].strip():
        problems.append("missing title")
    desc = row.get("description")
    if not isinstance(desc, str) or len(desc) < MIN_DESCRIPTION:
        problems.append("description missing or too short")
    for col in ("min_salary", "max_salary"):
        v = row.get(col)
        if v is not None and v == v and not (0 <= float(v) <= 10_000_000):  # v == v skips NaN
            problems.append(f"{col} out of range")
    lo, hi = row.get("min_salary"), row.get("max_salary")
    if lo is not None and hi is not None and lo == lo and hi == hi and lo > hi:
        problems.append("min_salary > max_salary")
    return problems


def clean_postings(df):
    """Return (clean_df, report)."""
    report = {"raw": len(df)}
    df = df.dropna(subset=["description"])
    df = df.drop_duplicates(subset=["job_id"])
    report["after_id_dedup"] = len(df)
    df = df[df["description"].str.len() >= MIN_DESCRIPTION]
    fp = [_fingerprint(c, t, d) for c, t, d in zip(df["company_name"], df["title"], df["description"])]
    df = df.assign(_fp=fp).drop_duplicates(subset=["_fp"]).drop(columns="_fp")
    report["after_repost_dedup"] = len(df)
    bad = [i for i, row in zip(df.index, df.to_dict("records")) if posting_problems(row)]
    df = df.drop(index=bad)
    report["schema_rejected"] = len(bad)
    report["clean"] = len(df)
    return df, report


def clean_resumes(df, categories=RELEVANT_CATEGORIES):
    report = {"raw": len(df)}
    df = df[df["Category"].isin(categories)].copy()
    df = df.dropna(subset=["Resume_str"])
    df = df.drop_duplicates(subset=["ID"])
    df["Resume_clean"] = df["Resume_str"].apply(mask_pii)
    report["clean"] = len(df)
    report["categories"] = int(df["Category"].nunique())
    return df, report


def preprocess(raw_postings, raw_resumes, out_postings, out_resumes):
    import pandas as pd
    posts = pd.read_csv(raw_postings, usecols=POSTING_COLUMNS)
    posts, preport = clean_postings(posts)
    res, rreport = clean_resumes(pd.read_csv(raw_resumes))
    for path in (out_postings, out_resumes):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    posts.to_csv(out_postings, index=False)
    res.to_csv(out_resumes, index=False)
    return {"postings": preport, "resumes": rreport}


# ------------------------------------------------------------------ split and corpus

def stratified_split(groups, dev_ratio=0.5, seed=573):
    """{doc_id: 'dev'|'test'}, stratified by group; deterministic per seed."""
    rng = random.Random(seed)
    out, carry = {}, 0.0
    for group in sorted(groups):
        ids = sorted(str(i) for i in groups[group])
        rng.shuffle(ids)
        exact = len(ids) * dev_ratio + carry
        n_dev = max(0, min(len(ids), int(round(exact))))
        carry = exact - n_dev
        for i, doc_id in enumerate(ids):
            out[doc_id] = "dev" if i < n_dev else "test"
    return out


def make_split(resumes, dev_ratio=0.5, seed=573):
    """resumes: iterable of (resume_id, category)."""
    groups = defaultdict(list)
    for rid, cat in resumes:
        groups[cat].append(rid)
    return stratified_split(groups, dev_ratio, seed)


def save_split(split, path, seed, dev_ratio):
    """Write the split once; never overwrite."""
    if os.path.exists(path):
        return False
    write_json_atomic(path, {"seed": seed, "dev_ratio": dev_ratio, "resumes": dict(sorted(split.items()))})
    return True


def load_split(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["resumes"]


def sample_postings(df, n, seed=573):
    """Sample n postings stratified by experience level."""
    if n is None or n >= len(df):
        return df
    level = df["formatted_experience_level"].fillna("Unknown")
    frac = n / len(df)
    parts = [g.sample(n=max(1, round(len(g) * frac)), random_state=seed) for _, g in df.groupby(level)]
    import pandas as pd
    out = pd.concat(parts)
    if len(out) < n:                                     # top up rounding shortfall
        rest = df.drop(index=out.index)
        out = pd.concat([out, rest.sample(n=n - len(out), random_state=seed)])
    return out.sample(n=n, random_state=seed)


def file_hash(*paths):
    """Short content hash of the inputs."""
    h = hashlib.sha256()
    for p in paths:
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
    return h.hexdigest()[:12]
