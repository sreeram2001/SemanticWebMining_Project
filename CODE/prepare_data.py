import argparse
import os
import re

import pandas as pd

POSTING_COLS = ["job_id", "company_name", "title", "description", "skills_desc",
                "max_salary", "min_salary", "med_salary", "pay_period",
                "location", "company_id", "formatted_work_type",
                "formatted_experience_level", "remote_allowed"]
RELEVANT_CATEGORIES = [
    "INFORMATION-TECHNOLOGY", "ENGINEERING", "FINANCE", "BANKING",
    "BUSINESS-DEVELOPMENT", "ACCOUNTANT", "SALES", "CONSULTANT",
    "HR", "DIGITAL-MEDIA", "DESIGNER", "HEALTHCARE"]
MIN_DESCRIPTION_CHARS = 50

_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE = re.compile(r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
_URL = re.compile(r"https?://\S+")


def mask_pii(text):
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PHONE.sub("[PHONE]", text)
    return _URL.sub("[URL]", text)


def clean_postings(df):
    df = df.dropna(subset=["description"])
    df = df.drop_duplicates(subset=["job_id"])
    return df[df["description"].str.len() >= MIN_DESCRIPTION_CHARS]


def clean_resumes(df):
    df = df[df["Category"].isin(RELEVANT_CATEGORIES)].copy()
    df["Resume_clean"] = df["Resume_str"].fillna("").apply(mask_pii)
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description="Clean the raw datasets into DATA/processed/.")
    p.add_argument("--postings", required=True, help="path to postings.csv")
    p.add_argument("--resumes", required=True, help="path to Resume.csv")
    p.add_argument("--out", default="DATA/processed")
    a = p.parse_args(argv)

    posts = pd.read_csv(a.postings, usecols=POSTING_COLS)
    print(f"Raw postings: {len(posts):,}")
    posts = clean_postings(posts)
    res = pd.read_csv(a.resumes)
    print(f"Raw resumes: {len(res):,}")
    res = clean_resumes(res)

    os.makedirs(a.out, exist_ok=True)
    posts.to_csv(os.path.join(a.out, "postings_cleaned.csv"), index=False)
    res.to_csv(os.path.join(a.out, "resumes_cleaned.csv"), index=False)
    print(f"Saved {len(posts):,} postings and {len(res):,} resumes "
          f"({res['Category'].nunique()} categories) to {a.out}")


if __name__ == "__main__":
    main()
