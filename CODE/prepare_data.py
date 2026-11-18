"""Clean the raw datasets into DATA/processed/ for the retrieval scripts.

    python CODE/prepare_data.py --postings job_posting/postings.csv \\
        --resumes resume/Resume/Resume.csv

This repeats the notebook's Step 1 so the rankers can run on a laptop with the
downloaded Kaggle files, without Colab. The rules are the notebook's: drop
postings with no description, duplicate ids or a description under 50 characters;
keep the 12 resume categories that overlap with the postings; mask emails,
phone numbers and URLs in resumes. The output files are the same:
  DATA/processed/postings_cleaned.csv
  DATA/processed/resumes_cleaned.csv   (adds the Resume_clean column)
If the team's notebook rules change, change them here too.
"""
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
    """Mask emails, phone numbers and URLs (same order as the notebook)."""
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
