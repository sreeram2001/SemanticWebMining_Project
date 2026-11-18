# Synthetic sample

A small, made-up corpus that lets `python CODE/app.py --sample` and `test_assistant.py` run offline, with no API keys or downloads. These are not results.

- `postings_cleaned.csv`, `resumes_cleaned.csv`: 26 postings and 20 resumes, in the same columns as `DATA/processed/`
- `skills.json`: 107 skills in the Lightcast cache format of `skill_linking.save_skills`. The IDs are fake (`SAMPLE001`, ...).
- `extracted/postings.jsonl`, `extracted/resumes.jsonl`: extraction records in the `extraction.run_batch` format, made by a rule-based extractor (`"model": "rules"`), not Qwen3
