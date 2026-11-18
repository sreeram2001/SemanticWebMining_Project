# LinkedIn Assistant

CSE 573 Semantic Web Mining, Group 5, Fall 2026

A resume and job matching assistant that extracts facts from resumes and postings into a skill knowledge graph, ranks jobs by fit, and explains each match with evidence and skill gaps.

## What's built

1. Preprocessing: 123,782 job postings and 1,368 resumes in 12 categories after cleaning. Emails, phone numbers and URLs are masked in resumes.
2. Ontology (`ontology.json`): 7 node types and 8 relationships. Extracted relationships store the source span and its offsets.
3. Grounded extraction (`extraction.py`): Qwen3-235B on ASU Voyager. A fact is kept only if its span is found in the text and it passes the type checks.
4. Extraction evaluation (`evaluation.py`): a gold set of 60 resumes and 60 postings (not labeled yet), per-field precision, recall and F1, and a model comparison on a dev/test split.
5. Skill linking: alias table and fuzzy matching to LinkedIn's 35 skill categories.

## Layout

```text
SemanticWebMining_Project/
├── CODE/
│   ├── SWM_Project_Linkedin_Assistant.ipynb
│   ├── ontology.json
│   ├── extraction.py
│   ├── evaluation.py
│   ├── test_extraction.py
│   └── test_evaluation.py
├── DATA/                       # downloaded at runtime, not committed
└── EVALUATIONS/
    ├── gold/                   # extraction gold set, one JSON per document
    ├── gold_second/            # copies for a second annotator
    ├── gold_split.json         # frozen dev/test split
    ├── sample_extraction.json
    └── skill_linking_demo.json
```

## Data

LinkedIn Job Postings 2023-24 (Kaggle, CC BY-SA 4.0) and the Resume Dataset (Kaggle, CC0).
