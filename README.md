# CSE 573 Semantic Web Mining, Group 5, Fall 2026

## Project : Linkedin Assistant

## Group 5 :
* Aakash Poondi Balaji
* Chandana Vinay Kumar
* Chirag Rajesh Nawale
* Kaushal Laxmikant Apastambh
* Richitha Rekula
* Sreeram Saravana Prasad

A resume and job matching assistant that extracts facts from resumes and postings into a skill knowledge graph, ranks jobs by fit, and explains each match with evidence and skill gaps.

## What's built

1. Preprocessing: 123,782 job postings and 1,368 resumes in 12 categories after cleaning. Emails, phone numbers and URLs are masked in resumes.
2. Ontology (`ontology.json`): 7 node types and 8 relationships. Extracted relationships store the source span and its offsets.
3. Grounded extraction (`extraction.py`): Qwen3-235B on ASU Voyager. A fact is kept only if its span is found in the text and it passes the type checks.
4. Extraction evaluation (`evaluation.py`): a gold set of 60 resumes and 60 postings (not labeled yet), per-field precision, recall and F1, and a model comparison on a dev/test split.
5. Skill linking: alias table and fuzzy matching to LinkedIn's 35 skill categories.
6. Retrieval baselines (`retrieval.py`, `rerank.py`, `run_baselines.py`): BM25, dense (SBERT), weighted TF-IDF + SBERT hybrid, Reciprocal Rank Fusion, and a cross-encoder reranker over the RRF top results.
7. Chat assistant (`assistant.py`, `app.py`): a Gradio web UI over the modules above. Load a resume (sample, .txt/.pdf upload or pasted text), then ask for the top jobs, the skill evidence behind a match, missing skills, job details, or a side-by-side comparison, and switch the ranking method.

## Layout

```text
SemanticWebMining_Project/
├── CODE/
│   ├── SWM_Project_Linkedin_Assistant.ipynb
│   ├── ontology.json
│   ├── extraction.py
│   ├── evaluation.py
│   ├── skill_linking.py
│   ├── skill_aliases.json
│   ├── linking_evaluation.py
│   ├── retrieval.py
│   ├── rerank.py
│   ├── run_baselines.py
│   ├── assistant.py            # chat logic over the modules above
│   ├── app.py                  # Gradio web UI
│   ├── requirements.txt
│   ├── sample_data/            # small synthetic corpus for the offline demo
│   └── test_*.py
├── DATA/                       # downloaded at runtime, not committed
└── EVALUATIONS/
    ├── gold/                   # extraction gold set, one JSON per document
    ├── gold_second/            # copies for a second annotator
    ├── gold_split.json         # frozen dev/test split
    ├── sample_extraction.json
    └── skill_linking_demo.json
```

## Run the chat assistant

From the repository root:

```bash
pip install -r CODE/requirements.txt
python CODE/app.py --sample       # offline demo on CODE/sample_data, no keys needed
python CODE/app.py                # real data
```

Then open http://127.0.0.1:7860. `--share` gives a public link (e.g. from Colab), and `--limit-postings N` ranks only the first N postings for a faster start.

The real-data run reads:

- `DATA/processed/postings_cleaned.csv` and `resumes_cleaned.csv` (notebook Step 1)
- `DATA/lightcast/skills.json` (`skill_linking.download_and_cache`)
- `DATA/extracted/postings.jsonl` and `resumes.jsonl` (notebook Step 3b), optional
- `DATA/embeddings/postings.npz`, the same embedding cache as `run_baselines.py`, built on first run if missing

Ranks and scores come from the team's rankers, with the resume text as the query. Skills come from the extraction records, linked to Lightcast with `skill_linking.Linker`. Each job skill is checked with `skill_linking.check_missing`, which searches the resume text before calling a skill missing. With `ASU_VOYAGER_KEY` set, a resume or posting with no extraction record is extracted on demand with `extraction.Extractor`. Uploaded resumes are never written to disk.

Without `sentence-transformers` (or with `--no-models`) only BM25 is offered. Tests: `python -m unittest discover -s CODE`.

## Data

LinkedIn Job Postings 2023-24 (Kaggle, CC BY-SA 4.0) and the Resume Dataset (Kaggle, CC0).
