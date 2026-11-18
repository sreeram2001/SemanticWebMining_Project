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
5. Skill linking (`skill_linking.py`): each skill mention is linked to a Lightcast Open Skills ID (alias table, fuzzy and embedding candidates, an LLM decision for unclear cases) or marked NIL. It also exports Neo4j CSV files.
6. Retrieval (`retrieval.py`, `run_baselines.py`): BM25, dense retrieval (SBERT `all-MiniLM-L6-v2`), the previous team's ranker, and RRF (k = 60) over BM25 and dense. Run on all 123,782 postings for the 60 resumes in the gold split.
   - The previous team's ranker is reproduced exactly from their `phase2_pipeline.py` (https://github.com/Panav0075/Linkedin-AI-Assistant): 0.5 × TF-IDF + 0.5 × SBERT, job text = title + description, query = their keyword skill list (`prior_skills.json`, copied from their `phase1_pipeline.py`) + resume text. They ranked only the first 1,000 postings; we rank all of them.
7. Reranking and graph scoring (`rerank.py`, `graph_score.py`): a cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescores the top 50 RRF jobs, and the final score is S = b1·CE + b2·G − b3·P.
   - CE is the cross-encoder score, min-max normalized per resume.
   - G is the weighted coverage of the job's linked skills, with partial credit for skills in the same Lightcast subcategory or category (`skill_linking.coverage`).
   - P is the share of required degree and experience requirements the resume explicitly fails. Missing information counts as unknown and is not penalized.
   - The graph ablation sets b2 = 0.
8. Relevance judgments (`pooling.py`): 30 resumes (15 dev, 15 test) chosen from the gold split by category. The top 10 jobs of each method are pooled (973 pairs) and split into one grading sheet per person, with 3 resumes shared by everyone for agreement (weighted Cohen's kappa). Once G and P have data, `tune_ranking.py pool-runs` and `pooling.py --extend` add the full pipeline's top 10 jobs that nobody has graded yet as `grades_<name>_extra.csv`.
9. Tuning and evaluation (`tune_ranking.py`): b2, b3, the partial-credit values, the required-skill weight and the cross-encoder shortlist size (20, 30 or 50) are tuned on dev only. A job counts as relevant for P@10 at grade 1 or higher. Every method is reported with P@10 and nDCG@10, plus ΔnDCG@10 between the full score and the no-graph ablation.
10. Chat assistant (`assistant.py`, `app.py`): a Gradio web UI over the modules above. Load a resume (sample, .txt/.pdf upload or pasted text), then ask for the top jobs, the skill evidence behind a match, missing skills, job details, or a side-by-side comparison, and switch the ranking method.

## Status

| Step | Status |
|---|---|
| Retrieval run on the real data | done (`EVALUATIONS/runs/baselines.json`, with the input data hash) |
| Grading sheets | ready in `EVALUATIONS/relevance/`, not graded yet |
| Documents to extract for G and P | listed in `EVALUATIONS/runs/extraction_ids.json` (30 resumes, 1,336 postings) |
| Linked extraction records for G and P | not produced yet (expected in `DATA/linked/`) |
| Tuning, ablation, test results | waiting on grades and linked records |

## Running the ranking steps

From the repository root:

```bash
python CODE/prepare_data.py --postings <path>/postings.csv --resumes <path>/Resume.csv
python CODE/run_baselines.py --split EVALUATIONS/gold_split.json --k 100 --rerank
python CODE/pooling.py
# after grading: fill the grade column (0, 1 or 2) in EVALUATIONS/relevance/grades_<name>.csv
python CODE/tune_ranking.py pool-runs        # once linked records exist
python CODE/pooling.py --extend EVALUATIONS/runs/full_pool_run.json
python CODE/tune_ranking.py tune
python CODE/tune_ranking.py evaluate --split dev
python CODE/tune_ranking.py evaluate --split test   # once, for the final report
```

`run_baselines.py` caches the posting embeddings in `DATA/embeddings/`. The first full run, which encodes all postings, took about 6 minutes on an M-series Mac, plus another 6 for the previous team's job text (`postings_prior.npz`). `tune_ranking.py` reads linked records from `DATA/linked/resumes.jsonl` and `DATA/linked/postings.jsonl`; without them G and P are 0 for every job.

Tests: `cd CODE && python -m unittest discover -p "test_*.py"`

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
│   ├── extraction.py, evaluation.py          # grounded extraction and its evaluation
│   ├── skill_linking.py, linking_evaluation.py, skill_aliases.json
│   ├── prepare_data.py                       # cleaning, same rules as the notebook
│   ├── retrieval.py, run_baselines.py        # BM25, dense, hybrid, RRF, metrics
│   ├── prior_skills.json                     # previous team's skill keywords
│   ├── rerank.py, graph_score.py             # cross-encoder, G, P and S
│   ├── pooling.py, tune_ranking.py           # judgments, tuning, ablation
│   └── test_*.py
├── DATA/                       # downloaded at runtime, not committed
└── EVALUATIONS/
    ├── gold/                   # extraction gold set, one JSON per document
    ├── gold_second/            # copies for a second annotator
    ├── gold_split.json         # frozen dev/test split
    ├── relevance/              # pool.json, resumes.csv, grades_<name>.csv
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
