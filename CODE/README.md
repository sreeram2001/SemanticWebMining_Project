# LinkedIn Assistant — Knowledge-Graph Conversational Agent

CSE 573: Semantic Web Mining | Group 5, Project 3 | Fall 2026

## What's Done (Steps 1–9)

| Step | What | Output |
|------|------|--------|
| 1 | **Data Preprocessing** | 123,782 cleaned job postings, 1,368 resumes (12 categories), PII masked |
| 2 | **Domain Ontology** | 7 node types, 8 relationship types → `CODE/resources/ontology.json`; extracted relationships carry a source span and character offsets |
| 3 | **LLM Extraction** | Structured extraction (skills, degrees, experience) using Qwen3-235B via ASU Voyager |
| 3b | **Grounded Extraction** | `CODE/extract/extraction.py`: schema-constrained output; a fact is kept only if its span is found in the text and it passes the ontology's type checks; unsupported-fact rate per field |
| 3c | **Extraction Evaluation** | `CODE/extract/evaluation.py`: 60 + 60 hand-labeled gold set, per-field P/R/F1 and macro-F1 with bootstrap 95% CI, grounded vs. ungrounded ablation, model comparison, inter-annotator agreement |
| 4 | **Skill Linking (first version)** | Fuzzy matching to LinkedIn's 35-category taxonomy in the notebook; superseded by 4a–4b |
| 4a | **Lightcast download** | `CODE/linking/skill_linking.py`: Lightcast Open Skills cache and name index |
| 4b | **Entity linking** | `CODE/linking/entity_linking.py`: alias fast path → candidate retrieval → ID or NIL; accuracy@1 vs. alias-only baseline |
| 5 | **Knowledge graph** | `CODE/kg/graph.py`: graph per `ontology.json` with provenance, integrity checks, Neo4j loader |
| 6–7 | **Ranking** | `CODE/rank/retrieval.py`, `CODE/rank/ranking.py`: BM25 + dense + RRF, cross-encoder, S = β₁CE + β₂G − β₃P, missed-skill recovery |
| 8 | **Evaluation** | `CODE/rank/ranking_eval.py`: two-tier labels, bootstrap CIs, paired randomization tests, graph ablation; `eval-extraction` step for extraction P/R/F1 |
| 9 | **Chat interface** | `CODE/assistant/chat.py` + `CODE/app.py`: Gradio chat with ranked jobs, verified explanations, skill gaps, resume upload |

End-to-end commands, labeling, the real Colab run and report assumptions: see **`CODE/PIPELINE.md`**
(`python CODE/run_pipeline.py --sample all` runs everything offline on synthetic data).

## What's Next

| Step | Owner | What to Build |
|------|-------|---------------|
| — | All | Real run in Colab (`CODE/notebooks/Run_Pipeline_Colab.ipynb`) with `--require-models` |
| — | All | Label the extraction, linking and relevance sets (`run_pipeline.py sheets`, then `merge-labels`) |
| — | All | Tune on dev, then Phase 5: `evaluate --split test --tuned` and `eval-extraction` once per configuration |

## Project Structure
```
SemanticWebMining_Project/
├── CODE/
│ ├── run_pipeline.py      # CLI: every pipeline step (python CODE/run_pipeline.py --help)
│ ├── app.py               # Gradio chat demo
│ ├── config.json          # All settings
│ ├── core/                # config.py (settings, helpers), pipeline.py (wires the stages)
│ ├── dataprep/            # preprocess.py (Step 1), labeling.py (team labeling sheets)
│ ├── extract/             # extraction.py (3b), rule_extractor.py (baseline), evaluation.py (3c)
│ ├── linking/             # skill_linking.py (4a Lightcast), entity_linking.py (4b)
│ ├── kg/                  # graph.py (Step 5 knowledge graph, Neo4j)
│ ├── rank/                # retrieval.py (6), ranking.py (7), ranking_eval.py (8)
│ ├── assistant/           # explanations.py, chat.py (Step 9)
│ ├── resources/           # ontology.json, skill_aliases.json
│ ├── tests/               # 107 unit / end-to-end tests
│ ├── notebooks/           # SWM_Project_Linkedin_Assistant.ipynb, Run_Pipeline_Colab.ipynb
│ ├── sample_data/         # Offline synthetic sample (not results)
│ └── README.md, PIPELINE.md, requirements.txt
├── DATA/                  # Downloaded at runtime (not committed)
│ ├── linkedin-job-postings/, resume-dataset/   # Kaggle datasets
│ ├── processed/           # Cleaned CSVs
│ ├── extracted/           # Grounded extraction output (JSONL, resumable)
│ ├── lightcast/           # Taxonomy, linking caches
│ └── embeddings/          # Cached posting embeddings
└── EVALUATIONS/
  ├── gold/, gold_second/  # Extraction gold set (+ second annotator)
  ├── labeling/            # Linking and relevance sheets per annotator
  ├── runs/                # eval_*.md / .json, tuned weights, graph.cypher
  └── sample_extraction.json, skill_linking_demo.json
```

## Setup

1. Clone and create your branch:
   ```bash
   git clone https://github.com/sreeram2001/SemanticWebMining_Project.git
   git checkout chandana-data-pipeline  # or create your own branch
   ```

2. Set Colab secrets (Settings → Secrets):
   - `KAGGLE_USERNAME` / `KAGGLE_KEY` — for dataset downloads
   - `ASU_VOYAGER_KEY` — API key from https://openai.rc.asu.edu
   - `GITHUB_TOKEN` — personal access token for pushing

3. Run the notebook cells in order. Datasets download automatically via Kaggle API.

4. Tests (no API key needed; `pip install pandas scikit-learn`, plus `gradio` for the app test):
   ```bash
   cd CODE && python -m unittest discover tests
   ```

5. Demo offline: `python CODE/app.py --sample` and open http://127.0.0.1:7860
   (on Colab: last section of `CODE/notebooks/Run_Pipeline_Colab.ipynb`, `--share` for a public link).

## Key Details

- **Model**: `qwen3-235b-a22b` on ASU Voyager (free, OpenAI-compatible API)
- **Datasets**: LinkedIn Job Postings (Kaggle, CC-BY-SA-4.0) + Resume Dataset (Kaggle, CC0-1.0)
- **Extraction**: Ontology-guided prompts → structured JSON with source spans; ungrounded or ill-typed facts are rejected and counted
- **Skill Linking**: alias table → character 3-gram + SBERT candidates → LLM picks a Lightcast ID or NIL
- **Ranking**: RRF(BM25, SBERT) → cross-encoder on top 50 → S = β1·CE + β2·G − β3·P; graph ablation sets β2 = 0
- **Chat**: answers only from stored, span-grounded facts; LLM answers are kept only if they pass verification
