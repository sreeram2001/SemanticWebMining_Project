# LinkedIn Assistant — Knowledge-Graph Conversational Agent

CSE 573: Semantic Web Mining | Group 5, Project 3 | Fall 2026

## What's Done (Steps 1–4)

| Step | What | Output |
|------|------|--------|
| 1 | **Data Preprocessing** | 123,782 cleaned job postings, 1,368 resumes (12 categories), PII masked |
| 2 | **Domain Ontology** | 6 node types, 7 relationship types → `CODE/ontology.json` |
| 3 | **LLM Extraction** | Structured extraction (skills, degrees, experience) using Qwen3-235B via ASU Voyager |
| 4 | **Skill Linking** | Fuzzy matching to LinkedIn's 35-category taxonomy (exact → substring → fuzzy → NIL) |

## What's Next

| Step | Owner | What to Build |
|------|-------|---------------|
| 5 | TBD | **Neo4j Knowledge Graph** — load extracted triples, skill hierarchy, integrity checks |
| 6 | TBD | **Hybrid Retrieval** — BM25 + dense embeddings, fused with RRF |
| 7 | TBD | **Reranking & Graph Scoring** — cross-encoder on top-50, graph-aware score S = β₁CE + β₂G − β₃P |
| 8 | TBD | **Chat Interface** — Gradio app with grounded explanations and skill-gap analysis |

## Project Structure
SemanticWebMining_Project/
├── CODE/
│ ├── ontology.json # Domain ontology (6 nodes, 7 relationships)
│ └── README.md # This file
├── DATA/ # Downloaded at runtime (not committed)
│ ├── linkedin-job-postings/ # ~160MB Kaggle dataset
│ ├── resume-dataset/ # ~55MB Kaggle dataset
│ └── processed/ # Cleaned CSVs
├── EVALUATIONS/
│ ├── sample_extraction.json # Sample LLM extraction output
│ └── skill_linking_demo.json # Skill linking results
└── SWM_Project_Linkedin_Assistant.ipynb

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

## Key Details

- **Model**: `qwen3-235b-a22b` on ASU Voyager (free, OpenAI-compatible API)
- **Datasets**: LinkedIn Job Postings (Kaggle, CC-BY-SA-4.0) + Resume Dataset (Kaggle, CC0-1.0)
- **Extraction**: Ontology-guided prompts → structured JSON with source spans
- **Skill Linking**: Alias table + fuzzy matching (SequenceMatcher ≥ 0.6) → LinkedIn categories
