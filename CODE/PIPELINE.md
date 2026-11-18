# End-to-end pipeline (Steps 4b–9)

Everything after the Lightcast download, built on the team's existing
`extract/extraction.py`, `extract/evaluation.py`, `linking/skill_linking.py` and `ontology.json`. One
command per stage, plus a chat demo (`app.py`).

```
DATA/processed/*.csv ─▶ split ─▶ extract (Voyager LLM, grounded) ─▶ entity linking ─▶ knowledge graph ─▶ Neo4j
                                                                                        │
             evaluate ◀─ S = β1·CE + β2·G − β3·P ◀─ cross-encoder ◀─ RRF(BM25, SBERT) ◀──┘
                                    │
                                    └─▶ chat agent (app.py / chat): ranked jobs, verified explanations, skill gaps
```

## Status

| | Done | Still to do |
|---|---|---|
| Code | every stage incl. chat demo, 107 tests, offline sample run, profiled at full scale | — |
| Real run | — | Colab: `CODE/notebooks/Run_Pipeline_Colab.ipynb` (preprocess → split → Lightcast → extract → build → rank) |
| Real models | code paths tested with stand-in libraries; `--require-models` stops a run instead of falling back | confirm on GPU: the `Backends:` line must name the real models; load Neo4j once |
| Labels | sheet generation, merging, agreement | the team labels: 60 + 60 extraction docs, 300 linking mentions, 30 graded resumes |
| Tuning | `tune-linking`, `tune` (dev split only) | run after labeling |

## Offline demo (no keys, about 10 seconds)

```bash
pip install pandas scikit-learn
python CODE/run_pipeline.py --sample all
cd CODE && python -m unittest discover tests
```

Chat demo on the sample (`pip install gradio`, then open http://127.0.0.1:7860):

```bash
python CODE/app.py --sample               # web UI
python CODE/run_pipeline.py --sample chat # same agent in the terminal
```

`--sample` uses `CODE/sample_data/`: 26 synthetic postings, 20 resumes in Kaggle
format, a 108-skill taxonomy in the Lightcast cache format (IDs `SAMPLE###`,
not real Lightcast IDs) and a hand-labeled extraction gold set (3 + 3 docs,
`sample_data/gold*/`), with rule-based extraction and stand-in models.
**Show it as a pipeline demo, not as results: the numbers mean nothing.**

## Real run

Use `CODE/notebooks/Run_Pipeline_Colab.ipynb` on a GPU runtime. It keeps `DATA/` on
Google Drive and runs these commands (add `--require-models` to every model
step):

| Step | Command | Output |
|---|---|---|
| Clean | `preprocess` | `DATA/processed/*_cleaned.csv` |
| Split | `split` | `EVALUATIONS/split.json`, frozen (never overwritten) |
| Extract | `extract --limit 50`, then `extract` | `DATA/extracted/*.jsonl` through `extraction.Extractor`; resumable |
| Link + graph | `build --cypher EVALUATIONS/runs/graph.cypher [--neo4j]` | graph stats, integrity check, linker backend, LLM link calls |
| Rank | `rank [--resume ID] [--method rrf]` | top jobs, verified explanation, skill gaps |
| Label sheets | `sheets [--task extraction\|linking\|relevance]` | `EVALUATIONS/gold/`, `EVALUATIONS/labeling/*_<name>.csv` |
| Merge labels | `merge-labels` | `EVALUATIONS/linking/linking_gold.csv`, `EVALUATIONS/relevance/graded_qrels.csv`, agreement (κ) |
| Tune linking | `tune-linking` | `min_score` and the LLM band on the dev half of the linking labels |
| Tune ranking | `tune` | `EVALUATIONS/runs/tuned_scoring.json` (dev split only) |
| Evaluate | `evaluate --split dev --tuned` | `EVALUATIONS/runs/eval_dev_tuned.{json,md}` |
| Phase 5 | `evaluate --split test --tuned` | once per reported configuration |
| Extraction eval | `eval-extraction [--models a,b,c]` | `EVALUATIONS/runs/eval_extraction.{json,md}`: per-field P/R/F1, grounded vs. ungrounded, agreement |
| Demo | `python CODE/app.py [--share]` or `chat` | Gradio chat / terminal chat |

All commands are `python CODE/run_pipeline.py <step>`. Settings are in
`CODE/config.json`; `--config file.json` overrides any part.

## Labeling (shared by all six)

`sheets` writes one CSV per person (names in `config.json → labeling.annotators`):

| Set | Size | Unit | Shared by everyone (agreement) | Each person |
|---|---|---|---|---|
| Extraction | 60 resumes + 60 postings | document (JSON in `EVALUATIONS/gold/`) | 10 + 10 copies in `gold_second/`, each given to a second person | ~20 files + ~3 second copies (see `gold/assignment.json`) |
| Skill linking | 300 mentions | mention | 30 | ~45 + 30 |
| Relevance | 30 resumes (15 dev, 15 test) | resume with its pooled jobs | 3 resumes | ~4.5 resumes + 3 |

Fill `gold_skill_id` with one Lightcast ID from `candidates` (or another ID
from the taxonomy) or `NIL`; fill `grade` with 0 (not relevant), 1 (partly), 2
(relevant). Extraction files follow the notebook's Step 3c instructions.
`merge-labels` resolves shared items by majority vote and reports pairwise
Cohen's κ (linear-weighted for grades).

## Skill linking (as proposed)

alias hit → candidates from character 3-grams + SBERT bi-encoder over
"name: description" → the LLM picks one candidate or NIL when the top score is
in the uncertain band [`llm_reject`, `llm_accept`) → otherwise accept / NIL.
Each distinct mention is asked once and answers are cached in
`DATA/lightcast/link_llm_cache.json`; skill embeddings are cached in
`DATA/lightcast/skill_embeddings.npy`. `tune-linking` tunes `min_score` and the
band on the dev half; report linking accuracy on the test half.

## Rules the code enforces (from the presentation)

- A fact enters the graph only through `extraction.ground` (span found + ontology type checks).
- Skills link to a Lightcast ID or NIL. NIL skills get a generated ID with `source: "extracted"` (per the ontology) and are **unresolved**: shown, not scored.
- **Missed skills (slide 9):** a required skill not among the extracted facts is searched for in the raw resume (Lightcast name and aliases, exact or fuzzy ≥ 0.9). If found: full credit, logged as an extraction miss. If not: `not_found`, with no penalty; only its G credit is withheld.
- **P fires only on explicit conflicts:** degree below the required level, or fewer years than required. No date ranges means `unknown` and no penalty.
- The first-stage query is the resume's linked skills (slide 6); `retrieval.query: "text"` uses the full resume.
- Every ranking number is per resume with a 95% bootstrap CI; every comparison with a baseline, and the β₂ = 0 ablation, has a paired randomization p-value. Every run logs backends, config, data hash and seed.

## Robustness

- `config.json` is validated before a run (λ order, weight signs, modes, `current_date` format, typos in section names).
- Missing or malformed inputs stop with a one-line `error:` saying which step to run. `--debug` shows the traceback.
- Every result and cache file is written atomically (temp file + rename), so a crash or Colab disconnect never leaves a half-written file. A corrupt cache is moved to `*.corrupt` and rebuilt.
- LLM linking answers are saved every 25 calls and when a build fails. Transient API errors (429, 5xx, timeouts) are retried with backoff.
- Facts whose quoted span no longer matches the text (the CSV was re-cleaned after extraction) are dropped and counted (`stale_facts_dropped`), not loaded with wrong provenance.
- IDs are normalized (`3884428798.0` → `3884428798`), and rows without an ID or text are skipped.
- Resumes added after the split was frozen are reported and left out of tuning and evaluation.
- Neo4j loads merge nodes on `key` and edges on `eid`, so loading twice does not duplicate anything.
- Posting embeddings are cached in `DATA/embeddings/postings.npz` (`retrieval.emb_cache`), keyed by model and posting text; first-stage top-k uses `argpartition`; retrieval and the cross-encoder see the job title twice before the description (all three from `sreeram-retrieval-ranking`).

## Performance

Measured on a synthetic corpus at the real run's size (2,000 postings,
1,400 resumes, 33,000 skills; rule extraction and CPU stand-in models), before
and after optimizing. Per-resume metrics were identical for all 700 dev resumes
and all six methods.

| Stage | Before | After |
|---|---|---|
| Tune 32 scoring configurations (100 resumes) | 495 s | 8–11 s |
| Evaluate 700 dev resumes × 6 methods | 246 s | 82–87 s |
| RELATED_TO edges | 622K | 464K (only pairs that can earn partial credit) |

What changed: the scoring-independent part of each resume–job match is
cached, so tuning and the ablation only redo arithmetic. The missed-skill text
search is memoized and prefiltered with a character-trigram bound that can
never drop a real match (checked against brute force on 19,200 cases). The
first-stage scores are kept in a bounded cache. On the real run, the GPU
cross-encoder and the Voyager calls dominate the run time.

## Assumptions and limitations to state in the report

- **PII masking** covers emails, phone numbers, URLs and street addresses, not personal names. The proposal mentions names, but the Kaggle resumes are already anonymized ("Company Name", "City , State") and carry no candidate names.
- **Years of experience** are summed from the resume's date ranges, with overlaps merged and months counted inclusively. "Current" / "Present" is read as `scoring.current_date` = `2021-12`, the approximate collection date of the resume dataset. Change it in `config.json` if the team picks another date, and state it in the report.
- **Working corpus**: `corpus.n_postings` = 2,000 postings sampled by experience level, not all 124K, because each posting needs an LLM extraction call.
- **Tier-1 relevance** comes from a category-to-title regex (`ranking_eval.CATEGORY_TITLES`): coarse, binary labels.
- **Partial credit** uses Lightcast subcategory / category only; the λ values are a heuristic, tested by the ablation.
- **Linking cache** is keyed by mention text, so the same string in two different contexts gets one decision.
- **Skill gaps** come from the extracted and linked skills plus the raw-text check. A skill missed by both counts as `not_found`.

## Before merging this branch

1. **One skill linker**: `linking/entity_linking.py` builds on `linking/skill_linking.py` (it imports `build_name_index`, `normalize_name` and `load_skills`), so it doesn't replace Kaushal's Lightcast step 4a. What it supersedes is the notebook's older Step 4 cell, which linked to LinkedIn's 35 categories. Its alias table maps to categories, not skills, so it can't be merged as is. `resources/skill_aliases.json` is the one alias table; skill-level aliases from that cell can be moved into it.
2. **Division of work**: the modules were written in one pass on this branch. The stage assignments in the proposal (Table IV) still apply for validating, tuning and reporting each stage. Agree as a team how authorship is presented before submitting.

## Files

| File | Stage |
|---|---|
| `dataprep/preprocess.py` | Step 1 as a script, frozen split, corpus sample |
| `extract/rule_extractor.py` | offline extractor with `Extractor`'s record format (tests, baseline) |
| `linking/entity_linking.py`, `resources/skill_aliases.json` | entity linking, `Gazetteer`, `TextSearch` (missed skills) |
| `kg/graph.py` | graph per `ontology.json`, integrity checks, Neo4j loader, Cypher export, `GRAPH_SCORE_CYPHER` |
| `rank/retrieval.py` | BM25, SBERT, TF-IDF, Proposal 3's weighted hybrid, RRF |
| `rank/ranking.py` | cross-encoder rerank, G, P, missed-skill recovery, ablation |
| `assistant/explanations.py` | grounded explanations, verification, skill gaps |
| `rank/ranking_eval.py` | two-tier labels, metrics, bootstrap CIs, randomization tests, tuning |
| `dataprep/labeling.py` | labeling sheets, merging, agreement |
| `core/pipeline.py`, `run_pipeline.py`, `config.*` | wiring and CLI |
| `notebooks/Run_Pipeline_Colab.ipynb` | the real run on a GPU runtime |
| `assistant/chat.py`, `app.py` | chat agent (intents, verified free-form answers, uploaded resumes) and Gradio web demo |
| `tests/test_end_to_end.py`, `tests/test_model_paths.py`, `tests/test_chat.py` | every stage, a full offline run, robustness, model/Neo4j/LLM call paths, chat and app |

## Chat agent

`chat.CareerAgent` answers from stored facts only: `jobs`/`top N`, `why N`,
`gaps`, `skills`, `job N`, `compare` (graph ablation for this resume) and
`method <name>`. A pasted or uploaded resume is PII-masked, extracted (LLM if
`ASU_VOYAGER_KEY` is set, else rules), linked and added to the graph. Other
questions go to the LLM with the top jobs' facts; the answer is shown only if
`explanations.verify` finds no skill outside those facts.
