## Evaluation on the test split

Backends: {'dense': 'tfidf-lsa-25 (stand-in)', 'reranker': 'lexical (stand-in)', 'linker': 'char3gram+threshold', 'query': 'skills'}
Data hash: d86e0c49ba72

### Ranking, tier 1 (auto labels) (10 resumes)

| Method | P@10 [95% CI] | nDCG@10 [95% CI] |
|---|---|---|
| bm25 | 0.290 [0.190, 0.400] | 0.941 [0.859, 1.000] |
| dense | 0.230 [0.180, 0.300] | 0.866 [0.740, 0.988] |
| weighted_hybrid | 0.230 [0.180, 0.300] | 0.866 [0.740, 0.988] |
| rrf | 0.280 [0.180, 0.400] | 0.927 [0.827, 0.995] |
| full | 0.310 [0.190, 0.430] | 0.980 [0.942, 1.000] |
| full_no_graph | 0.240 [0.180, 0.320] | 0.901 [0.780, 1.000] |

| Comparison | Metric | Δ [95% CI] | p (randomization) | wins/losses |
|---|---|---|---|---|
| full vs bm25 | nDCG@10 | 0.039 [-0.001, 0.091] | 0.2494 | 3/1 |
| full vs bm25 | P@10 | 0.020 [0.000, 0.060] | 1.0000 | 1/0 |
| full vs dense | nDCG@10 | 0.114 [0.012, 0.216] | 0.1219 | 4/0 |
| full vs dense | P@10 | 0.080 [0.000, 0.160] | 0.2499 | 3/0 |
| full vs weighted_hybrid | nDCG@10 | 0.114 [0.012, 0.216] | 0.1219 | 4/0 |
| full vs weighted_hybrid | P@10 | 0.080 [0.000, 0.160] | 0.2499 | 3/0 |
| full vs rrf | nDCG@10 | 0.053 [0.004, 0.115] | 0.1219 | 4/0 |
| full vs rrf | P@10 | 0.030 [0.000, 0.090] | 1.0000 | 1/0 |
| full vs full_no_graph | nDCG@10 | 0.080 [0.000, 0.164] | 0.2499 | 3/0 |
| full vs full_no_graph | P@10 | 0.070 [0.000, 0.150] | 0.2499 | 3/0 |

### Ranking, tier 2 (graded) (10 resumes)

| Method | P@10 [95% CI] | nDCG@10 [95% CI] |
|---|---|---|
| bm25 | 0.180 [0.160, 0.200] | 0.987 [0.966, 1.000] |
| dense | 0.170 [0.140, 0.200] | 0.977 [0.942, 1.000] |
| weighted_hybrid | 0.170 [0.140, 0.200] | 0.977 [0.942, 1.000] |
| rrf | 0.180 [0.160, 0.200] | 0.984 [0.961, 1.000] |
| full | 0.180 [0.160, 0.200] | 1.000 [1.000, 1.000] |
| full_no_graph | 0.180 [0.160, 0.200] | 0.994 [0.983, 1.000] |

| Comparison | Metric | Δ [95% CI] | p (randomization) | wins/losses |
|---|---|---|---|---|
| full vs bm25 | nDCG@10 | 0.013 [0.000, 0.034] | 0.4978 | 2/0 |
| full vs bm25 | P@10 | 0.000 [0.000, 0.000] | 1.0000 | 0/0 |
| full vs dense | nDCG@10 | 0.023 [0.000, 0.058] | 0.4978 | 2/0 |
| full vs dense | P@10 | 0.010 [0.000, 0.030] | 1.0000 | 1/0 |
| full vs weighted_hybrid | nDCG@10 | 0.023 [0.000, 0.058] | 0.4978 | 2/0 |
| full vs weighted_hybrid | P@10 | 0.010 [0.000, 0.030] | 1.0000 | 1/0 |
| full vs rrf | nDCG@10 | 0.016 [0.000, 0.039] | 0.4978 | 2/0 |
| full vs rrf | P@10 | 0.000 [0.000, 0.000] | 1.0000 | 0/0 |
| full vs full_no_graph | nDCG@10 | 0.006 [0.000, 0.017] | 1.0000 | 1/0 |
| full vs full_no_graph | P@10 | 0.000 [0.000, 0.000] | 1.0000 | 0/0 |

Extraction misses recovered from raw resume text: 1

### Skill linking
| Linker | Accuracy@1 | NIL accuracy | n |
|---|---|---|---|
| alias only (baseline) | 0.700 | 1.000 | 20 |
| entity linker (char3gram+threshold) | 0.950 [0.800, 1.000] | 1.000 | 20 |

Paired randomization p = 0.0545

