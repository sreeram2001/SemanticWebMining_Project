## Evaluation on the dev split

Backends: {'dense': 'tfidf-lsa-25 (stand-in)', 'reranker': 'lexical (stand-in)', 'linker': 'char3gram+threshold', 'query': 'skills'}
Data hash: d86e0c49ba72

### Ranking, tier 1 (auto labels) (10 resumes)

| Method | P@10 [95% CI] | nDCG@10 [95% CI] |
|---|---|---|
| bm25 | 0.260 [0.160, 0.390] | 0.943 [0.845, 1.000] |
| dense | 0.210 [0.160, 0.260] | 0.882 [0.756, 0.984] |
| weighted_hybrid | 0.210 [0.160, 0.260] | 0.882 [0.756, 0.984] |
| rrf | 0.230 [0.160, 0.310] | 0.910 [0.794, 1.000] |
| full | 0.270 [0.180, 0.390] | 0.974 [0.949, 1.000] |
| full_no_graph | 0.220 [0.160, 0.290] | 0.888 [0.768, 0.986] |

| Comparison | Metric | Δ [95% CI] | p (randomization) | wins/losses |
|---|---|---|---|---|
| full vs bm25 | nDCG@10 | 0.032 [-0.016, 0.115] | 0.4988 | 2/1 |
| full vs bm25 | P@10 | 0.010 [0.000, 0.040] | 1.0000 | 1/0 |
| full vs dense | nDCG@10 | 0.093 [0.000, 0.195] | 0.2414 | 3/0 |
| full vs dense | P@10 | 0.060 [0.000, 0.130] | 0.2414 | 3/0 |
| full vs weighted_hybrid | nDCG@10 | 0.093 [0.000, 0.195] | 0.2414 | 3/0 |
| full vs weighted_hybrid | P@10 | 0.060 [0.000, 0.130] | 0.2414 | 3/0 |
| full vs rrf | nDCG@10 | 0.064 [-0.008, 0.161] | 0.2414 | 3/1 |
| full vs rrf | P@10 | 0.040 [0.000, 0.080] | 0.2414 | 3/0 |
| full vs full_no_graph | nDCG@10 | 0.087 [0.009, 0.186] | 0.1199 | 4/0 |
| full vs full_no_graph | P@10 | 0.050 [0.000, 0.110] | 0.2414 | 3/0 |

### Ranking, tier 2 (graded) (10 resumes)

| Method | P@10 [95% CI] | nDCG@10 [95% CI] |
|---|---|---|
| bm25 | 0.170 [0.140, 0.200] | 0.962 [0.910, 1.000] |
| dense | 0.170 [0.140, 0.200] | 0.959 [0.907, 1.000] |
| weighted_hybrid | 0.170 [0.140, 0.200] | 0.959 [0.907, 1.000] |
| rrf | 0.170 [0.140, 0.200] | 0.962 [0.910, 1.000] |
| full | 0.180 [0.150, 0.200] | 0.993 [0.982, 1.000] |
| full_no_graph | 0.170 [0.140, 0.200] | 0.977 [0.930, 1.000] |

| Comparison | Metric | Δ [95% CI] | p (randomization) | wins/losses |
|---|---|---|---|---|
| full vs bm25 | nDCG@10 | 0.030 [-0.007, 0.075] | 0.5047 | 2/1 |
| full vs bm25 | P@10 | 0.010 [0.000, 0.040] | 1.0000 | 1/0 |
| full vs dense | nDCG@10 | 0.034 [0.000, 0.075] | 0.5047 | 2/0 |
| full vs dense | P@10 | 0.010 [0.000, 0.040] | 1.0000 | 1/0 |
| full vs weighted_hybrid | nDCG@10 | 0.034 [0.000, 0.075] | 0.5047 | 2/0 |
| full vs weighted_hybrid | P@10 | 0.010 [0.000, 0.040] | 1.0000 | 1/0 |
| full vs rrf | nDCG@10 | 0.030 [-0.007, 0.075] | 0.5047 | 2/1 |
| full vs rrf | P@10 | 0.010 [0.000, 0.040] | 1.0000 | 1/0 |
| full vs full_no_graph | nDCG@10 | 0.016 [0.000, 0.055] | 0.4983 | 2/0 |
| full vs full_no_graph | P@10 | 0.010 [0.000, 0.040] | 1.0000 | 1/0 |

Extraction misses recovered from raw resume text: 1

### Skill linking
| Linker | Accuracy@1 | NIL accuracy | n |
|---|---|---|---|
| alias only (baseline) | 0.700 | 1.000 | 20 |
| entity linker (char3gram+threshold) | 0.950 [0.800, 1.000] | 1.000 | 20 |

Paired randomization p = 0.0545

