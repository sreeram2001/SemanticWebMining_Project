# Automatic-label evaluation

60 resumes. A job counts as relevant when its title matches the resume's category (`CATEGORY_TITLES` in `auto_eval.py`). These are silver labels: they test whether a ranker finds jobs in the right field, not how good the fit is.

| Method | P@10 | nDCG@10 |
|---|---|---|
| bm25 | 0.310 | 0.311 |
| dense | 0.487 | 0.492 |
| hybrid | 0.415 | 0.422 |
| rrf | 0.413 | 0.410 |
| rerank | 0.533 | 0.541 |

## nDCG@10 by resume category

| Category | relevant jobs | bm25 | dense | hybrid | rrf | rerank |
|---|---|---|---|---|---|---|
| ACCOUNTANT | 3,361 | 0.644 | 0.965 | 0.919 | 0.825 | 0.883 |
| BANKING | 1,586 | 0.454 | 0.537 | 0.462 | 0.526 | 0.551 |
| BUSINESS-DEVELOPMENT | 992 | 0.015 | 0.411 | 0.152 | 0.192 | 0.423 |
| CONSULTANT | 2,941 | 0.041 | 0.405 | 0.078 | 0.182 | 0.460 |
| DESIGNER | 1,140 | 0.602 | 0.809 | 0.700 | 0.688 | 0.916 |
| DIGITAL-MEDIA | 651 | 0.223 | 0.326 | 0.441 | 0.394 | 0.362 |
| ENGINEERING | 2,770 | 0.223 | 0.405 | 0.476 | 0.411 | 0.506 |
| FINANCE | 1,831 | 0.352 | 0.261 | 0.243 | 0.263 | 0.394 |
| HEALTHCARE | 13,943 | 0.157 | 0.285 | 0.163 | 0.193 | 0.386 |
| HR | 1,598 | 0.435 | 0.545 | 0.683 | 0.515 | 0.709 |
| INFORMATION-TECHNOLOGY | 8,361 | 0.329 | 0.394 | 0.311 | 0.360 | 0.414 |
| SALES | 9,864 | 0.257 | 0.566 | 0.441 | 0.373 | 0.483 |
