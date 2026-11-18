DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_N = 50


def minmax(scores):
    scores = [float(s) for s in scores]
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


class CrossEncoderReranker:
    def __init__(
        self, model_name=DEFAULT_MODEL, max_length=512, batch_size=32, predict=None
    ):
        self.model_name, self.max_length = model_name, max_length
        self.batch_size = batch_size
        self._predict = predict

    def predict(self, pairs):
        if self._predict is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self.model_name, max_length=self.max_length)
            self._predict = lambda p: model.predict(
                p, batch_size=self.batch_size, show_progress_bar=False
            )
        return [float(s) for s in self._predict(pairs)]

    def rerank(self, query, candidates, top_n=DEFAULT_TOP_N):
        head = list(candidates)[:top_n]
        if not head:
            return []
        raw = self.predict([(query, text) for _, text in head])
        norm = minmax(raw)
        order = sorted(range(len(head)), key=lambda i: (-raw[i], i))
        return [(head[i][0], raw[i], norm[i]) for i in order]


def rerank_run(run, queries, job_texts, reranker, top_n=DEFAULT_TOP_N):
    out = {}
    for rid, ranking in run.items():
        cands = [
            (jid, job_texts[jid]) for jid, *_ in ranking[:top_n] if jid in job_texts
        ]
        out[rid] = reranker.rerank(queries[rid], cands, top_n=top_n)
    return out
