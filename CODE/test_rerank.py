import unittest

import rerank as rr


def fake_predict(pairs):
    return [len(set(q.split()) & set(d.split())) for q, d in pairs]


class MinMaxTests(unittest.TestCase):
    def test_scales_to_unit_interval(self):
        self.assertEqual(rr.minmax([2, 4, 6]), [0.0, 0.5, 1.0])

    def test_handles_negative_scores(self):
        out = rr.minmax([-10, 0, 10])
        self.assertEqual(out, [0.0, 0.5, 1.0])

    def test_constant_list_is_zeros(self):
        self.assertEqual(rr.minmax([3, 3, 3]), [0.0, 0.0, 0.0])

    def test_empty_and_single(self):
        self.assertEqual(rr.minmax([]), [])
        self.assertEqual(rr.minmax([5]), [0.0])


class RerankerTests(unittest.TestCase):
    def setUp(self):
        self.r = rr.CrossEncoderReranker(predict=fake_predict)
        self.cands = [
            ("j1", "nurse hospital"),
            ("j2", "python sql data"),
            ("j3", "python data"),
        ]

    def test_orders_by_score(self):
        out = self.r.rerank("python sql data engineer", self.cands)
        self.assertEqual([c[0] for c in out], ["j2", "j3", "j1"])

    def test_normalized_scores_span_zero_to_one(self):
        out = self.r.rerank("python sql data engineer", self.cands)
        self.assertEqual(out[0][2], 1.0)
        self.assertEqual(out[-1][2], 0.0)
        self.assertEqual(out[0][1], 3.0)  # raw score kept

    def test_top_n_drops_the_tail(self):
        out = self.r.rerank("python sql data", self.cands, top_n=2)
        self.assertEqual({c[0] for c in out}, {"j1", "j2"})

    def test_ties_keep_incoming_order(self):
        out = self.r.rerank("zzz", self.cands)
        self.assertEqual([c[0] for c in out], ["j1", "j2", "j3"])

    def test_empty_candidates(self):
        self.assertEqual(self.r.rerank("q", []), [])


class RerankRunTests(unittest.TestCase):
    def test_runs_per_resume_and_skips_missing_jobs(self):
        run = {"r1": [("j1", 0.9), ("j2", 0.8), ("gone", 0.7)], "r2": [("j2", 0.5)]}
        queries = {"r1": "nurse care", "r2": "python"}
        texts = {"j1": "nurse hospital care", "j2": "python developer"}
        out = rr.rerank_run(
            run, queries, texts, rr.CrossEncoderReranker(predict=fake_predict)
        )
        self.assertEqual([c[0] for c in out["r1"]], ["j1", "j2"])
        self.assertEqual([c[0] for c in out["r2"]], ["j2"])


if __name__ == "__main__":
    unittest.main()
