import math
import os
import tempfile
import unittest

import retrieval as rt

DOCS = {
    "j1": "Senior Python developer building REST APIs with Django and PostgreSQL",
    "j2": "Registered nurse for a busy hospital ward, patient care",
    "j3": "Data engineer: Python, Spark, SQL pipelines on AWS",
    "j4": "Accountant preparing tax returns and financial statements",
}


class TokenizeTests(unittest.TestCase):
    def test_keeps_tech_terms(self):
        self.assertEqual(
            rt.tokenize("C++ and C# with Node.js, Python."),
            ["c++", "c#", "node.js", "python"],
        )

    def test_empty_and_none(self):
        self.assertEqual(rt.tokenize(None), [])
        self.assertEqual(rt.tokenize(""), [])


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.idx = rt.BM25Index(list(DOCS), list(DOCS.values()))

    def test_relevant_doc_first(self):
        top = self.idx.search("python postgresql django")
        self.assertEqual(top[0][0], "j1")

    def test_only_matching_docs_returned(self):
        ids = [d for d, _ in self.idx.search("python")]
        self.assertEqual(set(ids), {"j1", "j3"})

    def test_unknown_terms_return_nothing(self):
        self.assertEqual(self.idx.search("zzzz qqqq"), [])

    def test_rarer_term_scores_higher(self):
        self.assertGreater(self.idx.idf["nurse"], self.idx.idf["python"])

    def test_score_matches_formula(self):
        idx = rt.BM25Index(["a", "b"], ["python python", "java"], k1=1.5, b=0.75)
        n, df = 2, 1
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        avg = 1.5
        norm = 1 - 0.75 + 0.75 * 2 / avg
        expected = idf * 2 * 2.5 / (2 + 1.5 * norm)
        self.assertAlmostEqual(idx.search("python")[0][1], expected)

    def test_k_limits_results(self):
        self.assertEqual(len(self.idx.search("python", k=1)), 1)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            rt.BM25Index(["a"], ["x", "y"])

    def test_empty_corpus(self):
        self.assertEqual(rt.BM25Index([], []).search("python"), [])


class DenseTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np
        emb = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype="float32")
        self.idx = rt.DenseIndex(["a", "b", "c"], emb)

    def test_search_order(self):
        res = self.idx.search([1, 0], k=3)
        self.assertEqual([d for d, _ in res], ["a", "c", "b"])
        self.assertAlmostEqual(res[1][1], 0.6, places=5)

    def test_k_larger_than_corpus(self):
        self.assertEqual(len(self.idx.search([1, 0], k=50)), 3)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "emb.npz")
            self.idx.save(path)
            back = rt.DenseIndex.load(path)
        self.assertEqual(back.ids, ["a", "b", "c"])
        self.assertTrue(self.np.allclose(back.emb, self.idx.emb))

    def test_encode_texts_uses_given_model(self):
        class Fake:
            def encode(self, texts, **kw):
                import numpy as np

                return np.ones((len(texts), 3))

        out = rt.encode_texts(["x", "y"], model=Fake())
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(str(out.dtype), "float32")


class HybridTests(unittest.TestCase):
    def test_equal_weight(self):
        res = rt.weighted_hybrid([1.0, 0.0], [0.0, 0.6], ["a", "b"])
        self.assertEqual([d for d, _ in res], ["a", "b"])
        self.assertAlmostEqual(res[0][1], 0.5)
        self.assertAlmostEqual(res[1][1], 0.3)

    def test_weight_shifts_order(self):
        res = rt.weighted_hybrid([1.0, 0.0], [0.0, 0.6], ["a", "b"], w_tfidf=0.1)
        self.assertEqual(res[0][0], "b")

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            rt.weighted_hybrid([1.0], [1.0, 2.0], ["a", "b"])

    def test_tfidf_index(self):
        idx = rt.TfidfIndex(list(DOCS), list(DOCS.values()))
        sims = idx.similarities("python developer")
        self.assertEqual(max(range(len(sims)), key=sims.__getitem__), 0)


class RRFTests(unittest.TestCase):
    def test_formula(self):
        res = dict(rt.rrf([["a", "b"], ["b", "a"]], k=60))
        self.assertAlmostEqual(res["a"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(res["a"], res["b"])

    def test_agreement_beats_single_vote(self):
        res = rt.rrf([["a", "b", "c"], ["d", "b", "e"]])
        self.assertEqual(res[0][0], "b")

    def test_absent_contributes_zero(self):
        res = dict(rt.rrf([["a"], ["b"]], k=60))
        self.assertAlmostEqual(res["a"], 1 / 61)

    def test_accepts_id_score_pairs(self):
        res = rt.rrf([[("a", 9.0), ("b", 1.0)], [("a", 0.3)]])
        self.assertEqual(res[0][0], "a")

    def test_top_and_empty(self):
        self.assertEqual(len(rt.rrf([["a", "b", "c"]], top=2)), 2)
        self.assertEqual(rt.rrf([]), [])

    def test_tie_breaks_by_best_rank(self):
        res = rt.rrf([["a", "b"], ["b", "a"]])
        self.assertEqual([d for d, _ in res], ["a", "b"])


class MetricTests(unittest.TestCase):
    REL = {"a": 2, "b": 1, "c": 0, "d": 2}

    def test_precision(self):
        self.assertAlmostEqual(
            rt.precision_at_k(["a", "c", "b", "x"], self.REL, k=4), 0.5
        )

    def test_precision_short_list_not_rewarded(self):
        self.assertAlmostEqual(rt.precision_at_k(["a"], self.REL, k=10), 0.1)

    def test_precision_threshold(self):
        self.assertAlmostEqual(
            rt.precision_at_k(["a", "b"], self.REL, k=2, threshold=2), 0.5
        )

    def test_ndcg_perfect_is_one(self):
        self.assertAlmostEqual(rt.ndcg_at_k(["a", "d", "b", "c"], self.REL, k=4), 1.0)

    def test_ndcg_worse_order_is_lower(self):
        good = rt.ndcg_at_k(["a", "d", "b"], self.REL, k=3)
        bad = rt.ndcg_at_k(["c", "b", "a"], self.REL, k=3)
        self.assertLess(bad, good)

    def test_ndcg_known_value(self):
        dcg = 3 / math.log2(2) + 0 + 1 / math.log2(4)
        ideal = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(
            rt.ndcg_at_k(["a", "c", "b"], self.REL, k=3), dcg / ideal
        )

    def test_ndcg_no_relevant_is_zero(self):
        self.assertEqual(rt.ndcg_at_k(["a"], {"a": 0}, k=10), 0.0)

    def test_unjudged_counts_zero(self):
        self.assertEqual(rt.ndcg_at_k(["zzz"], self.REL, k=3), 0.0)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            rt.precision_at_k(["a"], self.REL, k=0)

    def test_evaluate_means_and_skips_unjudged(self):
        run = {"r1": ["a", "d"], "r2": ["c", "c"], "r3": ["a"]}
        qrels = {"r1": self.REL, "r2": self.REL}
        out = rt.evaluate(run, qrels, k=2)
        self.assertEqual(out["n"], 2)
        self.assertAlmostEqual(out["P@2"], (1.0 + 0.0) / 2)
        self.assertAlmostEqual(out["nDCG@2"], (1.0 + 0.0) / 2)

    def test_evaluate_empty(self):
        self.assertEqual(rt.evaluate({}, {})["n"], 0)


class DataTests(unittest.TestCase):
    def test_posting_text(self):
        t = rt.posting_text({"title": "Data Analyst", "description": "SQL"})
        self.assertIn("Data Analyst", t)
        self.assertTrue(t.endswith("SQL"))

    def test_posting_text_handles_missing(self):
        self.assertIsInstance(
            rt.posting_text({"title": None, "description": None}), str
        )


if __name__ == "__main__":
    unittest.main()


class PriorTests(unittest.TestCase):
    def test_skill_list_loads(self):
        skills = rt.load_prior_skills()
        self.assertIn("python", skills)
        self.assertEqual(len(skills), len(set(skills)))

    def test_whole_word_matching(self):
        skills = ["sql", "r", "power bi"]
        self.assertEqual(rt.prior_extract_skills("MySQL and Power BI, R code", skills),
                         ["power bi", "r"])

    def test_query_is_skills_then_clean_text(self):
        self.assertEqual(rt.prior_query("Python\n\n  dev", ["python"]), "python Python dev")

    def test_job_text_title_once(self):
        self.assertEqual(rt.prior_job_text({"title": "Dev", "description": "SQL"}), "Dev SQL")

    def test_prior_tfidf_is_plain(self):
        idx = rt.prior_tfidf(["a", "b"], ["python sql", "nurse care"])
        self.assertFalse(idx.vectorizer.sublinear_tf)
        self.assertIsNone(idx.vectorizer.max_features)
