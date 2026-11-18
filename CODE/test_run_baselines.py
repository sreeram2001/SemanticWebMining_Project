import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import retrieval as rt
import run_baselines as rb

JOBS = {
    "j1": "python developer django sql",
    "j2": "nurse hospital patient care",
    "j3": "data engineer python spark sql",
    "j4": "accountant tax audit",
}

EMB = {"j1": [1, 0], "j2": [0, 1], "j3": [0.9, 0.1], "j4": [0.1, 0.9]}


def fake_encode(texts):
    return np.array(
        [[1.0, 0.0] if "python" in t else [0.0, 1.0] for t in texts], dtype="float32"
    )


def build():
    ids = list(JOBS)
    texts = list(JOBS.values())
    return (
        rt.BM25Index(ids, texts),
        rt.DenseIndex(ids, [EMB[i] for i in ids]),
        rt.TfidfIndex(ids, texts),
    )


class RunRankersTests(unittest.TestCase):
    def test_all_methods_rank_relevant_job_first(self):
        bm25, dense, tfidf = build()
        runs = rb.run_rankers(
            {"r1": "python sql developer"}, bm25, dense, tfidf, fake_encode, k=3
        )
        self.assertEqual(set(runs), {"bm25", "dense", "hybrid", "rrf"})
        for method in runs:
            self.assertIn(runs[method]["r1"][0][0], {"j1", "j3"}, method)
            self.assertLessEqual(len(runs[method]["r1"]), 3)

    def test_rrf_is_fusion_of_bm25_and_dense(self):
        bm25, dense, tfidf = build()
        runs = rb.run_rankers(
            {"r1": "python sql"}, bm25, dense, tfidf, fake_encode, k=4
        )
        expected = rt.rrf([runs["bm25"]["r1"], runs["dense"]["r1"]], top=4)
        self.assertEqual(runs["rrf"]["r1"], expected)

    def test_mismatched_job_order_rejected(self):
        bm25, dense, tfidf = build()
        dense.ids = list(reversed(dense.ids))
        with self.assertRaises(ValueError):
            rb.run_rankers({"r1": "x"}, bm25, dense, tfidf, fake_encode)


class SelectResumesTests(unittest.TestCase):
    DF = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5],
            "Category": ["IT", "IT", "IT", "HR", "HR"],
            "Resume_clean": list("abcde"),
        }
    )

    def test_stratified_cap_per_category(self):
        out = rb.select_resumes(self.DF, per_category=2)
        counts = out["Category"].value_counts().to_dict()
        self.assertEqual(counts, {"IT": 2, "HR": 2})

    def test_small_category_not_oversampled(self):
        out = rb.select_resumes(self.DF, per_category=10)
        self.assertEqual(len(out), 5)

    def test_explicit_ids(self):
        out = rb.select_resumes(self.DF, ids=["2", "5"])
        self.assertEqual(sorted(out["ID"]), [2, 5])

    def test_same_seed_same_sample(self):
        a = rb.select_resumes(self.DF, 1, seed=1)["ID"].tolist()
        b = rb.select_resumes(self.DF, 1, seed=1)["ID"].tolist()
        self.assertEqual(a, b)


class DenseCacheTests(unittest.TestCase):
    def test_cache_reused_when_ids_match(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "e.npz")
            rt.DenseIndex(["a", "b"], [[1, 0], [0, 1]]).save(path)
            idx = rb.build_dense_index(["a", "b"], ["x", "y"], path, "unused-model")
            self.assertEqual(idx.ids, ["a", "b"])


if __name__ == "__main__":
    unittest.main()


class CliTests(unittest.TestCase):
    def test_help_does_not_crash(self):
        with self.assertRaises(SystemExit) as cm:
            rb.main(["--help"])
        self.assertEqual(cm.exception.code, 0)


class PriorRunTests(unittest.TestCase):
    def test_hybrid_uses_prior_indexes(self):
        bm25, dense, tfidf = build()
        ids = list(JOBS)
        prior = {"skills": ["python"], "tfidf": rt.prior_tfidf(ids, list(JOBS.values())),
                 "dense": rt.DenseIndex(ids, [EMB[i] for i in ids])}
        runs = rb.run_rankers({"r1": "python sql"}, bm25, dense, tfidf, fake_encode, k=2, prior=prior)
        self.assertIn(runs["hybrid"]["r1"][0][0], {"j1", "j3"})

    def test_prior_order_mismatch(self):
        bm25, dense, tfidf = build()
        prior = {"skills": [], "tfidf": tfidf, "dense": rt.DenseIndex(["x"] * 4, [[1, 0]] * 4)}
        with self.assertRaises(ValueError):
            rb.run_rankers({"r1": "x"}, bm25, dense, tfidf, fake_encode, prior=prior)

    def test_split_ids_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            with open(path, "w") as f:
                f.write('{"docs": {"resume:1": "dev", "posting:9": "dev", "resume:2": "test"}}')
            self.assertEqual(rb.split_resume_ids(path), ["1", "2"])
            self.assertEqual(len(rb.file_hash(path)), 12)
