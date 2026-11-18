"""Unit tests for pooling.py (standard library only).

    cd CODE && python -m unittest test_pooling
"""
import csv
import os
import tempfile
import unittest

import pooling as pl
import retrieval as rt

RUNS = {
    "bm25": {"r1": [("j1", 3), ("j2", 2), ("j3", 1)], "r2": [("j9", 1)]},
    "dense": {"r1": [("j2", .9), ("j4", .8), ("j5", .7)], "r2": [("j8", 1)]},
}


class PoolTests(unittest.TestCase):
    def test_union_without_duplicates(self):
        pool = pl.build_pool(RUNS, ["r1"], depth=2)
        self.assertEqual({r["job_id"] for r in pool["r1"]}, {"j1", "j2", "j4"})

    def test_methods_recorded_per_job(self):
        pool = pl.build_pool(RUNS, ["r1"], depth=2)
        by_job = {r["job_id"]: r["methods"] for r in pool["r1"]}
        self.assertEqual(sorted(by_job["j2"]), ["bm25", "dense"])
        self.assertEqual(by_job["j1"], ["bm25"])

    def test_depth_limits_each_method(self):
        pool = pl.build_pool(RUNS, ["r1"], depth=1)
        self.assertEqual({r["job_id"] for r in pool["r1"]}, {"j1", "j2"})

    def test_deterministic_shuffle(self):
        a = pl.build_pool(RUNS, ["r1"], depth=3, seed=1)
        b = pl.build_pool(RUNS, ["r1"], depth=3, seed=1)
        self.assertEqual(a, b)

    def test_method_subset(self):
        pool = pl.build_pool(RUNS, ["r1"], depth=3, methods=["dense"])
        self.assertEqual({r["job_id"] for r in pool["r1"]}, {"j2", "j4", "j5"})

    def test_resume_with_no_results(self):
        self.assertEqual(pl.build_pool(RUNS, ["nobody"])["nobody"], [])

    def test_stats(self):
        pool = pl.build_pool(RUNS, ["r1", "r2"], depth=3)
        s = pl.pool_stats(pool)
        self.assertEqual(s["resumes"], 2)
        self.assertEqual(s["pairs"], 5 + 2)


class SheetTests(unittest.TestCase):
    def setUp(self):
        self.pool = pl.build_pool(RUNS, ["r1"], depth=2)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_write_and_read_back(self):
        paths = pl.write_sheets(self.pool, {"r1": "resume text"},
                                {"j1": {"title": "Dev", "description": "x" * 1000}},
                                self.tmp.name, ["ann", "bob"], {"r1": "IT"})
        self.assertEqual(len(paths), 3)  # resumes.csv + one sheet each
        with open(paths[1], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        self.assertNotIn("methods", rows[0])           # hidden from annotators
        self.assertTrue(all(r["grade"] == "" for r in rows))
        self.assertLessEqual(max(len(r["description"]) for r in rows), pl.DESC_CHARS)

    def test_read_grades_skips_blank_and_validates(self):
        path = os.path.join(self.tmp.name, "g.csv")
        with open(path, "w", newline="") as f:
            f.write("resume_id,job_id,title,company,description,grade\n"
                    "r1,j1,a,b,c,2\nr1,j2,a,b,c,\nr1,j3,a,b,c,0\n")
        self.assertEqual(pl.read_grades(path), {("r1", "j1"): 2, ("r1", "j3"): 0})

    def test_read_grades_rejects_bad_values(self):
        for bad in ("3", "x", "-1"):
            path = os.path.join(self.tmp.name, f"bad{bad}.csv")
            with open(path, "w") as f:
                f.write(f"resume_id,job_id,grade\nr1,j1,{bad}\n")
            with self.assertRaises(ValueError):
                pl.read_grades(path)


class MergeTests(unittest.TestCase):
    def test_single_annotator(self):
        self.assertEqual(pl.merge_grades([{("r", "j"): 2}]), {"r": {"j": 2}})

    def test_mean_rounded_half_up(self):
        a, b = {("r", "j"): 2, ("r", "k"): 0}, {("r", "j"): 1, ("r", "k"): 0}
        self.assertEqual(pl.merge_grades([a, b])["r"], {"j": 2, "k": 0})  # 1.5 -> 2

    def test_union_of_pairs(self):
        out = pl.merge_grades([{("r1", "a"): 1}, {("r2", "b"): 2}])
        self.assertEqual(out, {"r1": {"a": 1}, "r2": {"b": 2}})

    def test_feeds_the_metrics(self):
        qrels = pl.merge_grades([{("r", "a"): 2, ("r", "b"): 0}])
        self.assertEqual(rt.evaluate({"r": ["a", "b"]}, qrels, k=2)["nDCG@2"], 1.0)


class KappaTests(unittest.TestCase):
    def test_perfect_agreement(self):
        self.assertAlmostEqual(pl.weighted_kappa([0, 1, 2, 1], [0, 1, 2, 1]), 1.0)

    def test_opposite_extremes_are_negative(self):
        self.assertLess(pl.weighted_kappa([0, 2, 0, 2], [2, 0, 2, 0]), 0)

    def test_partial_agreement_in_between(self):
        k = pl.weighted_kappa([0, 1, 2, 2, 1, 0], [0, 1, 2, 1, 1, 1])
        self.assertTrue(0 < k < 1)

    def test_near_miss_beats_far_miss(self):
        near = pl.weighted_kappa([0, 1, 2, 0], [1, 1, 2, 0])
        far = pl.weighted_kappa([0, 1, 2, 0], [2, 1, 2, 0])
        self.assertGreater(near, far)

    def test_constant_identical(self):
        self.assertEqual(pl.weighted_kappa([1, 1, 1], [1, 1, 1]), 1.0)

    def test_constant_different(self):
        self.assertEqual(pl.weighted_kappa([1, 1], [2, 2]), 0.0)

    def test_errors(self):
        with self.assertRaises(ValueError):
            pl.weighted_kappa([1], [1, 2])
        with self.assertRaises(ValueError):
            pl.weighted_kappa([], [])

    def test_agreement_uses_shared_items_only(self):
        a = {("r", "1"): 0, ("r", "2"): 2, ("r", "3"): 1}
        b = {("r", "1"): 0, ("r", "2"): 2, ("r", "4"): 0}
        out = pl.agreement([a, b])
        self.assertEqual(out["pairs"][(0, 1)][1], 2)
        self.assertAlmostEqual(out["mean_kappa"], 1.0)

    def test_agreement_without_overlap(self):
        self.assertIsNone(pl.agreement([{("r", "1"): 1}, {("r", "2"): 1}])["mean_kappa"])


if __name__ == "__main__":
    unittest.main()


class SelectTests(unittest.TestCase):
    SPLIT = {f"resume:{i}": ("dev" if i % 2 else "test") for i in range(1, 25)}
    CATS = {str(i): f"C{i % 4}" for i in range(1, 25)}

    def test_sizes_and_split_respected(self):
        out = pl.select_relevance_resumes(self.SPLIT, self.CATS, n_dev=4, n_test=4)
        self.assertEqual((len(out["dev"]), len(out["test"])), (4, 4))
        self.assertTrue(all(self.SPLIT[f"resume:{r}"] == "dev" for r in out["dev"]))
        self.assertTrue(all(self.SPLIT[f"resume:{r}"] == "test" for r in out["test"]))

    def test_categories_covered_before_repeats(self):
        out = pl.select_relevance_resumes(self.SPLIT, self.CATS, n_dev=2, n_test=2)
        # dev ids are odd -> categories C1, C3 only; both must appear
        self.assertEqual({self.CATS[r] for r in out["dev"]}, {"C1", "C3"})

    def test_deterministic_and_capped(self):
        a = pl.select_relevance_resumes(self.SPLIT, self.CATS, n_dev=50, n_test=1)
        b = pl.select_relevance_resumes(self.SPLIT, self.CATS, n_dev=50, n_test=1)
        self.assertEqual(a, b)
        self.assertEqual(len(a["dev"]), 12)  # only 12 dev resumes exist

    def test_ignores_postings(self):
        out = pl.select_relevance_resumes({"posting:9": "dev"}, {}, 3, 3)
        self.assertEqual(out, {"dev": [], "test": []})


class AssignTests(unittest.TestCase):
    def test_shared_and_round_robin(self):
        out = pl.assign_resumes(list("abcdefg"), ["x", "y"], shared=2)
        self.assertEqual(out["x"], ["a", "b", "c", "e", "g"])
        self.assertEqual(out["y"], ["a", "b", "d", "f"])

    def test_sheets_follow_assignment(self):
        pool = {"r1": [{"job_id": "j1", "methods": []}], "r2": [{"job_id": "j2", "methods": []}]}
        with tempfile.TemporaryDirectory() as d:
            paths = pl.write_sheets(pool, {"r1": "a", "r2": "b"}, {}, d, ["x", "y"],
                                    assignment={"x": ["r1"], "y": ["r1", "r2"]})
            rows = {os.path.basename(p): list(csv.DictReader(open(p))) for p in paths[1:]}
        self.assertEqual([r["resume_id"] for r in rows["grades_x.csv"]], ["r1"])
        self.assertEqual(len(rows["grades_y.csv"]), 2)
