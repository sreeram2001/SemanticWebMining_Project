import unittest

import tune_ranking as tr
from test_graph_score import JV, NU, PY, job, resume

DATA = {
    "runs": {m: {"r1": [["jA", 2.0], ["jB", 1.0]], "r2": [["jA", 1.0]]}
             for m in ("bm25", "dense", "hybrid", "rrf")},
    "rerank": {"r1": [["jA", 3.0, 1.0], ["jB", 2.5, 0.8]],
               "r2": [["jA", 1.0, 0.0]]},
}
JOBS = {"jA": job([(NU, "required")]), "jB": job([(PY, "required")])}
RESUMES = {"r1": resume([PY]), "r2": resume([JV])}
QRELS = {"r1": {"jA": 0, "jB": 2}}


class FullRunTests(unittest.TestCase):
    def test_graph_fixes_order(self):
        run = tr.full_run(DATA, JOBS, RESUMES, {"b2": 1.0})
        self.assertEqual(run["r1"], ["jB", "jA"])
        no_graph = tr.full_run(DATA, JOBS, RESUMES, {"b2": 0.0})
        self.assertEqual(no_graph["r1"], ["jA", "jB"])

    def test_resume_without_record(self):
        run = tr.full_run({"runs": {"rrf": {}}, "rerank": {"zz": [["jA", 1, 0.3]]}}, JOBS, {}, None)
        self.assertEqual(run, {"zz": ["jA"]})


class MethodsTests(unittest.TestCase):
    def test_all_methods_present(self):
        runs = tr.method_runs(DATA, JOBS, RESUMES, {"b2": 1.0})
        self.assertEqual(set(runs), set(tr.BASELINES) | {"full", "full_no_graph"})
        self.assertEqual(runs["rerank"]["r1"], ["jA", "jB"])

    def test_compare_reports_positive_graph_delta(self):
        runs = tr.method_runs(DATA, JOBS, RESUMES, {"b2": 1.0})
        table = tr.compare(runs, QRELS, ["r1"], k=2)
        self.assertEqual(table["full"]["nDCG@2"], 1.0)
        self.assertGreater(table["delta_graph"]["nDCG@2"], 0)
        self.assertEqual(table["full"]["n"], 1)
        self.assertIn("ΔnDCG@2", tr.markdown(table, "t", k=2))


class TuneTests(unittest.TestCase):
    def test_picks_graph_weight_on_dev(self):
        grid = {"b2": [0.0, 1.0], "b3": [0.0], "lambdas": [(0.5, 0.25)], "w_required": [2.0], "top": [None]}
        best, tried = tr.tune(DATA, JOBS, RESUMES, QRELS, ["r1"], grid, k=2)
        self.assertEqual(best["betas"]["b2"], 1.0)
        self.assertEqual(len(tried), 2)

    def test_ties_keep_first_setting(self):
        grid = {"b2": [0.0, 0.1], "b3": [0.0], "lambdas": [(0.5, 0.25)]}
        best, _ = tr.tune(DATA, JOBS, RESUMES, {"r1": {"jA": 1}}, ["r1"], grid, k=2)
        self.assertEqual(best["betas"]["b2"], 0.0)

    def test_only_dev_resumes_used(self):
        grid = {"b2": [1.0], "b3": [0.0], "lambdas": [(0.5, 0.25)]}
        best, _ = tr.tune(DATA, JOBS, RESUMES, QRELS, ["r2"], grid, k=2)
        self.assertEqual(best["nDCG@2"], 0.0)


if __name__ == "__main__":
    unittest.main()


class ShortlistTests(unittest.TestCase):
    def test_top_keeps_rrf_head_and_renormalizes(self):
        data = {"runs": {"rrf": {"r1": [["jB", 1], ["jA", 0.5]]}},
                "rerank": {"r1": [["jA", 3.0, 1.0], ["jB", 2.5, 0.0]]}}
        self.assertEqual(tr.shortlist(data, "r1", top=1), [("jB", 0.0)])
        self.assertEqual(tr.shortlist(data, "r1"), [("jA", 1.0), ("jB", 0.0)])

    def test_grid_tries_every_combination(self):
        grid = {"b2": [0.0, 1.0], "b3": [0.0], "lambdas": [(0.5, 0.25)],
                "w_required": [1.0, 2.0], "top": [1, None]}
        best, tried = tr.tune(DATA, JOBS, RESUMES, QRELS, ["r1"], grid, k=2)
        self.assertEqual(len(tried), 8)
        self.assertEqual(best["betas"]["b2"], 1.0)

    def test_setting_args(self):
        out = tr.setting_args({"lambdas": {"lam_sub": 0.6, "lam_cat": 0.3}, "w_required": 3.0, "top": 20})
        self.assertEqual(out["weights"], {"required": 3.0, "preferred": 1.0})
        self.assertEqual(out["top"], 20)
