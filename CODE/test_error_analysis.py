import unittest

import error_analysis as ea
from test_graph_score import NU, PY, job, resume
from test_tune_ranking import DATA, JOBS, RESUMES

SETTING = {"betas": {"b1": 1.0, "b2": 1.0, "b3": 0.0}, "lambdas": {"lam_sub": 0.5, "lam_cat": 0.25}}


class AnalyzeTests(unittest.TestCase):
    def test_graph_helped_when_relevant_job_enters_top(self):
        cases = ea.analyze(DATA, JOBS, RESUMES, {"r1": {"jA": 0, "jB": 2}}, ["r1"], SETTING, k=1)
        kinds = {(c["job"], c["kind"]) for c in cases}
        self.assertEqual(kinds, {("jA", "graph helped"), ("jB", "graph helped")})

    def test_graph_hurt_gets_a_cause(self):
        cases = ea.analyze(DATA, JOBS, RESUMES, {"r1": {"jA": 2, "jB": 0}}, ["r1"], SETTING, k=1)
        hurt = [c for c in cases if c["kind"] == "graph hurt"]
        self.assertEqual(len(hurt), 2)
        self.assertTrue(all(c["cause"] in ea.CAUSES for c in hurt))

    def test_missed_relevant_outside_shortlist_is_retrieval(self):
        cases = ea.analyze(DATA, JOBS, RESUMES, {"r1": {"jZ": 2}}, ["r1"], SETTING, k=1)
        self.assertEqual([(c["job"], c["cause"]) for c in cases if c["kind"] == "missed relevant"],
                         [("jZ", "retrieval")])

    def test_missed_relevant_inside_shortlist_is_ranking(self):
        no_graph = {**SETTING, "betas": {"b1": 1.0, "b2": 0.0, "b3": 0.0}}
        cases = ea.analyze(DATA, JOBS, RESUMES, {"r1": {"jB": 2}}, ["r1"], no_graph, k=1)
        self.assertEqual([(c["job"], c["cause"]) for c in cases], [("jB", "ranking")])

    def test_unjudged_moves(self):
        cases = ea.analyze(DATA, JOBS, RESUMES, {}, ["r1"], SETTING, k=1)
        self.assertEqual({c["kind"] for c in cases}, {"unjudged"})


class CauseTests(unittest.TestCase):
    def test_order_of_causes(self):
        d = {"unresolved": 0, "linked": 2, "p": 0.5}
        self.assertEqual(ea.cause(d, False, True, False), "extraction")
        self.assertEqual(ea.cause({**d, "unresolved": 3}, True, True, False), "linking")
        self.assertEqual(ea.cause(d, True, True, True), "penalty")
        self.assertEqual(ea.cause(d, True, True, False), "graph scoring")

    def test_job_detail_lists_credited_skills(self):
        d = ea.job_detail(job([(PY, "required"), (NU, "required")]), resume([PY]),
                          {"lam_sub": 0.5, "lam_cat": 0.25}, None)
        self.assertEqual((d["linked"], d["g"]), (2, 0.5))
        self.assertEqual([e["skill"] for e in d["evidence"]], ["Python"])


class ReportTests(unittest.TestCase):
    def test_summary_and_markdown(self):
        cases = ea.analyze(DATA, JOBS, RESUMES, {"r1": {"jA": 2, "jB": 0}}, ["r1"], SETTING, k=1)
        s = ea.summarize(cases)
        self.assertEqual(s["by_kind"], {"graph hurt": 2, "missed relevant": 1})
        self.assertEqual(s["missed_relevant_by_cause"], {"ranking": 1})
        self.assertIn("Graph hurt, by cause", ea.markdown(s, "dev", k=1))


if __name__ == "__main__":
    unittest.main()
