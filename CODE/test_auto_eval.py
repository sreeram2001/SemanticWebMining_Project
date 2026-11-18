import unittest

import auto_eval as ae


class PatternTests(unittest.TestCase):
    def test_titles_map_to_categories(self):
        rel = ae.relevant_jobs({"1": "Senior Software Engineer", "2": "Registered Nurse (RN)",
                                "3": "Staff Accountant", "4": "Truck Driver", "5": None})
        self.assertIn("1", rel["INFORMATION-TECHNOLOGY"])
        self.assertIn("2", rel["HEALTHCARE"])
        self.assertIn("3", rel["ACCOUNTANT"])
        self.assertFalse(any("4" in s or "5" in s for s in rel.values()))

    def test_whole_words_only(self):
        rel = ae.relevant_jobs({"1": "Brand Ambassador"}, {"HR": r"\bhr\b"})
        self.assertEqual(rel["HR"], set())

    def test_every_resume_category_has_a_pattern(self):
        self.assertEqual(len(ae.CATEGORY_TITLES), 12)


class EvalTests(unittest.TestCase):
    DATA = {"runs": {"bm25": {"r1": [["a", 1], ["b", 0.5]]}, "rrf": {"r1": [["b", 1], ["a", 0.5]]}},
            "rerank": {"r1": [["a", 2.0, 1.0]]}}

    def test_runs_and_scores(self):
        runs = ae.method_runs(self.DATA)
        self.assertEqual(set(runs), {"bm25", "rrf", "rerank"})
        qrels = ae.qrels_for({"r1": "HR"}, {"HR": {"a"}})
        table = ae.evaluate_all(runs, qrels, k=1)
        self.assertEqual(table["bm25"]["P@1"], 1.0)
        self.assertEqual(table["rrf"]["P@1"], 0.0)

    def test_per_category_and_markdown(self):
        runs = ae.method_runs(self.DATA)
        qrels = ae.qrels_for({"r1": "HR"}, {"HR": {"a"}})
        cats = ae.per_category(runs, qrels, {"r1": "HR"}, k=1)
        self.assertEqual(cats["HR"]["bm25"], 1.0)
        md = ae.markdown(ae.evaluate_all(runs, qrels, 1), cats, {"HR": 1}, 1, k=1)
        self.assertIn("| HR | 1 |", md)


if __name__ == "__main__":
    unittest.main()
