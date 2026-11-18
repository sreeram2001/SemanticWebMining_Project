"""Unit tests for linking_evaluation.py (standard library only, no API key needed).

    cd CODE && python -m unittest test_linking_evaluation
"""

import csv
import os
import tempfile
import unittest

import linking_evaluation as le
import skill_linking as sl
from test_skill_linking import skills


def record(kind, doc_id, names):
    field = sl.SKILL_FIELDS[kind]
    facts = {field: [{"name": n, "source_span": f"uses {n}"} for n in names]}
    return {"doc_id": doc_id, "kind": kind, "status": "ok", "facts": facts, "rejected": []}


RECORDS = [
    record("resume", "r1", ["Python", "Java", "Postgres"]),
    record("resume", "r2", ["Python", "communication skills"]),
    record("posting", "p1", ["PostgreSQL", "network architecture design", "ground station operations"]),
    record("posting", "p2", ["Python"]),
    {"doc_id": "p3", "kind": "posting", "status": "error"},
]


class TestSampling(unittest.TestCase):
    def test_one_per_distinct_mention_and_balanced(self):
        sample = le.sample_mentions(RECORDS, n=4)
        self.assertEqual(len(sample), 4)
        kinds = [s[0] for s in sample]
        self.assertEqual(kinds.count("resume"), 2)
        resume_names = [sl.normalize_name(s[2]) for s in sample if s[0] == "resume"]
        self.assertEqual(len(resume_names), len(set(resume_names)))

    def test_tops_up_from_other_kind(self):
        sample = le.sample_mentions(RECORDS, n=7, resume_share=0.9)
        self.assertEqual(len(sample), 7)  # 4 distinct resume mentions + 3 postings

    def test_deterministic(self):
        self.assertEqual(le.sample_mentions(RECORDS, n=5), le.sample_mentions(RECORDS, n=5))

    def test_split_stable_per_document(self):
        self.assertEqual(le.split_of("resume", "r1"), le.split_of("resume", "r1"))
        self.assertIn(le.split_of("posting", "p1"), ("dev", "test"))


class TestSheet(unittest.TestCase):
    def setUp(self):
        self.linker = sl.Linker(skills(), aliases={"postgres": "postgresql"})
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "gold.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def test_template_rows_and_no_overwrite(self):
        sample = le.sample_mentions(RECORDS, n=6)
        self.assertEqual(le.write_gold_template(self.linker, sample, self.path), 6)
        self.assertEqual(le.write_gold_template(self.linker, sample, self.path), 0)
        with open(self.path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0]), le.COLUMNS)
        self.assertTrue(all(r["gold_id"] == "" for r in rows))
        python = next(r for r in rows if r["mention"] == "Python")
        shown = [python[f"candidate_{i}"] for i in range(1, le.CANDIDATES_SHOWN + 1)]
        self.assertIn("KS1 | Python (Programming Language)", shown)

    def _write_labels(self, labels):
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=le.COLUMNS)
            writer.writeheader()
            for i, (mention, gold) in enumerate(labels):
                writer.writerow({"mention_id": f"m{i}", "split": "test", "kind": "resume", "doc_id": "d",
                                 "mention": mention, "context": mention, "gold_id": gold})

    def test_load_gold_forms_and_problems(self):
        self._write_labels([("Python", "KS1"), ("Postgres", "KS2 | PostgreSQL"), ("ground ops", "nil"),
                            ("typo", "KS999"), ("not yet", "")])
        rows, problems, unlabeled = le.load_gold(self.path, skills())
        self.assertEqual([r["gold_id"] for r in rows], ["KS1", "KS2", None])
        self.assertEqual(unlabeled, 1)
        self.assertEqual(len(problems), 1)
        self.assertIn("KS999", problems[0])

    def test_search_skills(self):
        hits = le.search_skills(skills(), "network sec")
        self.assertEqual(hits[0][0], "KS11")
        self.assertEqual(le.search_skills(skills(), "  "), [])


class TestScoring(unittest.TestCase):
    ROWS = [
        {"mention_id": "1", "mention": "Python", "context": "Python", "gold_id": "KS1"},
        {"mention_id": "2", "mention": "Postgres", "context": "Postgres", "gold_id": "KS2"},
        {"mention_id": "3", "mention": "network architecture design", "context": "", "gold_id": "KS10"},
        {"mention_id": "4", "mention": "ground station operations", "context": "", "gold_id": None},
        {"mention_id": "5", "mention": "Design instruction creation", "context": "", "gold_id": None},
    ]

    def test_metrics(self):
        result = le.evaluate(sl.Linker(skills(), aliases={"postgres": "postgresql"}), self.ROWS, n_boot=200)
        m = result["metrics"]
        self.assertEqual((m["linkable"], m["gold_nil"]), (3, 2))
        self.assertEqual(m["accuracy@1"], 1.0)
        self.assertEqual(m["nil_accuracy"], 1.0)
        self.assertEqual(m["candidate_recall"], 1.0)

    def test_baseline_is_weaker_and_compared(self):
        configs = {
            "alias-only": sl.Linker(skills(), fuzzy=False, contains=False),
            "full": sl.Linker(skills(), aliases={"postgres": "postgresql"}),
        }
        table, results = le.compare(configs, self.ROWS, n_boot=200)
        self.assertEqual(table[0]["config"], "alias-only")
        self.assertAlmostEqual(table[0]["accuracy@1"], 1 / 3, places=3)  # only Python
        self.assertEqual(table[1]["accuracy@1"], 1.0)
        self.assertNotIn("p_acc@1_vs_baseline", table[0])
        self.assertGreater(table[1]["p_acc@1_vs_baseline"], 0)
        self.assertEqual(len(results["full"]["per_mention"]), 5)

    def test_bootstrap_bounds(self):
        per = [{"correct": i % 2 == 0, "gold_nil": False, "pred": "x", "in_candidates": True} for i in range(40)]
        lo, hi = le.bootstrap(per, "accuracy@1", n=300)
        self.assertLessEqual(lo, 0.5)
        self.assertGreaterEqual(hi, 0.5)
        self.assertIsNone(le.bootstrap(per[:1], "accuracy@1"))

    def test_randomization(self):
        same = [True, False] * 20
        self.assertEqual(le.paired_randomization(same, same), 1.0)
        better = [True] * 40
        worse = [False] * 40
        self.assertLess(le.paired_randomization(better, worse, n=2000), 0.01)


if __name__ == "__main__":
    unittest.main()
