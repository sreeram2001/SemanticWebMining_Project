import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from evaluation import (
    agreement, compare_models, comparison_rows, fact_key, for_kind, freeze_split, gold_docs,
    load_gold, mask_contacts, score, ungrounded, write_templates,
)
from extraction import UNSUPPORTED, ground, prompt_hash

RESUME = "Skills: Python, SQL, Tableau.  Experience: Data Analyst at Company Name.  Education: BS Statistics"
POSTING = "Need 3+ years of experience. Must know SQL. Spark is a plus. Bachelor's in Computer Science required."

GOLD_RESUME = {
    "skills": [{"name": "Python", "source_span": "Python"}, {"name": "SQL", "source_span": "SQL"},
               {"name": "Tableau", "source_span": "Tableau"}],
    "degrees": [{"level": "bachelors", "field": "Statistics", "institution": None, "year": None,
                 "source_span": "BS Statistics"}],
    "experience": [{"title": "Data Analyst", "company": None, "duration_months": None,
                    "source_span": "Data Analyst"}],
}
GOLD_POSTING = {
    "required_skills": [{"name": "SQL", "importance": "required", "source_span": "Must know SQL"},
                        {"name": "Spark", "importance": "preferred", "source_span": "Spark is a plus"}],
    "required_degrees": [{"level": "bachelors", "field": "Computer Science", "importance": "required",
                          "source_span": "Bachelor's in Computer Science"}],
    "experience_required": {"min_years": 3, "importance": "required", "source_span": "3+ years of experience"},
}


class ResponderClient:
    """openai.OpenAI stand-in whose reply depends on the model and prompt."""

    def __init__(self, respond):
        self.respond = respond
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        content = json.dumps(self.respond(kw["model"], kw["messages"][1]["content"]))
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def write_gold(gold_dir, kind, doc_id, text, facts, annotator="aakash"):
    write_templates(kind, [(doc_id, text, "IT")], gold_dir)
    path = os.path.join(gold_dir, f"{kind}_{doc_id}.json")
    with open(path) as f:
        doc = json.load(f)
    doc.update(annotator=annotator, facts=facts)
    with open(path, "w") as f:
        json.dump(doc, f)


class KeyTest(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(fact_key("skills", {"name": "  MS  Visio. "}), "ms visio")
        self.assertEqual(fact_key("required_degrees", {"level": "masters", "field": "Data Science"}),
                         "masters|data science")
        self.assertEqual(fact_key("degrees", {"level": "phd", "field": None}), "phd|")
        self.assertEqual(fact_key("experience_required", {"min_years": 3}), "3.0")
        with self.assertRaises(ValueError):
            fact_key("hobbies", {})


class UngroundedTest(unittest.TestCase):
    def test_restores_all_rejected_facts(self):
        raw = {"skills": [{"name": "SQL", "source_span": "SQL"},
                          {"name": "Kubernetes", "source_span": "Kubernetes"},
                          {"name": "", "source_span": "Python"}],
               "degrees": [{"level": "diploma", "field": None, "institution": None, "year": None,
                            "source_span": "Education"}],
               "experience": "not a list"}
        facts, rejected = ground("resume", raw, RESUME)
        rec = {"kind": "resume", "facts": facts, "rejected": rejected}
        plain = ungrounded(rec)
        self.assertEqual([s["name"] for s in plain["skills"]], ["SQL", "Kubernetes", ""])
        self.assertEqual(plain["degrees"][0]["level"], "diploma")
        self.assertEqual(plain["experience"], [])
        self.assertEqual([s["name"] for s in facts["skills"]], ["SQL"])

    def test_bad_values_still_score(self):
        self.assertEqual(fact_key("experience_required", {"min_years": "five"}), "five")
        self.assertEqual(fact_key("experience_required", {"min_years": True}), "True")

    def test_single_field(self):
        raw = {"required_skills": [], "required_degrees": [],
               "experience_required": {"min_years": 5, "importance": "required", "source_span": "5 years"}}
        facts, rejected = ground("posting", raw, POSTING)
        self.assertIsNone(facts["experience_required"])
        rec = {"kind": "posting", "facts": facts, "rejected": rejected}
        self.assertEqual(ungrounded(rec)["experience_required"]["min_years"], 5)


class MaskTest(unittest.TestCase):
    def test_mask_contacts(self):
        text = "Apply to jobs@acme.co or call (480) 555-0199, see https://acme.co/jobs. Salary $95,000."
        self.assertEqual(mask_contacts(text),
                         "Apply to [EMAIL] or call [PHONE], see [URL] Salary $95,000.")
        self.assertIsNone(mask_contacts(None))


class GoldFilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_templates_never_overwrite(self):
        self.assertEqual(write_templates("resume", [("r/1", RESUME, "IT")], self.dir), (1, 0))
        path = os.path.join(self.dir, "resume_r_1.json")
        with open(path) as f:
            doc = json.load(f)
        self.assertEqual(doc["facts"], {"skills": [], "degrees": [], "experience": []})
        doc["annotator"] = "richitha"
        with open(path, "w") as f:
            json.dump(doc, f)
        self.assertEqual(write_templates("resume", [("r/1", RESUME, "IT")], self.dir), (0, 1))
        with open(path) as f:
            self.assertEqual(json.load(f)["annotator"], "richitha")

    def test_posting_template_has_null_experience(self):
        write_templates("posting", [(7, POSTING, None)], self.dir)
        with open(os.path.join(self.dir, "posting_7.json")) as f:
            self.assertIsNone(json.load(f)["facts"]["experience_required"])

    def test_load_gold(self):
        write_gold(self.dir, "resume", "1", RESUME, GOLD_RESUME)
        write_templates("resume", [("2", RESUME, "IT")], self.dir)
        typo = dict(GOLD_RESUME, skills=[{"name": "Python", "source_span": "Pyhton"}])
        write_gold(self.dir, "resume", "3", RESUME, typo)
        with open(os.path.join(self.dir, "resume_4.json"), "w") as f:
            f.write("{not json")

        gold, problems, unlabeled = load_gold(self.dir)
        self.assertEqual(sorted(gold), [("resume", "1"), ("resume", "3")])
        self.assertEqual(unlabeled, 1)
        self.assertEqual(len(gold[("resume", "1")]["skills"]), 3)
        self.assertEqual(len(problems), 2)
        self.assertIn("Pyhton", problems[0])
        self.assertIn(UNSUPPORTED, problems[0])
        self.assertIn("unreadable", problems[1])
        self.assertEqual([d for d, _ in gold_docs(self.dir, "resume")], ["1", "3"])


class ScoreTest(unittest.TestCase):
    def gold(self):
        facts, _ = ground("resume", GOLD_RESUME, RESUME)
        return {"1": facts}

    def test_counts(self):
        pred = {"1": {"skills": [{"name": "python"}, {"name": "Excel"}],
                      "degrees": [{"level": "bachelors", "field": "statistics"}],
                      "experience": []}}
        s = score("resume", pred, self.gold())
        skills = s["fields"]["skills"]
        self.assertEqual((skills["tp"], skills["fp"], skills["fn"]), (1, 1, 2))
        self.assertAlmostEqual(skills["precision"], 0.5)
        self.assertAlmostEqual(skills["recall"], 1 / 3)
        self.assertEqual(s["fields"]["degrees"]["f1"], 1.0)
        # experience: nothing predicted against one gold fact
        self.assertIsNone(s["fields"]["experience"]["precision"])
        self.assertEqual(s["fields"]["experience"]["f1"], 0.0)
        self.assertAlmostEqual(s["macro_f1"], (0.4 + 1.0 + 0.0) / 3)

    def test_field_without_gold_is_left_out_of_macro(self):
        gold = {"1": {"skills": [{"name": "SQL"}], "degrees": [], "experience": []}}
        pred = {"1": {"skills": [{"name": "SQL"}], "degrees": [{"level": "phd", "field": None}], "experience": []}}
        s = score("resume", pred, gold)
        self.assertEqual(s["macro_f1"], 1.0)
        self.assertEqual(s["fields"]["degrees"]["fp"], 1)

    def test_missing_prediction_counts_against_recall(self):
        gold = self.gold()
        gold["2"] = gold["1"]
        s = score("resume", {"1": gold["1"]}, gold)
        self.assertEqual(s["missing_predictions"], 1)
        self.assertAlmostEqual(s["fields"]["skills"]["recall"], 0.5)

    def test_duplicate_skills_count_once(self):
        pred = {"1": {"skills": [{"name": "SQL"}, {"name": "sql"}], "degrees": [], "experience": []}}
        s = score("resume", pred, self.gold())
        self.assertEqual(s["fields"]["skills"]["fp"], 0)

    def test_bootstrap_ci(self):
        gold, pred = {}, {}
        for i in range(20):
            gold[str(i)] = {"skills": [{"name": "SQL"}, {"name": "Python"}], "degrees": [], "experience": []}
            pred[str(i)] = {"skills": [{"name": "SQL"}] + ([{"name": "Python"}] if i % 2 else []),
                            "degrees": [], "experience": []}
        s = score("resume", pred, gold)
        lo, hi = s["macro_f1_ci95"]
        self.assertLessEqual(lo, s["macro_f1"])
        self.assertGreaterEqual(hi, s["macro_f1"])
        self.assertLess(lo, hi)
        self.assertEqual(score("resume", pred, gold)["macro_f1_ci95"], (lo, hi))

    def test_empty_gold(self):
        self.assertIsNone(score("resume", {}, {}))


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gold_dir = os.path.join(self.tmp.name, "gold")
        self.out_dir = os.path.join(self.tmp.name, "runs")
        write_gold(self.gold_dir, "resume", "1", RESUME, GOLD_RESUME)
        write_gold(self.gold_dir, "posting", "9", POSTING, GOLD_POSTING)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def respond(model, prompt):
        resume = "RESUME TEXT" in prompt
        if model == "careful":
            return GOLD_RESUME if resume else GOLD_POSTING
        # invents a skill with a span that is not in the text, and misses one
        if resume:
            return dict(GOLD_RESUME, skills=GOLD_RESUME["skills"][:2] + [
                {"name": "Kubernetes", "source_span": "Kubernetes"}])
        return dict(GOLD_POSTING, required_skills=[
            {"name": "SQL", "importance": "required", "source_span": "Must know SQL"},
            {"name": "Spark", "importance": "preferred", "source_span": "Apache Spark preferred"}])

    def test_compare(self):
        client = ResponderClient(self.respond)
        results = compare_models(client, ["careful", "org/sloppy"], self.gold_dir, self.out_dir, backoff=0)

        careful = results["careful"]["resume"]
        self.assertEqual(careful["grounded"]["macro_f1"], 1.0)
        self.assertEqual(results["careful"]["posting"]["grounded"]["macro_f1"], 1.0)

        sloppy = results["org/sloppy"]
        # grounding drops the invented skill, so precision stays perfect
        self.assertEqual(sloppy["resume"]["grounded"]["fields"]["skills"]["precision"], 1.0)
        self.assertLess(sloppy["resume"]["ungrounded"]["fields"]["skills"]["precision"], 1.0)
        # but a real skill quoted with a wrong span is lost to grounding
        self.assertEqual(sloppy["posting"]["grounded"]["fields"]["required_skills"]["recall"], 0.5)
        self.assertEqual(sloppy["posting"]["ungrounded"]["fields"]["required_skills"]["recall"], 1.0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, f"org_sloppy_resume_{prompt_hash('resume')}.jsonl")))
        self.assertEqual(sloppy["resume"]["config"]["model"], "org/sloppy")
        self.assertEqual(sloppy["resume"]["part"], "all")

        rows = comparison_rows(results)
        self.assertEqual(len(rows), 8)
        row = next(r for r in rows if r["model"] == "org/sloppy" and r["kind"] == "resume"
                   and r["mode"] == "ungrounded")
        # 1 invented skill out of 5 facts (2 skills, 1 degree, 1 role kept)
        self.assertAlmostEqual(row["unsupported_rate"], 0.2)
        self.assertEqual(row["failed_docs"], 0)

        # rerunning reuses the saved output instead of calling the model again
        calls = []
        counting = ResponderClient(lambda m, p: calls.append(m) or self.respond(m, p))
        again = compare_models(counting, ["careful"], self.gold_dir, self.out_dir)
        self.assertEqual(calls, [])
        self.assertEqual(again["careful"]["resume"]["grounded"]["macro_f1"], 1.0)

    def test_agreement(self):
        second = os.path.join(self.tmp.name, "second")
        other = dict(GOLD_RESUME, skills=GOLD_RESUME["skills"][:2])
        write_gold(second, "resume", "1", RESUME, other, annotator="kaushal")
        result = agreement(self.gold_dir, second)
        self.assertEqual(list(result), ["resume"])
        self.assertAlmostEqual(result["resume"]["fields"]["skills"]["f1"], 0.8)


class SplitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, "gold")
        self.path = os.path.join(self.tmp.name, "split.json")
        for cat in ("IT", "HR", "SALES"):
            write_templates("resume", [(f"{cat}{i}", RESUME, cat) for i in range(5)], self.dir)
        write_templates("posting", [(i, POSTING, None) for i in range(10)], self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_stratified_halves(self):
        split = freeze_split(self.dir, self.path)
        self.assertEqual(len(split), 25)
        self.assertEqual(sum(v == "dev" for v in split.values()), 13)
        for cat in ("IT", "HR", "SALES"):
            dev = sum(split[("resume", f"{cat}{i}")] == "dev" for i in range(5))
            self.assertIn(dev, (2, 3))
        posting_dev = sum(v == "dev" for (k, _), v in split.items() if k == "posting")
        self.assertEqual(posting_dev, 5)

    def test_frozen(self):
        first = freeze_split(self.dir, self.path)
        write_templates("posting", [(99, POSTING, None)], self.dir)
        self.assertEqual(freeze_split(self.dir, self.path, seed=1), first)
        self.assertNotIn(("posting", "99"), first)

    def test_compare_on_dev_only(self):
        split = freeze_split(self.dir, self.path)
        dev_id = next(d for (k, d), v in split.items() if k == "posting" and v == "dev")
        test_id = next(d for (k, d), v in split.items() if k == "posting" and v == "test")
        for doc_id in (dev_id, test_id):
            write_gold_over(self.dir, "posting", doc_id, POSTING, GOLD_POSTING)
        seen = []
        client = ResponderClient(lambda m, p: seen.append(p) or GOLD_POSTING)
        results = compare_models(client, ["m"], self.dir, os.path.join(self.tmp.name, "runs"), split=split)
        entry = results["m"]["posting"]
        self.assertEqual((entry["part"], entry["grounded"]["documents"], len(seen)), ("dev", 1, 1))


def write_gold_over(gold_dir, kind, doc_id, text, facts):
    path = os.path.join(gold_dir, f"{kind}_{doc_id}.json")
    with open(path) as f:
        doc = json.load(f)
    doc.update(annotator="richitha", facts=facts)
    with open(path, "w") as f:
        json.dump(doc, f)


class ForKindTest(unittest.TestCase):
    def test_for_kind(self):
        gold = {("resume", "1"): "a", ("posting", "1"): "b"}
        self.assertEqual(for_kind(gold, "posting"), {"1": "b"})


if __name__ == "__main__":
    unittest.main()
