"""Chat agent, uploaded-resume and extraction-eval tests (offline).

    cd CODE && python -m unittest tests.test_chat
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from assistant.chat import HELP, CareerAgent
from core.config import load_config
from core.pipeline import InputError, Pipeline

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # CODE/

NEW_RESUME = """Jane Doe  jane@example.com  (480) 555-1234
SOFTWARE DEVELOPER
Skills: Python, React, JavaScript, Docker, Kubernetes, AWS, SQL, Git
Experience
Software Developer  Jan 2017 to Current  Built REST APIs and React front ends deployed on AWS.
Education: Bachelor of Science in Computer Science, 2016"""


class FakeLLM:
    """OpenAI-shaped client that returns a fixed reply."""

    def __init__(self, text):
        self.text = text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))])


class TestChat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = tempfile.mkdtemp()
        cfg = load_config(os.path.join(HERE, "sample_data", "config_sample.json"), {
            "paths": {"extracted_postings": os.path.join(cls.out, "p.jsonl"),
                      "extracted_resumes": os.path.join(cls.out, "r.jsonl"),
                      "split": os.path.join(cls.out, "split.json"), "out_dir": cls.out},
            "evaluation": {"bootstrap": 20, "permutations": 20}})
        cls.p = Pipeline(cfg, use_models=False)
        cls.p.extract()
        cls.agent = CareerAgent(cls.p, use_llm=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out)

    def setUp(self):
        self.a = self.agent.session()

    def test_needs_resume_first(self):
        self.assertEqual(self.a.reply("help"), HELP)
        self.assertIn("load a resume first", self.a.reply("jobs"))

    def test_dialogue(self):
        self.assertIn("loaded", self.a.use_resume("2002"))
        jobs = self.a.reply("top 3")
        self.assertIn("| 1 | Data Scientist |", jobs)
        self.assertIn("verified against stored facts", self.a.reply("why 1"))
        self.assertIn("only", self.a.reply("why 99"))
        self.assertIn("Skills the top", self.a.reply("what skills am I missing?"))
        self.assertIn("Required skills", self.a.reply("job 1"))
        self.assertIn("without", self.a.reply("compare"))
        self.assertIn("`bm25`", self.a.reply("method bm25"))
        self.assertIn("Unknown method", self.a.reply("method nope"))
        self.assertIn("did not understand", self.a.reply("tell me a joke"))

    def test_sessions_are_independent(self):
        a, b = self.agent.session(), self.agent.session()
        a.use_resume("2001")
        self.assertIsNone(b.resume_id)
        self.assertEqual(self.agent.session("2001", "rrf").method, "rrf")

    def test_uploaded_resume(self):
        a = self.agent.session()
        msg = a.reply(NEW_RESUME * 2)                        # long text is read as a pasted resume
        self.assertIn("loaded", msg)
        rid = a.resume_id
        self.assertTrue(rid.startswith("user-"))
        self.assertNotIn("jane@example.com", self.p.graph.texts[rid])     # PII masked first
        self.assertIn("Kubernetes", msg)
        self.assertIn("|", a.reply("jobs"))
        self.assertEqual(self.p.add_resume(NEW_RESUME * 2), rid)         # same text, same ID
        self.assertNotIn(rid, [r for _, r in self.agent.sample_resumes()])
        with self.assertRaises(InputError):
            self.p.add_resume("too short")

    def test_free_form_answers_are_verified(self):
        a = self.agent.session("2002")
        a.client, a.model = FakeLLM("Your top match needs Python, which you have."), "fake"
        self.assertIn("checked against the stored facts", a.reply("which of my skills matter most?"))
        a.client = FakeLLM("You should learn Tableau and Accounts Payable.")
        self.assertIn("could not give a fact-checked answer", a.reply("which of my skills matter most?"))


class TestExtractionEval(unittest.TestCase):
    def test_sample_gold(self):
        out = tempfile.mkdtemp()
        try:
            cfg = load_config(os.path.join(HERE, "sample_data", "config_sample.json"), {"paths": {"out_dir": out}})
            res = Pipeline(cfg, use_models=False).evaluate_extraction()
            kinds = {(r["kind"], r["mode"]) for r in res["rows"]}
            self.assertEqual(kinds, {(k, m) for k in ("resume", "posting") for m in ("grounded", "ungrounded")})
            self.assertTrue(all(0 < r["macro_f1"] <= 1 for r in res["rows"]))
            self.assertEqual(set(res["agreement"]), {"resume", "posting"})
            self.assertTrue(os.path.exists(os.path.join(out, "eval_extraction.md")))
        finally:
            shutil.rmtree(out)

    def test_missing_gold(self):
        cfg = load_config(os.path.join(HERE, "sample_data", "config_sample.json"),
                          {"labeling": {"gold_dir": "nowhere"}})
        with self.assertRaises(InputError):
            Pipeline(cfg, use_models=False).evaluate_extraction()


class TestApp(unittest.TestCase):
    def test_build_app(self):
        try:
            import gradio  # noqa: F401
        except ImportError:
            self.skipTest("gradio not installed")
        from app import build_app, read_upload
        out = tempfile.mkdtemp()
        try:
            cfg = load_config(os.path.join(HERE, "sample_data", "config_sample.json"), {
                "paths": {"extracted_postings": os.path.join(HERE, "sample_data/out/extracted/postings.jsonl"),
                          "extracted_resumes": os.path.join(HERE, "sample_data/out/extracted/resumes.jsonl"),
                          "out_dir": out}})
            if not os.path.exists(cfg["paths"]["extracted_postings"]):
                self.skipTest("run `run_pipeline.py --sample extract` first")
            self.assertIsNotNone(build_app(CareerAgent(Pipeline(cfg, use_models=False), use_llm=False)))
            path = os.path.join(out, "cv.txt")
            with open(path, "w") as f:
                f.write("hello")
            self.assertEqual(read_upload(path), "hello")
        finally:
            shutil.rmtree(out)


if __name__ == "__main__":
    unittest.main()
