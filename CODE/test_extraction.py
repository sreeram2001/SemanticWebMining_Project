import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from extraction import (
    Extractor, UNSUPPORTED, done_ids, ground, load_records, locate_span,
    parse_json, run_batch, summarize,
)

RESUME = (
    "         INFORMATION TECHNOLOGY SPECIALIST       Summary     Network  administrator with "
    "experience in  Cisco routing and Active Directory.   Skills    Python, SQL,  MS Visio, "
    "“Lean” process design   Experience    IT Specialist   Jan 2015 to Mar 2018   "
    "Company Name   City , State   Maintained LAN/WAN links.   Education    Bachelor of Science : "
    "Computer Science   2014   University of Arizona"
)

POSTING = (
    "We are hiring a Data Analyst.\nRequirements:\n- 3+ years of experience with SQL\n"
    "- Bachelor's degree in Statistics or related field\n- Tableau is a plus\n"
)


class FakeError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeClient:
    """Stands in for openai.OpenAI; replies are queued per call."""

    def __init__(self, replies, reject_schema=False):
        self.replies = list(replies)
        self.reject_schema = reject_schema
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_schema and "response_format" in kwargs:
            raise FakeError(400)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content, finish = reply if isinstance(reply, tuple) else (reply, "stop")
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)])


class LocateSpanTest(unittest.TestCase):
    def test_exact(self):
        start, end = locate_span(RESUME, "MS Visio")
        self.assertEqual(RESUME[start:end], "MS Visio")

    def test_ignores_whitespace_and_case(self):
        start, end = locate_span(RESUME, "network administrator with experience in cisco routing")
        self.assertEqual(RESUME[start:end], "Network  administrator with experience in  Cisco routing")

    def test_smart_quotes(self):
        start, end = locate_span(RESUME, '"Lean" process design')
        self.assertEqual(RESUME[start:end], "“Lean” process design")

    def test_missing_and_empty(self):
        self.assertIsNone(locate_span(RESUME, "Kubernetes"))
        self.assertIsNone(locate_span(RESUME, "   "))
        self.assertIsNone(locate_span(RESUME, None))

    def test_span_at_end_of_text(self):
        start, end = locate_span(RESUME, "university of  arizona")
        self.assertEqual(RESUME[start:end], "University of Arizona")
        self.assertEqual(end, len(RESUME))


class GroundTest(unittest.TestCase):
    def test_resume(self):
        raw = {
            "skills": [
                {"name": "Python", "source_span": "Python"},
                {"name": "python", "source_span": "Python"},
                {"name": "Kubernetes", "source_span": "Kubernetes"},
                {"name": "", "source_span": "SQL"},
            ],
            "degrees": [
                {"level": "bachelors", "field": "Computer Science", "institution": "University of Arizona",
                 "year": "2014", "source_span": "Bachelor of Science : Computer Science"},
                {"level": "diploma", "field": None, "institution": None, "year": None, "source_span": "Education"},
            ],
            "experience": [
                {"title": "IT Specialist", "company": None, "duration_months": 38,
                 "source_span": "IT Specialist   Jan 2015 to Mar 2018"},
                {"title": "Admin", "company": None, "duration_months": True, "source_span": "Network"},
            ],
        }
        facts, rejected = ground("resume", raw, RESUME)

        self.assertEqual([s["name"] for s in facts["skills"]], ["Python"])
        skill = facts["skills"][0]
        self.assertEqual(RESUME[skill["start"]:skill["end"]], skill["source_span"])
        self.assertEqual(facts["degrees"][0]["level"], "bachelors")
        self.assertEqual(facts["experience"][0]["duration_months"], 38)

        reasons = sorted(r["reason"] for r in rejected)
        self.assertEqual(reasons, sorted([
            UNSUPPORTED, "skill name missing or too long",
            "degree level not in ontology", "duration_months not a plausible integer",
        ]))

    def test_posting(self):
        raw = {
            "required_skills": [
                {"name": "SQL", "importance": "required", "source_span": "experience with SQL"},
                {"name": "Tableau", "importance": "preferred", "source_span": "Tableau is a plus"},
                {"name": "Excel", "importance": "must", "source_span": "SQL"},
            ],
            "required_degrees": [
                {"level": "bachelors", "field": "Statistics", "importance": "required",
                 "source_span": "Bachelor’s degree in Statistics"},
            ],
            "experience_required": {"min_years": 3, "importance": "required",
                                    "source_span": "3+ years of experience"},
        }
        facts, rejected = ground("posting", raw, POSTING)
        self.assertEqual([s["name"] for s in facts["required_skills"]], ["SQL", "Tableau"])
        self.assertEqual(facts["required_degrees"][0]["source_span"], "Bachelor's degree in Statistics")
        self.assertEqual(facts["experience_required"]["min_years"], 3)
        self.assertEqual([r["reason"] for r in rejected], ["importance not required/preferred"])

    def test_posting_without_experience(self):
        raw = {"required_skills": [], "required_degrees": [], "experience_required": None}
        facts, rejected = ground("posting", raw, POSTING)
        self.assertIsNone(facts["experience_required"])
        self.assertEqual(rejected, [])

    def test_bad_shapes(self):
        facts, rejected = ground("resume", {"skills": "Python", "degrees": None}, RESUME)
        self.assertEqual(facts, {"skills": [], "degrees": [], "experience": []})
        self.assertEqual(rejected[0]["reason"], "expected a list")
        with self.assertRaises(ValueError):
            ground("resume", ["not", "a", "dict"], RESUME)
        with self.assertRaises(ValueError):
            ground("cover_letter", {}, RESUME)

    def test_min_years_bool(self):
        raw = {"experience_required": {"min_years": True, "importance": "required", "source_span": "3+ years"}}
        facts, rejected = ground("posting", raw, POSTING)
        self.assertIsNone(facts["experience_required"])
        self.assertEqual(rejected[0]["reason"], "min_years not a plausible number")


class ParseJsonTest(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(parse_json('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json('<think>\nreasoning {x}\n</think>\n{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json('Here you go: {"a": 1} hope it helps'), {"a": 1})
        # raw tab and newline inside a string, as Qwen sometimes returns
        self.assertEqual(parse_json('{"a": "x\ty\nz"}'), {"a": "x\ty\nz"})

    def test_garbage(self):
        with self.assertRaises(ValueError):
            parse_json("no json here")
        with self.assertRaises(ValueError):
            parse_json(None)


def resume_reply(**overrides):
    body = {"skills": [{"name": "SQL", "source_span": "SQL"}], "degrees": [], "experience": []}
    body.update(overrides)
    return json.dumps(body)


class ExtractorTest(unittest.TestCase):
    def test_constrained_request(self):
        client = FakeClient([resume_reply()])
        rec = Extractor(client, "m").extract("resume", 7, RESUME)
        self.assertEqual(rec["status"], "ok")
        self.assertTrue(rec["constrained"])
        self.assertEqual(rec["doc_id"], "7")
        fmt = client.calls[0]["response_format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertIn("skills", fmt["json_schema"]["schema"]["properties"])

    def test_falls_back_when_schema_rejected(self):
        client = FakeClient([resume_reply(), resume_reply()], reject_schema=True)
        ex = Extractor(client, "m")
        self.assertEqual(ex.extract("resume", 1, RESUME)["status"], "ok")
        rec = ex.extract("resume", 2, RESUME)
        self.assertFalse(rec["constrained"])
        # one rejected constrained call, then plain calls only
        self.assertEqual(len(client.calls), 3)
        self.assertNotIn("response_format", client.calls[2])

    def test_retries_transient_errors(self):
        client = FakeClient([FakeError(429), FakeError(503), resume_reply()])
        rec = Extractor(client, "m", backoff=0).extract("resume", 1, RESUME)
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(len(client.calls), 3)

    def test_gives_up(self):
        client = FakeClient([FakeError(500)] * 3)
        rec = Extractor(client, "m", retries=2, backoff=0).extract("resume", 1, RESUME)
        self.assertEqual(rec["status"], "error")
        self.assertIn("HTTP 500", rec["error"])

    def test_auth_error_not_retried(self):
        client = FakeClient([FakeError(401), resume_reply()])
        rec = Extractor(client, "m", backoff=0).extract("resume", 1, RESUME)
        self.assertEqual(rec["status"], "error")
        self.assertEqual(len(client.calls), 1)

    def test_truncated_reply(self):
        client = FakeClient([('{"skills": [', "length")])
        rec = Extractor(client, "m").extract("resume", 1, RESUME)
        self.assertEqual(rec["status"], "error")
        self.assertIn("max_tokens", rec["error"])

    def test_empty_document(self):
        client = FakeClient([])
        rec = Extractor(client, "m").extract("resume", 1, "   ")
        self.assertEqual(rec["status"], "error")
        self.assertEqual(client.calls, [])

    def test_truncation_keeps_offsets_valid(self):
        client = FakeClient([resume_reply(skills=[
            {"name": "Python", "source_span": "Python"},
            {"name": "University of Arizona", "source_span": "University of Arizona"},
        ])])
        rec = Extractor(client, "m", max_chars=200).extract("resume", 1, RESUME)
        self.assertTrue(rec["truncated"])
        self.assertEqual(rec["source_chars"], 200)
        self.assertEqual([s["name"] for s in rec["facts"]["skills"]], ["Python"])
        s = rec["facts"]["skills"][0]
        self.assertEqual(RESUME[s["start"]:s["end"]], "Python")

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            Extractor(FakeClient([]), "m").extract("letter", 1, RESUME)

    def test_records_run_settings(self):
        client = FakeClient([resume_reply()])
        rec = Extractor(client, "m", seed=7).extract("resume", 1, RESUME)
        self.assertEqual(client.calls[0]["seed"], 7)
        self.assertEqual(rec["config"]["seed"], 7)
        self.assertEqual(rec["config"]["temperature"], 0.0)
        self.assertEqual(len(rec["config"]["prompt"]), 8)
        self.assertEqual(len(rec["source_sha"]), 12)
        plain = FakeClient([resume_reply()])
        Extractor(plain, "m").extract("resume", 1, RESUME)
        self.assertNotIn("seed", plain.calls[0])


class BatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "out", "resumes.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_after_interruption(self):
        docs = [(1, RESUME), (2, RESUME), (3, RESUME)]
        first = FakeClient([resume_reply(), FakeError(401)])
        counts = run_batch(Extractor(first, "m"), "resume", docs, self.path, limit=2, log_every=0)
        self.assertEqual(counts, {"ok": 1, "error": 1, "skipped": 0})

        # simulate a run killed mid-write
        with open(self.path, "a") as f:
            f.write('{"doc_id": "3", "sta')

        self.assertEqual(done_ids(self.path), {"1"})
        second = FakeClient([resume_reply(), resume_reply()])
        counts = run_batch(Extractor(second, "m"), "resume", docs, self.path, log_every=0)
        self.assertEqual(counts, {"ok": 2, "error": 0, "skipped": 1})

        records = load_records(self.path)
        self.assertEqual(sorted(r["doc_id"] for r in records), ["1", "2", "3"])
        self.assertTrue(all(r["status"] == "ok" for r in records))

    def test_summarize(self):
        records = [
            {"status": "ok", "truncated": True,
             "facts": {"skills": [{"name": "SQL"}], "degrees": [], "experience": []},
             "rejected": [{"field": "skills", "reason": UNSUPPORTED},
                          {"field": "skills", "reason": "skill name missing or too long"}]},
            {"status": "ok", "truncated": False,
             "facts": {"required_skills": [], "required_degrees": [], "experience_required": {"min_years": 2}},
             "rejected": []},
            {"status": "error", "error": "x"},
        ]
        s = summarize(records)
        self.assertEqual((s["documents"], s["errors"], s["truncated"]), (3, 1, 1))
        self.assertEqual(s["fields"]["skills"],
                         {"kept": 1, "unsupported": 1, "invalid": 1, "unsupported_rate": 0.3333})
        self.assertEqual(s["fields"]["experience_required"]["kept"], 1)
        self.assertIsNone(s["fields"]["degrees"]["unsupported_rate"])


if __name__ == "__main__":
    unittest.main()
