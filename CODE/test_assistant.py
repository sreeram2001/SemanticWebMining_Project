import os
import shutil
import tempfile
import unittest
import zlib

import numpy as np

import assistant as asst
import extraction as ex
import rerank as rr
import retrieval as rt


class FakeEncoder:
    """Hashed bag of words, in place of a sentence-transformers model."""

    def encode(self, texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False,
               convert_to_numpy=True):
        vecs = np.zeros((len(texts), 64), dtype="float32")
        for i, text in enumerate(texts):
            for tok in rt.tokenize(text):
                vecs[i, zlib.crc32(tok.encode()) % 64] += 1
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms


def overlap(pairs):
    return [len(set(rt.tokenize(q)) & set(rt.tokenize(d))) for q, d in pairs]


class FakeExtractor:
    """Grounds a fixed list of skill names with extraction.ground."""

    model = "fake"
    names = ["Python", "SQL", "AWS", "Apache Spark", "Kafka", "Excel"]

    def __init__(self):
        self.calls = []

    def extract(self, kind, doc_id, text):
        self.calls.append((kind, str(doc_id)))
        found = [n for n in self.names if n in text]
        if kind == "resume":
            raw = {"skills": [{"name": n, "source_span": n} for n in found], "degrees": [], "experience": []}
        else:
            raw = {"required_skills": [{"name": n, "importance": "required", "source_span": n} for n in found],
                   "required_degrees": [], "experience_required": None}
        facts, rejected = ex.ground(kind, raw, text)
        return {"doc_id": str(doc_id), "kind": kind, "model": self.model, "status": "ok",
                "facts": facts, "rejected": rejected}


def full_assistant(**kw):
    return asst.Assistant(sample=True, encoder=FakeEncoder(), reranker=rr.CrossEncoderReranker(predict=overlap),
                          use_llm=False, **kw)


UPLOAD = ("Software developer. Wrote Python services and SQL reports on AWS for five years. "
          "Bachelor of Science in Computer Science.")


class RankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = full_assistant()

    def test_all_methods_available(self):
        self.assertEqual(self.a.methods, ["bm25", "dense", "hybrid", "rrf", "rerank"])

    def test_every_method_ranks_the_data_engineer_job_first(self):
        for method in self.a.methods:
            ranked = self.a.ranked("2001", method)
            self.assertEqual([j["rank"] for j in ranked], list(range(1, len(ranked) + 1)))
            self.assertTrue(all(j["job_id"] in self.a.jobs for j in ranked))
            self.assertEqual(ranked[0]["job_id"], "1001", method)

    def test_rrf_is_the_team_fusion_of_bm25_and_dense(self):
        query = self.a.resume_text("2002")
        idx = self.a.index
        qvec = rt.encode_texts([query], model=idx.encoder)[0]
        expected = rt.rrf([idx.bm25.search(query, asst.POOL), idx.dense.search(qvec, asst.POOL)], top=asst.KEEP)
        self.assertEqual([j for j, _ in idx.rank(query, "rrf")], [j for j, _ in expected])

    def test_rerank_keeps_only_the_rerank_pool(self):
        self.assertLessEqual(len(self.a.ranked("2001", "rerank")), self.a.index.rerank_top)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            self.a.index.rank("python", "graph")


class BM25OnlyTests(unittest.TestCase):
    def test_without_models_only_bm25(self):
        a = asst.Assistant(sample=True, dense_model=None, use_llm=False)
        self.assertEqual(a.methods, ["bm25"])
        chat = asst.Chat(a, "2001")
        self.assertEqual(chat.method, "bm25")
        self.assertIn("not available", chat.reply("use dense"))
        self.assertEqual(chat.method, "bm25")


class MatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = asst.Assistant(sample=True, dense_model=None, use_llm=False)

    def rows(self, rid, job_id):
        return {r["name"]: r for r in self.a.match(rid, job_id)["skills"]}

    def test_statuses_and_evidence(self):
        rows = self.rows("2001", "1001")
        self.assertEqual(rows["Python"]["status"], "matched")
        self.assertEqual(rows["Python"]["evidence"], "Python")
        self.assertEqual(rows["Amazon Web Services"]["status"], "not_found")
        self.assertIsNone(rows["Amazon Web Services"]["evidence"])
        self.assertEqual(rows["Python"]["posting_span"], "Python")

    def test_one_row_per_lightcast_skill(self):
        ids = [r["skill_id"] for r in self.a.match("2001", "1001")["skills"] if r["skill_id"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_upload_without_facts_is_searched_as_text(self):
        rid = self.a.add_upload(UPLOAD)
        self.assertTrue(rid.startswith("upload-"))
        rows = self.rows(rid, "1001")
        self.assertEqual(rows["Python"]["status"], "found_in_text")
        self.assertIn("Python", rows["Python"]["evidence"])
        self.assertEqual(rows["Apache Spark"]["status"], "not_found")
        self.assertNotIn("matched", {r["status"] for r in rows.values()})

    def test_too_short_upload_is_refused(self):
        with self.assertRaises(ValueError):
            self.a.add_upload("python")


class OnDemandExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        src = asst.resolve(asst.SAMPLE_PATHS["extracted"])
        # postings.jsonl without job 1001; no resumes.jsonl at all
        with open(os.path.join(src, "postings.jsonl")) as f, \
                open(os.path.join(self.tmp, "postings.jsonl"), "w") as out:
            out.writelines(line for line in f if '"doc_id": "1001"' not in line)
        self.ext = FakeExtractor()
        self.a = asst.Assistant(sample=True, dense_model=None, extractor=self.ext,
                                paths={"extracted": self.tmp})

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_missing_posting_is_extracted_once_and_saved(self):
        self.assertFalse(self.a.facts.has("posting", "1001"))
        m = self.a.match("2001", "1001")
        self.assertIn(("posting", "1001"), self.ext.calls)
        self.assertIn("Python", {r["name"] for r in m["skills"]})
        self.a.match("2001", "1001")
        self.assertEqual(self.ext.calls.count(("posting", "1001")), 1)
        saved = {r["doc_id"] for r in ex.load_records(os.path.join(self.tmp, "postings.jsonl"))}
        self.assertIn("1001", saved)

    def test_corpus_resume_is_saved_but_upload_is_not(self):
        self.a.resume_facts("2001")
        rid = self.a.add_upload(UPLOAD)
        rec = self.a.resume_facts(rid)
        self.assertEqual({s["name"] for s in rec["facts"]["skills"]}, {"Python", "SQL", "AWS"})
        saved = {r["doc_id"] for r in ex.load_records(os.path.join(self.tmp, "resumes.jsonl"))}
        self.assertEqual(saved, {"2001"})

    def test_recommend_table_does_not_extract(self):
        chat = asst.Chat(self.a, "2001")
        chat.reply("show my top jobs")
        self.assertEqual(self.ext.calls, [])


class ChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = full_assistant()

    def setUp(self):
        self.chat = asst.Chat(self.a, "2001")

    def test_needs_a_resume_first(self):
        self.assertIn("Load a resume first", asst.Chat(self.a).reply("show my top jobs"))

    def test_help(self):
        self.assertEqual(self.chat.reply("help"), asst.HELP)

    def test_default_method_is_rrf(self):
        self.assertEqual(self.chat.method, "rrf")

    def test_routing(self):
        cases = {
            "show my top jobs": "**Top 10 jobs**",
            "top 3": "**Top 3 jobs**",
            "explain #2": "**#2 ",
            "why is job 1 a match?": "**#1 ",
            "what skills am I missing": "does not show",
            "compare these": "| | #1 | #2 | #3 |",
            "compare 1 and 4": "| | #1 | #4 |",
            "compare methods": "Where each ranker",
            "job 1": "(job 1001)",
            "tell me about #2": "(job ",
            "my skills": "Skills linked to Lightcast",
        }
        for message, marker in cases.items():
            self.assertIn(marker, self.chat.reply(message), message)

    def test_switch_method(self):
        answer = self.chat.reply("switch to bm25")
        self.assertEqual(self.chat.method, "bm25")
        self.assertIn("ranked by BM25", answer)
        self.chat.reply("use cross-encoder rerank")
        self.assertEqual(self.chat.method, "rerank")

    def test_out_of_range(self):
        self.assertIn("pick a number", self.chat.reply("explain #99"))
        self.assertIn("at least two", self.chat.reply("compare 2"))

    def test_explain_quotes_both_texts(self):
        answer = self.chat.explain(1)
        self.assertIn('"Python"', answer)
        self.assertIn("✓ in your skills", answer)
        self.assertIn("✗ not found", answer)

    def test_gaps_list_only_not_found_skills(self):
        answer = self.chat.gaps()
        self.assertIn("Amazon Web Services", answer)
        self.assertNotIn("| Python |", answer)

    def test_pasted_resume_is_loaded(self):
        chat = asst.Chat(self.a)
        answer = chat.reply((UPLOAD + "\n") * 5)
        self.assertTrue(chat.resume_id.startswith("upload-"))
        self.assertIn("No extracted facts", answer)


class ReadUploadTests(unittest.TestCase):
    def test_text_file(self):
        import app
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(UPLOAD)
        try:
            self.assertEqual(app.read_upload(f.name), UPLOAD)
        finally:
            os.remove(f.name)
        self.assertEqual(app.read_upload(None), "")


if __name__ == "__main__":
    unittest.main()
