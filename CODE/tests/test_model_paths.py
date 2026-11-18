"""Model, Neo4j and LLM call paths, tested with injected stand-in modules.

    cd CODE && python -m unittest tests.test_model_paths
"""

import json
import os
import sys
import tempfile
import types
import unittest
import zlib
from types import SimpleNamespace

import numpy as np

from linking.entity_linking import NIL, SkillIndex, SkillLinker, Gazetteer
from kg.graph import KnowledgeGraph
from rank.ranking import CrossEncoderReranker, Ranker
from extract.rule_extractor import RuleExtractor
from tests.test_end_to_end import ALIASES, CFG, HERE, POSTING, POSTING_K8S, RESUME, SKILLS

DIM = 128


def _bow(text):
    v = np.zeros(DIM)
    for w in str(text).lower().replace(":", " ").replace("|", " ").split():
        v[zlib.crc32(w.encode()) % DIM] += 1.0
    return v


class FakeSentenceTransformer:
    encoded = 0

    def __init__(self, name):
        if name == "missing/model":
            raise OSError("model not found")
        self.name = name

    def encode(self, texts, normalize_embeddings=False, batch_size=32):
        FakeSentenceTransformer.encoded += len(texts)
        m = np.array([_bow(t) for t in texts])
        if normalize_embeddings:
            m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
        return m


class FakeCrossEncoder:
    def __init__(self, name, max_length=512):
        self.name = name

    def predict(self, pairs):
        return np.array([float(len(set(a.lower().split()) & set(b.lower().split()))) - 1.0 for a, b in pairs])


def install_fake_sentence_transformers():
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer, mod.CrossEncoder = FakeSentenceTransformer, FakeCrossEncoder
    sys.modules["sentence_transformers"] = mod


class FakeLLM:
    """Fake OpenAI client: fixed mention -> skill id."""

    def __init__(self, answers):
        self.answers, self.calls = answers, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, **kw):
        self.calls += 1
        prompt = messages[-1]["content"]
        mention = prompt.split("Skill mention: ", 1)[1].split("\n", 1)[0]
        content = "<think>…</think>" + json.dumps({"skill_id": self.answers.get(mention, NIL)})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeNeo4j:
    """Records statements sent to Neo4j."""
    statements = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, query, **params):
            FakeNeo4j.statements.append((query, params))

    class _Driver(_Session):
        def session(self):
            return FakeNeo4j._Session()

    @staticmethod
    def driver(uri, auth):
        return FakeNeo4j._Driver()


class TestModelPaths(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k) for k in ("sentence_transformers", "neo4j")}
        install_fake_sentence_transformers()
        self.idx = SkillIndex(SKILLS, ALIASES)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_bi_encoder_embeddings_are_cached_on_disk(self):
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "emb.npy")
            FakeSentenceTransformer.encoded = 0
            a = SkillLinker(self.idx, dense_model="fake/minilm", emb_cache=cache)
            self.assertIn("bi-encoder", a.backend)
            self.assertEqual(FakeSentenceTransformer.encoded, len(SKILLS))
            SkillLinker(self.idx, dense_model="fake/minilm", emb_cache=cache)
            self.assertEqual(FakeSentenceTransformer.encoded, len(SKILLS))      # second load reads the .npy
            SkillLinker(self.idx, dense_model="fake/other", emb_cache=cache)
            self.assertEqual(FakeSentenceTransformer.encoded, 2 * len(SKILLS))  # different model: re-embedded

    def test_posting_embeddings_are_cached_on_disk(self):
        from rank.retrieval import Dense
        docs, ids = ["python developer", "nurse", "sql analyst"], ["1", "2", "3"]
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "postings.npz")
            FakeSentenceTransformer.encoded = 0
            a = Dense(docs, "fake/minilm", ids=ids, cache=cache)
            self.assertEqual(FakeSentenceTransformer.encoded, 3)
            b = Dense(docs, "fake/minilm", ids=ids, cache=cache)
            self.assertEqual(FakeSentenceTransformer.encoded, 3)                # read from the .npz
            self.assertEqual(a.scores("python"), b.scores("python"))           # cached = fresh
            FakeSentenceTransformer.encoded = 0                                 # (queries were encoded too)
            Dense(docs[:2] + ["tax accountant"], "fake/minilm", ids=ids, cache=cache)
            self.assertEqual(FakeSentenceTransformer.encoded, 3)                # postings changed: re-embedded
            Dense(docs, "fake/other", ids=ids, cache=cache)
            self.assertEqual(FakeSentenceTransformer.encoded, 6)                # model changed: re-embedded

    def test_strict_mode_raises_instead_of_falling_back(self):
        self.assertIsNone(SkillLinker(self.idx, dense_model="missing/model")._dense)
        with self.assertRaises(RuntimeError):
            SkillLinker(self.idx, dense_model="missing/model", strict=True)

    def test_llm_band_and_cache(self):
        llm = FakeLLM({"Kubernets": "K7", "Snowflake": NIL})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "llm.json")
            lk = SkillLinker(self.idx, llm_client=llm, llm_model="fake-qwen", llm_accept=0.95, llm_reject=0.05,
                             cache_path=path)
            self.assertEqual(lk.link("Postgres")["method"], "alias")          # alias hit: no LLM
            self.assertEqual(lk.link("Kubernets", "Docker, Kubernets")["skill_id"], "K7")
            lk.link("Kubernets")                                               # same mention: cached
            self.assertEqual(llm.calls, 1)
            lk.save_cache()
            again = SkillLinker(self.idx, llm_client=llm, llm_model="fake-qwen", llm_accept=0.95,
                                llm_reject=0.05, cache_path=path)
            self.assertEqual(again.link("Kubernets")["skill_id"], "K7")
            self.assertEqual(llm.calls, 1)                                     # answered from disk
            # outside the band the LLM is not asked
            lk.set_params(llm_reject=0.99)
            self.assertEqual(lk.link("Kubernetes cluster ops")["skill_id"], NIL)
            self.assertEqual(llm.calls, 1)

    def test_llm_answer_outside_candidates_is_nil(self):
        lk = SkillLinker(self.idx, llm_client=FakeLLM({"Kubernets": "K1-not-a-candidate"}), llm_model="m",
                         llm_accept=1.1, llm_reject=0.0)
        self.assertEqual(lk.link("Kubernets")["skill_id"], NIL)

    def _graph(self):
        ex = RuleExtractor(Gazetteer(self.idx))
        g = KnowledgeGraph(self.idx, SkillLinker(self.idx), KnowledgeGraph.load_ontology(os.path.join(HERE, "resources", "ontology.json")))
        for kind, doc_id, text in (("posting", "J1", POSTING), ("posting", "J2", POSTING_K8S), ("resume", "R1", RESUME)):
            g.add_record(ex.extract(kind, doc_id, text), text, {"title": doc_id, "company_name": "Acme"})
        g.add_related_edges(0.6)
        return g

    def test_ranker_uses_real_model_classes(self):
        cfg = {**CFG, "retrieval": {**CFG["retrieval"], "dense_model": "fake/minilm"},
               "rerank": {"model": "fake/ms-marco", "top_n": 10}}
        r = Ranker(self._graph(), cfg, use_models=True, strict=True)
        self.assertEqual(r.dense.backend, "fake/minilm")
        self.assertIsInstance(r.reranker, CrossEncoderReranker)
        ranked = r.rank("R1", "full")
        self.assertEqual(sorted(x["job_id"] for x in ranked), ["J1", "J2"])

    def test_ranker_strict_fails_on_missing_model(self):
        cfg = {**CFG, "retrieval": {**CFG["retrieval"], "dense_model": "missing/model"}}
        with self.assertRaises(RuntimeError):
            Ranker(self._graph(), cfg, use_models=True, strict=True)

    def test_neo4j_loader_batches(self):
        sys.modules["neo4j"] = types.SimpleNamespace(GraphDatabase=FakeNeo4j)
        FakeNeo4j.statements = []
        g = self._graph()
        out = g.load_neo4j("bolt://x", "neo4j", "pw", batch=3, clear=True)
        queries = [q for q, _ in FakeNeo4j.statements]
        self.assertTrue(queries[0].startswith("MATCH (n) DETACH DELETE n"))
        self.assertTrue(any(q.startswith("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Skill)") for q in queries))
        sent_nodes = sum(len(p["rows"]) for q, p in FakeNeo4j.statements if "MERGE (n:" in q)
        sent_edges = sum(len(p["rows"]) for q, p in FakeNeo4j.statements if "MERGE (a)-[e:" in q)
        self.assertEqual((sent_nodes, sent_edges), (len(g.nodes), len(g.edges)))
        self.assertEqual(out, {"nodes": len(g.nodes), "edges": len(g.edges)})
        self.assertTrue(all(len(p["rows"]) <= 3 for _, p in FakeNeo4j.statements if "rows" in p))


if __name__ == "__main__":
    unittest.main()
