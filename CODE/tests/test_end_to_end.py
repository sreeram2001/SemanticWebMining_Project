"""Offline unit and end-to-end tests.

    cd CODE && python -m unittest tests.test_end_to_end
"""

import json
import os
import shutil
import tempfile
import unittest

from linking.entity_linking import NIL, Gazetteer, SkillIndex, SkillLinker, TextSearch, evaluate_linking
from assistant.explanations import build, explain, verify
from extract.extraction import ground
from kg.graph import KnowledgeGraph
from dataprep.preprocess import clean_postings, mask_pii, make_split, save_split
from rank.ranking import METHODS, Ranker
from rank.ranking_eval import (auto_qrels, bootstrap_ci, ndcg_at_k, precision_at_k, randomization_test)
from rank.retrieval import BM25, rrf
from extract.rule_extractor import RuleExtractor, years_from_dates

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # CODE/

SKILLS = [
    {"id": "K1", "name": "Python (Programming Language)", "category": "IT", "subcategory": "Languages"},
    {"id": "K2", "name": "Java (Programming Language)", "category": "IT", "subcategory": "Languages",
     "description": "Object-oriented programming language for enterprise software."},
    {"id": "K3", "name": "SQL (Programming Language)", "category": "IT", "subcategory": "Databases"},
    {"id": "K4", "name": "PostgreSQL", "category": "IT", "subcategory": "Databases"},
    {"id": "K5", "name": "Deep Learning", "category": "IT", "subcategory": "ML"},
    {"id": "K6", "name": "PyTorch (Machine Learning Library)", "category": "IT", "subcategory": "ML"},
    {"id": "K7", "name": "Kubernetes", "category": "IT", "subcategory": "DevOps"},
    {"id": "K8", "name": "Docker (Software)", "category": "IT", "subcategory": "DevOps"},
    {"id": "K9", "name": "Accounts Payable", "category": "Finance", "subcategory": "Accounting"},
    {"id": "K10", "name": "Java (Coffee)", "category": "Hospitality", "subcategory": "Food",
     "description": "Brewed coffee beverage served by a barista."},
    {"id": "K11", "name": "Tableau (Business Intelligence Software)", "category": "IT", "subcategory": "BI"},
]
ALIASES = {"postgres": "PostgreSQL", "k8s": "Kubernetes", "pytorch": "PyTorch (Machine Learning Library)",
           "ap": "Accounts Payable", "missing": "Not In Taxonomy"}

CFG = {
    "retrieval": {"query": "skills", "first_stage_k": 50, "rrf_k": 60, "dense_model": None, "hybrid_weight": 0.5},
    "rerank": {"model": None, "top_n": 10},
    "scoring": {"beta_ce": 0.6, "beta_graph": 0.4, "beta_penalty": 0.3, "lambda_sub": 0.6, "lambda_cat": 0.3,
                "w_required": 1.0, "w_preferred": 0.5, "fuzzy_threshold": 0.9, "current_date": "2021-12"},
}

POSTING = ("Data Scientist. Requirements: 3+ years of experience. Deep Learning and SQL are required. "
           "Python is required. Tableau is a plus. Master's degree in Computer Science.")
RESUME = ("   DATA SCIENTIST   Skills   PyTorch, PostgreSQL, Python   Experience   ML Engineer   "
          "Jan 2020 to Current   Company Name   Education   Bachelor of Science : Statistics   2019")
RESUME_MISS = "   ENGINEER   Skills   Docker, Kubernets   Experience   Engineer   Jan 2015 to Dec 2020   Company Name"
POSTING_K8S = "Platform Engineer. Must have Docker and Kubernetes. 2+ years of experience."


def setup_graph(records):
    idx = SkillIndex(SKILLS, ALIASES)
    g = KnowledgeGraph(idx, SkillLinker(idx), KnowledgeGraph.load_ontology(os.path.join(HERE, "resources", "ontology.json")))
    for rec, text, meta in records:
        g.add_record(rec, text, meta)
    g.add_related_edges(0.6)
    return idx, g


class TestDataPrep(unittest.TestCase):
    def test_mask_pii(self):
        t = mask_pii("Mail me at a.b@x.com or (480) 555-0101, 120 Mill Avenue, see https://x.io. Skills: SQL")
        for leak in ("a.b@x.com", "555-0101", "Mill Avenue", "https://x.io"):
            self.assertNotIn(leak, t)
        self.assertIn("Skills: SQL", t)

    def test_reposts_removed(self):
        import pandas as pd
        desc = "We need a data engineer with SQL and Python skills for our team in Phoenix. " * 2
        df = pd.DataFrame([
            {"job_id": 1, "company_name": "A", "title": "DE", "description": desc, "min_salary": None, "max_salary": None},
            {"job_id": 2, "company_name": "A", "title": "DE", "description": desc.replace("Phoenix", "Phoenix "),
             "min_salary": None, "max_salary": None},
            {"job_id": 3, "company_name": "A", "title": "DE", "description": "too short", "min_salary": None,
             "max_salary": None}])
        clean, report = clean_postings(df)
        self.assertEqual(list(clean["job_id"]), [1])
        self.assertEqual(report["clean"], 1)

    def test_split_is_frozen_and_stratified(self):
        res = [(str(i), "A" if i < 6 else "B") for i in range(10)]
        s = make_split(res, 0.5, 1)
        self.assertEqual(s, make_split(res, 0.5, 1))
        self.assertEqual(sum(v == "dev" for v in s.values()), 5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "split.json")
            self.assertTrue(save_split(s, path, 1, 0.5))
            self.assertFalse(save_split({"x": "dev"}, path, 1, 0.5))   # never overwritten

    def test_years_from_dates_merges_overlaps(self):
        text = "Analyst Jan 2015 to Dec 2016  Lead Jun 2016 to Dec 2017  Intern 2010 to 2010"
        self.assertEqual(years_from_dates(text), round(37 / 12, 2))   # 36 merged months + "2010 to 2010"
        self.assertIsNone(years_from_dates("no dates here"))


class TestRuleExtractor(unittest.TestCase):
    def setUp(self):
        self.idx = SkillIndex(SKILLS, ALIASES)
        self.ex = RuleExtractor(Gazetteer(self.idx))

    def test_posting_record_matches_team_format(self):
        rec = self.ex.extract("posting", "J1", POSTING)
        self.assertEqual(rec["status"], "ok")
        f = rec["facts"]
        imp = {s["name"]: s["importance"] for s in f["required_skills"]}
        self.assertEqual(imp, {"Deep Learning": "required", "SQL": "required", "Python": "required",
                               "Tableau": "preferred"})
        self.assertEqual(f["experience_required"]["min_years"], 3)
        self.assertEqual(f["required_degrees"][0]["level"], "masters")
        for s in f["required_skills"]:
            self.assertEqual(POSTING[s["start"]:s["end"]], s["source_span"])
        # passes grounding unchanged
        again, rejected = ground("posting", f, POSTING)
        self.assertEqual(rejected, [])

    def test_resume(self):
        rec = self.ex.extract("resume", "R1", RESUME)
        names = {s["name"] for s in rec["facts"]["skills"]}
        self.assertEqual(names, {"PyTorch", "PostgreSQL", "Python"})
        self.assertEqual(rec["facts"]["degrees"][0]["level"], "bachelors")
        self.assertEqual(rec["facts"]["experience"][0]["title"], "ML Engineer")

    def test_short_names_need_capitals(self):
        hits = Gazetteer(SkillIndex(SKILLS, ALIASES)).find("we ap ply ... AP clerk")
        self.assertEqual([h[2] for h in hits], [["K9"]])


class TestLinking(unittest.TestCase):
    def setUp(self):
        self.idx = SkillIndex(SKILLS, ALIASES)
        self.linker = SkillLinker(self.idx)

    def test_alias_fast_path(self):
        self.assertEqual(self.linker.link("Postgres")["skill_id"], "K4")
        self.assertEqual(self.linker.link("Postgre SQL")["skill_id"], "K4")
        self.assertEqual(self.linker.link("K8s")["method"], "alias")

    def test_alias_with_target_outside_taxonomy_is_ignored(self):
        self.assertEqual(self.idx.lookup("missing"), [])

    def test_ambiguous_name_uses_context(self):
        self.assertEqual(self.linker.link("Java", "enterprise software in Java")["skill_id"], "K2")
        self.assertEqual(self.linker.link("Java", "barista brewing coffee")["skill_id"], "K10")

    def test_variant_and_nil(self):
        self.assertEqual(self.linker.link("Kubernets")["skill_id"], "K7")
        self.assertEqual(self.linker.link("Snowflake")["skill_id"], NIL)

    def test_alias_only_baseline(self):
        rows = [{"mention": "Postgres", "gold_skill_id": "K4"}, {"mention": "Kubernets", "gold_skill_id": "K7"},
                {"mention": "Snowflake", "gold_skill_id": NIL}]
        base = evaluate_linking(SkillLinker(self.idx, alias_only=True), rows)
        ours = evaluate_linking(self.linker, rows)
        self.assertAlmostEqual(base["accuracy@1"], 2 / 3)
        self.assertEqual(ours["accuracy@1"], 1.0)

    def test_text_search_exact_and_fuzzy(self):
        ts = TextSearch(RESUME_MISS, self.idx)
        self.assertEqual(ts.find("K8")["how"], "exact")
        hit = ts.find("K7")
        self.assertEqual(hit["text"], "Kubernets")
        self.assertTrue(hit["how"].startswith("fuzzy"))
        self.assertIsNone(ts.find("K7", fuzzy_threshold=1.0))


class TestGraphAndScoring(unittest.TestCase):
    def setUp(self):
        idx = SkillIndex(SKILLS, ALIASES)
        ex = RuleExtractor(Gazetteer(idx))
        recs = [(ex.extract("posting", "J1", POSTING), POSTING, {"title": "Data Scientist", "company_name": "Acme"}),
                (ex.extract("posting", "J2", POSTING_K8S), POSTING_K8S, {"title": "Platform Engineer",
                                                                         "company_name": "Acme"}),
                (ex.extract("resume", "R1", RESUME), RESUME, {"category": "INFORMATION-TECHNOLOGY"})]
        # R2: Kubernetes missed (typo)
        miss = ex.extract("resume", "R2", RESUME_MISS)
        recs.append((miss, RESUME_MISS, {"category": "INFORMATION-TECHNOLOGY"}))
        self.idx, self.g = setup_graph(recs)
        self.r = Ranker(self.g, CFG, use_models=False)

    def test_graph_follows_ontology(self):
        self.assertEqual(self.g.integrity_errors(), [])
        st = self.g.stats()["edges"]
        for rel in ("REQUIRES_SKILL", "HAS_SKILL", "POSTED_BY", "HAS_REQUIREMENT", "REQUIRES_DEGREE",
                    "HAS_DEGREE", "HAS_EXPERIENCE", "RELATED_TO"):
            self.assertIn(rel, st)

    def test_tampered_span_is_caught(self):
        src, rel, dst, props = self.g.edges[[e[1] for e in self.g.edges].index("HAS_SKILL")]
        props["source_span"] = "Haskell"
        self.assertTrue(any("source_span" in e for e in self.g.integrity_errors()))

    def test_partial_credit_matches_slide(self):
        m = self.r.match("R1", "J1")
        by = {x["name"]: x for x in m["skills"]}
        self.assertEqual(by["Deep Learning"]["status"], "related")          # PyTorch, same subcategory
        self.assertEqual(by["Deep Learning"]["credit"], 0.6)
        self.assertEqual(by["SQL"]["related_skill"], "PostgreSQL")
        self.assertEqual(by["Python"]["status"], "met")
        self.assertEqual((by["Tableau"]["relation"], by["Tableau"]["credit"]), ("category", 0.3))
        # G = (1*0.6 + 1*0.6 + 1*1 + 0.5*0.3) / (1 + 1 + 1 + 0.5)
        self.assertAlmostEqual(m["G"], 2.35 / 3.5)
        acct = self.r.match("R1", "J2")
        self.assertTrue(all(x["status"] != "not_found" or x["credit"] == 0 for x in acct["skills"]))

    def test_explicit_conflicts_only(self):
        m = self.r.match("R1", "J1")
        hard = {h["kind"]: h for h in m["hard"]}
        self.assertEqual(hard["degree"]["status"], "conflict")             # bachelors < masters
        self.assertEqual(hard["experience"]["status"], "conflict")         # 2 < 3 years
        self.assertGreater(m["P"], 0)

    def test_missing_info_is_unknown_not_penalized(self):
        rec = RuleExtractor(Gazetteer(self.idx)).extract("resume", "R3", "Skills: Python")
        self.g.add_record(rec, "Skills: Python", {"category": "INFORMATION-TECHNOLOGY"})
        m = self.r.match("R3", "J1")
        self.assertTrue(all(h["status"] == "unknown" for h in m["hard"]))
        self.assertEqual(m["P"], 0.0)

    def test_missed_skill_recovered_from_text(self):
        m = self.r.match("R2", "J2")
        k8s = next(x for x in m["skills"] if x["skill_id"] == "K7")
        self.assertEqual(k8s["status"], "recovered")
        self.assertEqual(k8s["credit"], 1.0)
        self.assertIn(("R2", "K7"), self.r.miss_log)

    def test_every_method_ranks_every_job(self):
        for method in METHODS:
            ranked = self.r.rank("R1", method)
            self.assertEqual(sorted(x["job_id"] for x in ranked), ["J1", "J2"], method)

    def test_ablation_removes_only_the_graph_term(self):
        full = {x["job_id"]: x for x in self.r.rank("R1", "full")}
        nog = {x["job_id"]: x for x in self.r.rank("R1", "full_no_graph")}
        for j in full:
            self.assertAlmostEqual(full[j]["score"] - nog[j]["score"], 0.4 * full[j]["G"])

    def test_explanation_verified(self):
        e = explain(self.r, "R1", "J1")
        self.assertTrue(e["verified"], e["unsupported_skills"])
        self.assertIn('"PyTorch"', e["text"])
        text, allowed, ignore = build(self.g, self.r.match("R1", "J1"))
        self.assertEqual(verify(text + "\nYou also know Kubernetes.", allowed, Gazetteer(self.idx), ignore), ["K7"])

    def test_cypher_export(self):
        with tempfile.TemporaryDirectory() as d:
            n = self.g.to_cypher(os.path.join(d, "g.cypher"))
            self.assertGreater(n, len(self.g.nodes))


class TestRetrievalAndMetrics(unittest.TestCase):
    def test_bm25_prefers_matching_doc(self):
        s = BM25(["python sql spark", "accounts payable excel"]).scores("python spark")
        self.assertGreater(s[0], s[1])

    def test_rrf(self):
        s = rrf([["a", "b"], ["b"]], k=60)
        self.assertAlmostEqual(s["b"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(s["a"], 1 / 61)

    def test_metrics(self):
        g = {"a": 2, "b": 1}
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "c"], g, 3), 1.0)
        self.assertLess(ndcg_at_k(["c", "b", "a"], g, 3), 1.0)
        self.assertEqual(precision_at_k(["a", "c"], g, 2), 0.5)

    def test_bootstrap_and_randomization(self):
        lo, hi = bootstrap_ci([0.2, 0.4, 0.6, 0.8], n=500)
        self.assertTrue(lo <= 0.5 <= hi)
        self.assertEqual(randomization_test([0.5] * 5, [0.5] * 5), 1.0)
        self.assertLess(randomization_test([1.0] * 12, [0.0] * 12, n=2000), 0.01)

    def test_auto_labels(self):
        q = auto_qrels({"r1": "ACCOUNTANT", "r2": "HR"},
                       {"j1": "Staff Accountant", "j2": "Data Engineer", "j3": "Technical Recruiter"})
        self.assertEqual(q, {"r1": {"j1": 1}, "r2": {"j3": 1}})


class TestRetrievalPort(unittest.TestCase):
    def test_top_k_matches_full_sort(self):
        import random
        from rank.retrieval import order, top_k
        rng = random.Random(0)
        for _ in range(200):
            n = rng.randint(1, 60)
            scores = [rng.choice([0.0, 0.5, 1.0, rng.random()]) for _ in range(n)]   # many ties
            ids = [f"j{i}" for i in range(n)]
            for k in (1, 5, n, n + 3):
                self.assertEqual(top_k(scores, ids, k), order(scores, ids)[:k])

    def test_posting_text_weights_title(self):
        from rank.retrieval import posting_text
        self.assertEqual(posting_text("Nurse", "Care for patients."), "Nurse. Nurse. Care for patients.")
        self.assertEqual(posting_text(None, "Care for patients."), "Care for patients.")
        bm = BM25([posting_text("Data Engineer", "Build pipelines."), posting_text("Nurse", "Data entry.")])
        s = bm.scores("data engineer")
        self.assertGreater(s[0], s[1])


class TestSamplePipeline(unittest.TestCase):
    """The whole pipeline on CODE/sample_data, offline."""

    def test_end_to_end(self):
        from core.config import load_config
        from core.pipeline import Pipeline
        out = tempfile.mkdtemp()
        try:
            cfg = load_config(os.path.join(HERE, "sample_data", "config_sample.json"), {
                "paths": {"extracted_postings": os.path.join(out, "p.jsonl"),
                          "extracted_resumes": os.path.join(out, "r.jsonl"),
                          "split": os.path.join(out, "split.json"), "out_dir": out},
                "evaluation": {"bootstrap": 50, "permutations": 50}})
            p = Pipeline(cfg, use_models=False)
            self.assertEqual(len(p.split()), 20)
            ext = p.extract()
            self.assertEqual(ext["posting"]["ok"], 26)
            info = p.build()
            self.assertEqual(info["integrity_errors"], 0)
            rec = p.recommend("2002")
            self.assertEqual(rec["jobs"][0]["title"], "Data Scientist")
            self.assertTrue(rec["explanation_top1"]["verified"])
            res = p.evaluate("dev")
            for tier in ("ranking_auto", "ranking_graded"):
                self.assertEqual(set(res[tier]["summary"]), set(METHODS))
            self.assertIn("linking", res)
            self.assertTrue(os.path.exists(os.path.join(out, "eval_dev.md")))
        finally:
            shutil.rmtree(out)


class TestRobustness(unittest.TestCase):
    def test_config_validation(self):
        from core.config import ConfigError, load_config
        load_config()                                                      # the shipped config is valid
        for bad in ({"scoring": {"lambda_cat": 0.7}}, {"retrieval": {"query": "resume"}},
                    {"extraction": {"mode": "gpt"}}, {"scorng": {}}, {"scoring": {"current_date": "Dec 2021"}}):
            with self.assertRaises(ConfigError, msg=str(bad)):
                load_config(overrides=bad)

    def test_ids_and_text(self):
        from core.config import as_text, norm_id
        self.assertEqual([norm_id(x) for x in (3884428798.0, " 12 ", "3884428798.0", "R1.0", None)],
                         ["3884428798", "12", "3884428798", "R1.0", ""])
        self.assertEqual(as_text(float("nan")), "")

    def test_atomic_write_and_corrupt_cache(self):
        from core.config import read_json_or, write_json_atomic
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            write_json_atomic(path, {"a": 1})
            self.assertEqual(read_json_or(path, {}), {"a": 1})
            with open(path, "w") as f:
                f.write('{"a": 1, "b"')                                     # a write cut off by a crash
            self.assertEqual(read_json_or(path, {}), {})
            self.assertTrue(os.path.exists(path + ".corrupt"))
            self.assertEqual(os.listdir(d), ["c.json.corrupt"])             # no temp files left behind

    def test_linker_survives_corrupt_llm_cache(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "llm.json")
            with open(path, "w") as f:
                f.write("not json")
            lk = SkillLinker(SkillIndex(SKILLS, ALIASES), cache_path=path)
            self.assertEqual(lk.link("Postgres")["skill_id"], "K4")

    def test_stale_facts_are_dropped(self):
        idx = SkillIndex(SKILLS, ALIASES)
        rec = RuleExtractor(Gazetteer(idx)).extract("resume", "R1", RESUME)
        g = KnowledgeGraph(idx, SkillLinker(idx), KnowledgeGraph.load_ontology(os.path.join(HERE, "resources", "ontology.json")))
        g.add_record(rec, "   EDITED " + RESUME, {})                        # text changed after extraction
        self.assertGreater(g.stats()["stale_facts_dropped"], 0)
        self.assertEqual(g.integrity_errors(), [])

    def test_cached_match_equals_fresh_match(self):
        idx = SkillIndex(SKILLS, ALIASES)
        ex = RuleExtractor(Gazetteer(idx))
        _, g = setup_graph([(ex.extract("posting", "J1", POSTING), POSTING, {}),
                            (ex.extract("resume", "R1", RESUME), RESUME, {})])
        r = Ranker(g, CFG, use_models=False)
        a = r.match("R1", "J1", {"lambda_sub": 0.7, "lambda_cat": 0.2})
        r2 = Ranker(g, CFG, use_models=False)
        r2.match("R1", "J1")                                                # fills the cache with other lambdas
        b = r2.match("R1", "J1", {"lambda_sub": 0.7, "lambda_cat": 0.2})
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            r.match("R1", "J1", {"lambda_sub": 0.2, "lambda_cat": 0.3})

    def test_cli_reports_missing_inputs_cleanly(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as d:
            cfg = os.path.join(d, "c.json")
            with open(cfg, "w") as f:
                json.dump({"paths": {"postings": os.path.join(d, "none.csv")}}, f)
            out = subprocess.run([sys.executable, os.path.join(HERE, "run_pipeline.py"), "--config", cfg, "split"],
                                 capture_output=True, text=True)
        self.assertEqual(out.returncode, 1)
        self.assertIn("error: missing", out.stderr)
        self.assertNotIn("Traceback", out.stderr)


if __name__ == "__main__":
    unittest.main()
