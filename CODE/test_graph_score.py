"""Unit tests for graph_score.py (standard library only).

cd CODE && python -m unittest test_graph_score
"""

import json
import os
import tempfile
import unittest

import graph_score as gs

PY = {"id": "PY", "name": "Python", "category": "IT", "subcategory": "Programming"}
JV = {"id": "JV", "name": "Java", "category": "IT", "subcategory": "Programming"}
PG = {"id": "PG", "name": "PostgreSQL", "category": "IT", "subcategory": "Databases"}
NU = {"id": "NU", "name": "Nursing", "category": "Health", "subcategory": "Care"}
NIL = {"id": None, "name": None, "category": None, "subcategory": None}


def job(skills=(), degrees=(), exp=None):
    """Linked posting record in the extraction.run_batch shape."""
    return {
        "doc_id": "j",
        "kind": "posting",
        "status": "ok",
        "facts": {
            "required_skills": [
                {"name": l["name"] or "odd", "importance": imp, "link": l}
                for l, imp in skills
            ],
            "required_degrees": [
                {"level": lv, "importance": imp} for lv, imp in degrees
            ],
            "experience_required": exp,
        },
    }


def resume(skills=(), degrees=(), months=()):
    return {
        "doc_id": "r",
        "kind": "resume",
        "status": "ok",
        "facts": {
            "skills": [{"name": l["name"], "link": l} for l in skills],
            "degrees": [{"level": lv} for lv in degrees],
            "experience": [{"title": "t", "duration_months": m} for m in months],
        },
    }


class GraphScoreTests(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(
            gs.graph_score(job([(PY, "required")]), resume([PY]))["score"], 1.0
        )

    def test_related_skill_partial_credit(self):
        g = gs.graph_score(job([(PY, "required")]), resume([JV]))
        self.assertAlmostEqual(g["score"], gs.DEFAULT_LAMBDAS["lam_sub"])
        self.assertEqual(g["rows"][0]["matched_by"], "Java")

    def test_same_category_credit(self):
        g = gs.graph_score(
            job([(PY, "required")]), resume([PG]), lam_sub=0.6, lam_cat=0.3
        )
        self.assertAlmostEqual(g["score"], 0.3)

    def test_required_weighs_more(self):
        g = gs.graph_score(job([(PY, "required"), (NU, "preferred")]), resume([PY]))
        self.assertAlmostEqual(g["score"], 2 / 3)

    def test_no_linked_job_skills(self):
        g = gs.graph_score(job([(NIL, "required")]), resume([PY]))
        self.assertEqual(g["score"], 0.0)
        self.assertTrue(g["insufficient_evidence"])
        self.assertEqual(g["unresolved"], ["odd"])

    def test_missing_or_failed_records(self):
        self.assertTrue(gs.graph_score(None, resume([PY]))["insufficient_evidence"])
        failed = {"status": "error", "facts": None}
        self.assertEqual(gs.graph_score(job([(PY, "required")]), failed)["score"], 0.0)

    def test_unlinked_facts_ignored(self):
        rec = resume([PY])
        rec["facts"]["skills"].append({"name": "no link yet"})
        self.assertEqual(gs.graph_score(job([(PY, "required")]), rec)["score"], 1.0)

    def test_invalid_lambdas(self):
        with self.assertRaises(ValueError):
            gs.graph_score(job(), resume(), lam_sub=0.2, lam_cat=0.4)


class PenaltyTests(unittest.TestCase):
    def test_degree_met_and_mismatch(self):
        j = job(degrees=[("masters", "required")])
        self.assertEqual(gs.penalty(j, resume(degrees=["phd"]))["score"], 0.0)
        self.assertEqual(gs.penalty(j, resume(degrees=["bachelors"]))["score"], 1.0)

    def test_lowest_accepted_degree_counts(self):
        j = job(degrees=[("bachelors", "required"), ("masters", "required")])
        self.assertEqual(gs.penalty(j, resume(degrees=["bachelors"]))["score"], 0.0)

    def test_preferred_degree_never_penalized(self):
        j = job(degrees=[("phd", "preferred")])
        p = gs.penalty(j, resume(degrees=["bachelors"]))
        self.assertEqual((p["score"], p["details"]), (0.0, []))

    def test_no_resume_degree_is_unknown(self):
        p = gs.penalty(job(degrees=[("masters", "required")]), resume())
        self.assertEqual(p["score"], 0.0)
        self.assertEqual(p["details"][0]["status"], "unknown")

    def test_experience(self):
        j = job(exp={"min_years": 3, "importance": "required"})
        self.assertEqual(gs.penalty(j, resume(months=[12, 6]))["score"], 1.0)
        self.assertEqual(gs.penalty(j, resume(months=[24, 12]))["score"], 0.0)

    def test_undated_entry_makes_years_unknown(self):
        j = job(exp={"min_years": 3, "importance": "required"})
        p = gs.penalty(j, resume(months=[6, None]))
        self.assertEqual(p["details"][0]["status"], "unknown")
        self.assertEqual(p["score"], 0.0)

    def test_preferred_experience_ignored(self):
        j = job(exp={"min_years": 10, "importance": "preferred"})
        self.assertEqual(gs.penalty(j, resume(months=[12]))["details"], [])

    def test_fraction(self):
        j = job(
            degrees=[("phd", "required")],
            exp={"min_years": 1, "importance": "required"},
        )
        self.assertAlmostEqual(
            gs.penalty(j, resume(degrees=["masters"], months=[24]))["score"], 0.5
        )

    def test_missing_job_record(self):
        self.assertEqual(gs.penalty(None, resume())["score"], 0.0)


class FinalScoreTests(unittest.TestCase):
    def test_formula_and_ablation(self):
        self.assertAlmostEqual(
            gs.final_score(0.8, 0.5, 0.2, {"b1": 1, "b2": 2, "b3": 3}), 1.2
        )
        self.assertEqual(gs.without_graph({"b1": 0.7})["b2"], 0.0)

    def test_graph_changes_order(self):
        jobs = {"jA": job([(NU, "required")]), "jB": job([(PY, "required")])}
        cands = [("jA", 0.6), ("jB", 0.5)]
        r = resume([PY])
        self.assertEqual(
            gs.rank_candidates(cands, jobs, r, {"b2": 1.0})[0]["job_id"], "jB"
        )
        self.assertEqual(
            gs.rank_candidates(cands, jobs, r, gs.without_graph())[0]["job_id"], "jA"
        )

    def test_penalty_pushes_down(self):
        jobs = {"j1": job(degrees=[("phd", "required")]), "j2": job()}
        out = gs.rank_candidates(
            [("j1", 0.9), ("j2", 0.8)], jobs, resume(degrees=["bachelors"])
        )
        self.assertEqual([r["job_id"] for r in out], ["j2", "j1"])

    def test_job_without_record(self):
        out = gs.rank_candidates([(123, 0.4)], {}, resume([PY]))
        self.assertEqual(
            (out[0]["job_id"], out[0]["g"], out[0]["p"], out[0]["has_facts"]),
            ("123", 0.0, 0.0, False),
        )

    def test_lambdas_passed_through(self):
        jobs = {"j": job([(PY, "required")])}
        out = gs.rank_candidates(
            [("j", 0.0)], jobs, resume([JV]), lambdas={"lam_sub": 0.9}
        )
        self.assertAlmostEqual(out[0]["g"], 0.9)

    def test_load_linked(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({**job(), "doc_id": 7}) + "\n")
            self.assertEqual(list(gs.load_linked(path)), ["7"])


if __name__ == "__main__":
    unittest.main()
