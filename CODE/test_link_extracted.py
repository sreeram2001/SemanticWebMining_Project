import json
import os
import tempfile
import unittest

import graph_score as gs
import link_extracted as le

HERE = os.path.dirname(os.path.abspath(__file__))
SKILLS = os.path.join(HERE, "sample_data", "skills.json")

RESUME = {"doc_id": "r1", "kind": "resume", "status": "ok", "facts": {
    "skills": [{"name": "Accounting", "source_span": "Accounting", "start": 0, "end": 10}],
    "degrees": [], "experience": []}}
POSTING = {"doc_id": "j1", "kind": "posting", "status": "ok", "facts": {
    "required_skills": [{"name": "Accounting", "importance": "required",
                         "source_span": "Accounting", "start": 0, "end": 10}],
    "required_degrees": [], "experience_required": None}}


class LinkExtractedTests(unittest.TestCase):
    def test_end_to_end_into_graph_score(self):
        with tempfile.TemporaryDirectory() as d:
            ext, out = os.path.join(d, "ext"), os.path.join(d, "linked")
            os.makedirs(ext)
            for name, rec in (("resumes", RESUME), ("postings", POSTING)):
                with open(os.path.join(ext, f"{name}.jsonl"), "w") as f:
                    f.write(json.dumps(rec) + "\n")
            le.main(["--extracted", ext, "--out", out, "--skills", SKILLS])
            r = gs.load_linked(os.path.join(out, "resumes.jsonl"))["r1"]
            j = gs.load_linked(os.path.join(out, "postings.jsonl"))
        self.assertEqual(gs.graph_score(j["j1"], r)["score"], 1.0)
        self.assertEqual(gs.rank_candidates([("j1", 0.5)], j, r)[0]["g"], 1.0)

    def test_missing_skills_file(self):
        with self.assertRaises(SystemExit):
            le.main(["--skills", "nope.json"])


if __name__ == "__main__":
    unittest.main()
