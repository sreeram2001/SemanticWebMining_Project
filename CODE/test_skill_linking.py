"""Unit tests for skill_linking.py (standard library only, no API key needed).

    cd CODE && python -m unittest test_skill_linking
"""

import json
import os
import tempfile
import unittest

import skill_linking as sl

RAW = [
    {"id": "KS1", "name": "Python (Programming Language)",
     "type": {"id": "ST1", "name": "Specialized Skill"},
     "category": {"id": 2, "name": "Information Technology"},
     "subcategory": {"id": 21, "name": "Programming Languages"},
     "description": "A high-level programming language."},
    {"id": "KS2", "name": "PostgreSQL",
     "type": {"id": "ST1", "name": "Specialized Skill"},
     "category": {"id": 2, "name": "Information Technology"},
     "subcategory": {"id": 22, "name": "Database Management Systems"},
     "description": None},
    {"id": "KS3", "name": "Java (Programming Language)",
     "type": {"id": "ST1", "name": "Specialized Skill"},
     "category": {"id": 2, "name": "Information Technology"},
     "subcategory": {"id": 21, "name": "Programming Languages"}},
    {"id": "KS4", "name": "Java (Coffee)",
     "type": {"id": "ST2", "name": "Common Skill"}},
    {"id": "KS5", "name": "C++ (Programming Language)",
     "type": {"id": "ST1", "name": "Specialized Skill"},
     "category": {"id": 2, "name": "Information Technology"},
     "subcategory": {"id": 21, "name": "Programming Languages"}},
]


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Replays queued responses per URL and records every call."""

    def __init__(self, get_responses, post_response=None):
        self.get_responses = {url: list(r) for url, r in get_responses.items()}
        self.post_response = post_response or FakeResponse(200, {"access_token": "tok"})
        self.calls = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append(("POST", url, data))
        return self.post_response

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(("GET", url, params))
        queue = self.get_responses.get(url)
        if not queue:
            raise ConnectionError(f"no route to {url}")
        return queue.pop(0)


NEW = sl.SKILLS_URLS[0].format(version="latest")
OLD = sl.SKILLS_URLS[1].format(version="latest")


class TestDownload(unittest.TestCase):
    def test_token_request(self):
        session = FakeSession({})
        self.assertEqual(sl.fetch_token("id", "secret", session=session), "tok")
        method, url, data = session.calls[0]
        self.assertEqual((method, url), ("POST", sl.TOKEN_URL))
        self.assertEqual(data["grant_type"], "client_credentials")
        self.assertEqual(data["scope"], "emsi_open")

    def test_full_fields_requested_first(self):
        session = FakeSession({NEW: [FakeResponse(200, {"data": RAW})]})
        raw, fields, url = sl.download_skills("tok", session=session)
        self.assertEqual(len(raw), len(RAW))
        self.assertEqual(fields, sl.FULL_FIELDS)
        self.assertEqual(url, NEW)
        self.assertEqual(session.calls[0][2], {"fields": sl.FULL_FIELDS})

    def test_falls_back_to_default_fields(self):
        session = FakeSession({NEW: [FakeResponse(400), FakeResponse(200, {"data": RAW})]})
        _, fields, _ = sl.download_skills("tok", session=session)
        self.assertEqual(fields, "default")
        self.assertEqual(session.calls[1][2], {})

    def test_falls_back_to_older_host(self):
        session = FakeSession({NEW: [FakeResponse(404)], OLD: [FakeResponse(200, {"data": RAW})]})
        _, _, url = sl.download_skills("tok", session=session)
        self.assertEqual(url, OLD)

    def test_unreachable_host_falls_back(self):
        session = FakeSession({OLD: [FakeResponse(200, {"data": RAW})]})
        _, _, url = sl.download_skills("tok", session=session)
        self.assertEqual(url, OLD)

    def test_bad_credentials_stop(self):
        session = FakeSession({NEW: [FakeResponse(401)]})
        with self.assertRaises(RuntimeError):
            sl.download_skills("tok", session=session)
        self.assertEqual(len(session.calls), 1)

    def test_retries_server_errors(self):
        session = FakeSession({NEW: [FakeResponse(503), FakeResponse(200, {"data": RAW})]})
        raw, _, _ = sl.download_skills("tok", session=session, backoff=0)
        self.assertEqual(len(raw), len(RAW))

    def test_download_and_cache_round_trip(self):
        session = FakeSession({NEW: [FakeResponse(200, {"data": RAW})]})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lightcast", "skills.json")
            skills = sl.download_and_cache("id", "secret", path=path, session=session)
            self.assertEqual(sl.load_skills(path), skills)
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["count"], len(RAW))
            self.assertEqual(meta["fields"], sl.FULL_FIELDS)


class TestRecords(unittest.TestCase):
    def test_simplify_flattens_objects(self):
        s = sl.simplify(RAW[0])
        self.assertEqual(s["type"], "Specialized Skill")
        self.assertEqual(s["category"], "Information Technology")
        self.assertEqual(s["subcategory"], "Programming Languages")

    def test_simplify_missing_hierarchy(self):
        s = sl.simplify(RAW[3])
        self.assertIsNone(s["category"])
        self.assertIsNone(s["subcategory"])
        self.assertIsNone(s["description"])


class TestNames(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(sl.normalize_name("  Microsoft   Excel. "), "microsoft excel")
        self.assertEqual(sl.normalize_name("Node.js"), "node.js")
        self.assertEqual(sl.normalize_name("C++"), "c++")
        self.assertEqual(sl.normalize_name("C#"), "c#")
        self.assertEqual(sl.normalize_name("Client–server"), "client-server")
        self.assertEqual(sl.normalize_name(None), "")

    def test_strip_qualifier(self):
        self.assertEqual(sl.strip_qualifier("Python (Programming Language)"), "Python")
        self.assertEqual(sl.strip_qualifier("PostgreSQL"), "PostgreSQL")

    def test_index_has_full_and_bare_names(self):
        index = sl.build_name_index([sl.simplify(r) for r in RAW])
        self.assertEqual(index["python"], ["KS1"])
        self.assertEqual(index["python (programming language)"], ["KS1"])
        self.assertEqual(index["c++"], ["KS5"])

    def test_ambiguous_bare_name_keeps_all_ids(self):
        index = sl.build_name_index([sl.simplify(r) for r in RAW])
        self.assertEqual(sorted(index["java"]), ["KS3", "KS4"])

    def test_stats(self):
        stats = sl.taxonomy_stats([sl.simplify(r) for r in RAW])
        self.assertEqual(stats["skills"], 5)
        self.assertEqual(stats["categories"], 1)
        self.assertEqual(stats["subcategories"], 2)
        self.assertEqual(stats["missing_category"], 1)
        self.assertEqual(stats["ambiguous_keys"], 1)


if __name__ == "__main__":
    unittest.main()
