"""Unit tests for skill_linking.py (standard library only, no API key needed).

The retrieval tests also need numpy and are skipped without it.

    cd CODE && python -m unittest test_skill_linking
"""

import csv
import json
import os
import tempfile
import types
import unittest

import skill_linking as sl

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

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

MORE = [
    {"id": "KS10", "name": "Network Architecture", "type": "Specialized Skill",
     "category": "Information Technology", "subcategory": "Networking", "description": None},
    {"id": "KS11", "name": "Network Security", "type": "Specialized Skill",
     "category": "Information Technology", "subcategory": "Cybersecurity", "description": None},
    {"id": "KS12", "name": "Security", "type": "Common Skill",
     "category": "Law", "subcategory": "Security", "description": None},
    {"id": "KS13", "name": "Security Clearance", "type": "Certification",
     "category": "Law", "subcategory": "Security Clearances", "description": None},
    {"id": "KS14", "name": "Design", "type": "Common Skill",
     "category": "Design", "subcategory": "Design", "description": None},
    {"id": "KS15", "name": "C (Programming Language)", "type": "Specialized Skill",
     "category": "Information Technology", "subcategory": "Programming Languages", "description": None},
    {"id": "KS6", "name": "Communication", "type": "Common Skill",
     "category": "Business", "subcategory": "Soft Skills", "description": None},
    {"id": "KS7", "name": "Satellite Communications", "type": "Specialized Skill",
     "category": "Engineering", "subcategory": "Telecommunications", "description": None},
    {"id": "KS8", "name": ".NET Framework", "type": "Specialized Skill",
     "category": "Information Technology", "subcategory": "Software Frameworks", "description": None},
    {"id": "KS9", "name": "Relational Databases", "type": "Specialized Skill",
     "category": "Information Technology", "subcategory": "Database Management Systems",
     "description": None},
]


def skills():
    return [sl.simplify(r) for r in RAW] + [dict(s) for s in MORE]


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


# ================================================================ step 1

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
            got = sl.download_and_cache("id", "secret", path=path, session=session)
            self.assertEqual(sl.load_skills(path), got)
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
        self.assertEqual(sl.normalize_name(".NET"), ".net")
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


# ================================================================ step 2

class TestVariants(unittest.TestCase):
    def test_wrapper_words_removed(self):
        self.assertEqual(sl.core_form("Knowledge of satellite communication protocols"),
                         "satellite communication protocols")
        self.assertEqual(sl.core_form("Strong experience with SQL"), "sql")
        self.assertEqual(sl.core_form("communication skills"), "communication")
        self.assertEqual(sl.core_form("Python"), "python")

    def test_variants_order_and_plural(self):
        v = sl.mention_variants("Relational Databases")
        self.assertEqual(v[0], "relational databases")
        self.assertIn("relational database", v)
        self.assertNotIn("clas", sl.mention_variants("class"))

    def test_and_ampersand(self):
        self.assertIn("research & development", sl.mention_variants("Research and Development"))


class TestAliases(unittest.TestCase):
    def test_load_skips_comments_and_normalizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "aliases.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"_about": "x", "Postgres ": "PostgreSQL", "JS": "JavaScript"}, f)
            table = sl.load_aliases(path)
        self.assertEqual(table, {"postgres": "postgresql", "js": "javascript"})

    def test_check_aliases_lists_missing_targets(self):
        index = sl.build_name_index(skills())
        missing = sl.check_aliases({"postgres": "postgresql", "js": "javascript"}, index)
        self.assertEqual(missing, ["javascript"])

    def test_shipped_alias_file_is_valid(self):
        here = os.path.dirname(os.path.abspath(__file__))
        table = sl.load_aliases(os.path.join(here, "skill_aliases.json"))
        self.assertGreater(len(table), 50)
        self.assertNotIn("", table)


class TestLinker(unittest.TestCase):
    def setUp(self):
        self.linker = sl.Linker(skills(), aliases={"postgres": "postgresql", "py": "python"})

    def test_exact_bare_name(self):
        r = self.linker.link("Python")
        self.assertEqual((r["id"], r["method"], r["score"]), ("KS1", "exact", 1.0))
        self.assertEqual(r["subcategory"], "Programming Languages")
        self.assertFalse(r["needs_review"])

    def test_exact_after_wrapper_words(self):
        self.assertEqual(self.linker.link("communication skills")["id"], "KS6")

    def test_exact_plural(self):
        self.assertEqual(self.linker.link("relational database")["id"], "KS9")

    def test_dotnet(self):
        self.assertEqual(self.linker.link(".NET Framework")["id"], "KS8")

    def test_alias(self):
        r = self.linker.link("Postgres")
        self.assertEqual((r["id"], r["method"]), ("KS2", "alias"))

    def test_ambiguous_decided_by_context(self):
        r = self.linker.link("Java", context="Java programming for backend services")
        self.assertEqual(r["id"], "KS3")
        self.assertFalse(r["needs_review"])
        self.assertEqual(sorted(r["candidates"]), ["KS3", "KS4"])

    def test_ambiguous_without_context_is_flagged(self):
        r = self.linker.link("Java")
        self.assertEqual(r["id"], "KS3")  # specialized skill wins the tie
        self.assertTrue(r["needs_review"])

    def test_fuzzy_spelling(self):
        r = self.linker.link("PostgresQ")
        self.assertEqual((r["id"], r["method"]), ("KS2", "fuzzy"))
        self.assertGreaterEqual(r["score"], 0.92)

    def test_contains_longest_name(self):
        r = self.linker.link("network architecture design")
        self.assertEqual((r["id"], r["method"]), ("KS10", "contains"))
        self.assertAlmostEqual(r["score"], 2 / 3, places=3)
        self.assertFalse(r["needs_review"])

    def test_contains_prefers_longer_over_generic(self):
        r = self.linker.link("space-based network security standards")
        self.assertEqual(r["id"], "KS11")  # not the generic "Security"
        self.assertTrue(r["needs_review"])  # covers 2 of 4 words ("space-based" is one)

    def test_contains_two_word_certification(self):
        self.assertEqual(self.linker.link("Secret Security Clearance")["id"], "KS13")

    def test_contains_plural_form(self):
        r = self.linker.link("satellite communication protocols")
        self.assertEqual((r["id"], r["method"]), ("KS7", "contains"))

    def test_contains_ignores_generic_single_words(self):
        self.assertEqual(self.linker.link("Design instruction creation")["method"], "nil")

    def test_contains_single_specialized_word(self):
        self.assertEqual(self.linker.link("python scripting")["id"], "KS1")

    def test_contains_ignores_one_letter_names(self):
        self.assertEqual(self.linker.link("c level reporting")["method"], "nil")

    def test_nil(self):
        r = self.linker.link("ground station operations")
        self.assertIsNone(r["id"])
        self.assertEqual(r["method"], "nil")
        self.assertIsNone(r["category"])

    def test_empty_mention_is_nil(self):
        self.assertEqual(self.linker.link("  ")["method"], "nil")
        self.assertEqual(self.linker.link(None)["method"], "nil")

    def test_cache_keeps_original_mention(self):
        self.assertEqual(self.linker.link("PYTHON")["mention"], "PYTHON")
        self.assertEqual(self.linker.link("python")["mention"], "python")


class TestApplying(unittest.TestCase):
    def setUp(self):
        self.linker = sl.Linker(skills(), aliases={"postgres": "postgresql"})

    def test_link_facts_uses_span_as_context(self):
        facts = [{"name": "Java", "source_span": "Java programming", "start": 0, "end": 16}]
        out = sl.link_facts(self.linker, facts)
        self.assertEqual(out[0]["link"]["id"], "KS3")
        self.assertEqual(out[0]["start"], 0)

    def test_link_record_posting(self):
        record = {"doc_id": "1", "kind": "posting", "status": "ok",
                  "facts": {"required_skills": [{"name": "Postgres", "importance": "required",
                                                 "source_span": "Postgres", "start": 0, "end": 8}],
                            "required_degrees": [], "experience_required": None},
                  "rejected": []}
        out = sl.link_record(self.linker, record)
        self.assertEqual(out["facts"]["required_skills"][0]["link"]["id"], "KS2")
        self.assertNotIn("link", record["facts"]["required_skills"][0])  # input left unchanged

    def test_link_record_skips_errors(self):
        record = {"doc_id": "2", "kind": "resume", "status": "error", "error": "x"}
        self.assertIs(sl.link_record(self.linker, record), record)

    def test_summary(self):
        links = [self.linker.link(m) for m in ("Python", "Postgres", "ground station operations")]
        s = sl.link_summary(links)
        self.assertEqual((s["mentions"], s["linked"], s["nil"]), (3, 2, 1))
        self.assertEqual(s["by_method"], {"exact": 1, "alias": 1, "nil": 1})

    def test_skill_nodes(self):
        linked = sl.to_skill_node(self.linker.link("Python"))
        self.assertEqual((linked["skill_id"], linked["source"]), ("KS1", "lightcast"))
        nil = sl.to_skill_node(self.linker.link("Ground Station Operations"))
        self.assertEqual(nil["skill_id"], "EXT:ground-station-operations")
        self.assertEqual(nil["source"], "extracted")


class TestPartialCredit(unittest.TestCase):
    def setUp(self):
        linker = sl.Linker(skills())
        self.job = sl.link_facts(linker, [
            {"name": "Python", "importance": "required"},
            {"name": "PostgreSQL", "importance": "preferred"},
            {"name": "ground station operations", "importance": "required"},
        ])
        self.resume = sl.link_facts(linker, [{"name": "Java (Programming Language)"}, {"name": "C++"}])

    def test_credit_levels(self):
        python, postgres = self.job[0]["link"], self.job[1]["link"]
        java = self.resume[0]["link"]
        self.assertEqual(sl.credit(python, python), (1.0, "same skill"))
        self.assertEqual(sl.credit(python, java), (sl.LAMBDA_SUB, "same subcategory"))
        self.assertEqual(sl.credit(postgres, java), (sl.LAMBDA_CAT, "same category"))

    def test_coverage(self):
        g, rows, unresolved = sl.coverage(self.job, self.resume)
        # (2 * 0.5 + 1 * 0.25) / (2 + 1); the unlinked job skill is not scored
        self.assertAlmostEqual(g, 1.25 / 3)
        self.assertEqual(unresolved, ["ground station operations"])
        self.assertEqual(rows[0]["level"], "same subcategory")

    def test_nothing_linked(self):
        g, rows, unresolved = sl.coverage(self.job[2:], self.resume)
        self.assertEqual((g, rows), (0.0, []))

    def test_same_skill_counted_once(self):
        linker = sl.Linker(skills())
        job = sl.link_facts(linker, [{"name": "Python", "importance": "preferred"},
                                     {"name": "python scripting", "importance": "required"},
                                     {"name": "PostgreSQL", "importance": "preferred"}])
        g, rows, _ = sl.coverage(job, self.resume)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["importance"], "required")
        self.assertEqual(rows[0]["job_skill"], "Python; python scripting")
        self.assertAlmostEqual(g, (2 * 0.5 + 1 * 0.25) / 3)


# ================================================================ step 3

class FakeEncoder:
    """Bag-of-words vectors over a fixed vocabulary; counts how often it is called."""

    VOCAB = ["python", "java", "c++", "programming", "language", "coffee", "postgresql",
             "database", "satellite", "communications", "network", "relational", "framework"]

    def __init__(self):
        self.calls = 0

    def encode(self, texts, batch_size=32):
        self.calls += 1
        out = []
        for text in texts:
            words = set(sl._WORD.findall(text.lower()))
            out.append([1.0 if w in words else 0.0 for w in self.VOCAB])
        return out


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestRetrieval(unittest.TestCase):
    def test_search_ranks_by_meaning(self):
        retriever = sl.EmbeddingRetriever(skills(), encoder=FakeEncoder())
        top_id, score = retriever.search("satellite communications")[0]
        self.assertEqual(top_id, "KS7")
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_cache_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vectors.npz")
            first = FakeEncoder()
            sl.EmbeddingRetriever(skills(), encoder=first, cache_path=path)
            second = FakeEncoder()
            retriever = sl.EmbeddingRetriever(skills(), encoder=second, cache_path=path)
            self.assertEqual(second.calls, 0)
            retriever.search("coffee")
            self.assertEqual(second.calls, 1)

    # "satellite links communications" contains no whole skill name and is not
    # spelled closely enough to any name to link, so only the retriever finds it.

    def test_suggestions_only_by_default(self):
        retriever = sl.EmbeddingRetriever(skills(), encoder=FakeEncoder())
        linker = sl.Linker(skills(), retriever=retriever, review_embed=0.3)
        r = linker.link("satellite links communications")
        self.assertEqual(r["method"], "nil")
        self.assertEqual(r["suggestions"][0]["id"], "KS7")
        self.assertEqual(r["suggestions"][0]["via"], "embedding")
        self.assertTrue(r["needs_review"])

    def test_auto_embed_links_but_flags(self):
        retriever = sl.EmbeddingRetriever(skills(), encoder=FakeEncoder())
        linker = sl.Linker(skills(), retriever=retriever, auto_embed=0.3)
        r = linker.link("satellite links communications")
        self.assertEqual((r["id"], r["method"]), ("KS7", "embedding"))
        self.assertTrue(r["needs_review"])


# ================================================================ step 4: decision step

class FakeHTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


class FakeCompletions:
    """Returns queued replies (strings) or raises queued exceptions; records each request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        content, finish = item if isinstance(item, tuple) else (item, "stop")
        choice = types.SimpleNamespace(message=types.SimpleNamespace(content=content), finish_reason=finish)
        return types.SimpleNamespace(choices=[choice])


def fake_client(*replies):
    completions = FakeCompletions(replies)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions)), completions


CANDS = [sl.simplify(RAW[0]), sl.simplify(RAW[2])]  # Python, Java (programming)


class TestDecider(unittest.TestCase):
    def test_constrained_choice(self):
        client, comp = fake_client('{"choice": "KS3"}')
        decider = sl.LLMDecider(client)
        self.assertEqual(decider.decide("Java 8", "Java 8 and Spring", CANDS), "KS3")
        schema = comp.requests[0]["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["choice"]["enum"], ["KS1", "KS3", "NIL"])
        self.assertIn("/no_think", comp.requests[0]["messages"][1]["content"])
        self.assertEqual(decider.calls, 1)

    def test_nil(self):
        client, _ = fake_client('{"choice": "NIL"}')
        self.assertIsNone(sl.LLMDecider(client).decide("ground terminals", None, CANDS))

    def test_falls_back_to_plain_json(self):
        client, comp = fake_client(FakeHTTPError(400), '```json\n{"choice": "KS1"}\n```')
        decider = sl.LLMDecider(client)
        self.assertEqual(decider.decide("py", None, CANDS), "KS1")
        self.assertFalse(decider.constrained)
        self.assertNotIn("response_format", comp.requests[1])

    def test_think_block_stripped(self):
        client, _ = fake_client('<think>both are languages</think>{"choice": "KS1"}')
        self.assertEqual(sl.LLMDecider(client).decide("python", None, CANDS), "KS1")

    def test_invalid_choice_raises(self):
        client, _ = fake_client('{"choice": "KS999"}')
        decider = sl.LLMDecider(client)
        with self.assertRaises(ValueError):
            decider.decide("python", None, CANDS)
        self.assertEqual(decider.errors, 1)

    def test_retries_server_error(self):
        client, _ = fake_client(FakeHTTPError(503), '{"choice": "KS1"}')
        self.assertEqual(sl.LLMDecider(client, backoff=0).decide("python", None, CANDS), "KS1")

    def test_cut_off_reply_raises(self):
        client, _ = fake_client(('{"choi', "length"))
        with self.assertRaises(ValueError):
            sl.LLMDecider(client).decide("python", None, CANDS)

    def test_no_switch_for_other_models(self):
        client, comp = fake_client('{"choice": "KS1"}')
        sl.LLMDecider(client, model="llama3.1-8b").decide("python", None, CANDS)
        self.assertNotIn("/no_think", comp.requests[0]["messages"][1]["content"])


class FakeDecider:
    def __init__(self, answer):
        self.answer = answer
        self.seen = []

    def decide(self, mention, context, candidates):
        self.seen.append((mention, [c["id"] for c in candidates]))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class TestLinkerWithDecider(unittest.TestCase):
    MENTION = "space-based network security standards"  # contains-link flagged for review

    def test_confirmed_link_keeps_method(self):
        decider = FakeDecider("KS11")
        r = sl.Linker(skills(), decider=decider).link(self.MENTION)
        self.assertEqual((r["id"], r["method"], r["decided_by"]), ("KS11", "contains", "llm"))
        self.assertFalse(r["needs_review"])
        self.assertEqual(decider.seen[0][1][0], "KS11")

    def test_decider_can_reject(self):
        r = sl.Linker(skills(), decider=FakeDecider(None)).link(self.MENTION)
        self.assertEqual((r["id"], r["method"], r["proposed"]), (None, "nil", "contains"))
        self.assertFalse(r["needs_review"])

    def test_decider_can_pick_another_candidate(self):
        r = sl.Linker(skills(), decider=FakeDecider("KS12")).link(self.MENTION)
        self.assertEqual((r["id"], r["method"]), ("KS12", "llm"))

    def test_decider_error_keeps_stage_result(self):
        r = sl.Linker(skills(), decider=FakeDecider(RuntimeError("down"))).link(self.MENTION)
        self.assertEqual(r["id"], "KS11")
        self.assertTrue(r["needs_review"])
        self.assertIn("down", r["decision_error"])

    def test_confident_links_skip_the_decider(self):
        decider = FakeDecider("KS3")
        sl.Linker(skills(), decider=decider).link("Python")
        self.assertEqual(decider.seen, [])

    def test_ambiguous_name_goes_to_decider(self):
        decider = FakeDecider("KS4")
        r = sl.Linker(skills(), decider=decider).link("Java")
        self.assertEqual(sorted(decider.seen[0][1]), ["KS3", "KS4"])
        self.assertEqual(r["id"], "KS4")

    def test_decision_cached(self):
        decider = FakeDecider("KS11")
        linker = sl.Linker(skills(), decider=decider)
        linker.link(self.MENTION)
        linker.link(self.MENTION)
        self.assertEqual(len(decider.seen), 1)

    def test_summary_counts_decisions(self):
        linker = sl.Linker(skills(), decider=FakeDecider(RuntimeError("down")))
        s = sl.link_summary([linker.link(self.MENTION), linker.link("Python")])
        self.assertEqual((s["decided_by_llm"], s["decision_errors"]), (0, 1))

    def test_candidates_for_fills_from_fuzzy(self):
        pool = sl.Linker(skills()).candidates_for("postgres sql server", k=4)
        self.assertIn("KS2", pool)
        self.assertLessEqual(len(pool), 4)


# ================================================================ step 5: corpus

def extraction_records():
    return [
        {"doc_id": "r1", "kind": "resume", "status": "ok",
         "facts": {"skills": [{"name": "Python", "source_span": "Python", "start": 10, "end": 16},
                              {"name": "ground station operations", "source_span": "ground station operations",
                               "start": 20, "end": 45}],
                   "degrees": [], "experience": []}, "rejected": []},
        {"doc_id": "p1", "kind": "posting", "status": "ok",
         "facts": {"required_skills": [{"name": "PostgreSQL", "importance": "preferred",
                                        "source_span": "PostgreSQL", "start": 0, "end": 10}],
                   "required_degrees": [], "experience_required": None}, "rejected": []},
        {"doc_id": "p2", "kind": "posting", "status": "error", "error": "timeout"},
    ]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestCorpus(unittest.TestCase):
    def test_run_linking_resumes(self):
        linker = sl.Linker(skills())
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "in.jsonl"), os.path.join(tmp, "linked", "out.jsonl")
            write_jsonl(src, extraction_records())
            self.assertEqual(sl.run_linking(linker, src, dst, log_every=0), {"ok": 2, "error": 1, "skipped": 0})
            self.assertEqual(sl.run_linking(linker, src, dst, log_every=0), {"ok": 0, "error": 1, "skipped": 2})
            linked = {r["doc_id"]: r for r in sl.latest_records(dst)}
            self.assertEqual(linked["r1"]["facts"]["skills"][0]["link"]["id"], "KS1")

    def test_run_linking_limit_and_partial_line(self):
        linker = sl.Linker(skills())
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "in.jsonl"), os.path.join(tmp, "out.jsonl")
            write_jsonl(src, extraction_records())
            with open(dst, "w", encoding="utf-8") as f:
                f.write('{"doc_id": "x", "status"')  # what a killed run leaves behind
            self.assertEqual(sl.run_linking(linker, src, dst, limit=1, log_every=0)["ok"], 1)
            self.assertEqual(len(sl.latest_records(dst)), 1)

    def test_report_and_graph_export(self):
        linker = sl.Linker(skills())
        records = [sl.link_record(linker, r) for r in extraction_records()]
        report = sl.corpus_report(records)
        self.assertEqual(report["resume"]["nil"], 1)
        self.assertEqual(report["resume"]["top_nil"], [("ground station operations", 1)])
        self.assertEqual(report["posting"]["distinct_skills"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            paths = sl.export_graph(records, tmp)
            with open(paths["skills"], newline="", encoding="utf-8") as f:
                nodes = {row["skill_id"]: row for row in csv.DictReader(f)}
            self.assertEqual(nodes["KS1"]["source"], "lightcast")
            self.assertEqual(nodes["EXT:ground-station-operations"]["source"], "extracted")
            with open(paths["posting"], newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["importance"], "preferred")
            self.assertEqual((rows[0]["span_start"], rows[0]["span_end"]), ("0", "10"))
            with open(paths["resume"], newline="", encoding="utf-8") as f:
                self.assertNotIn("importance", next(csv.reader(f)))


# ================================================================ missed-skill check

class TestMissedSkills(unittest.TestCase):
    def setUp(self):
        self.finder = sl.SkillTextFinder(skills(), aliases={"postgres": "postgresql"})

    def test_finds_alias_in_text(self):
        text = "Built ETL jobs on Postgres and Airflow."
        start, end = self.finder.find("KS2", text)
        self.assertEqual(text[start:end], "Postgres")

    def test_respects_word_boundaries(self):
        self.assertIsNone(self.finder.find("KS3", "Senior JavaScript developer"))
        self.assertIsNotNone(self.finder.find("KS3", "Senior Java developer"))

    def test_spacing_in_text(self):
        self.assertIsNotNone(self.finder.find("KS11", "Network    Security policies"))

    def test_short_names_not_searched(self):
        self.assertIsNone(self.finder.find("KS15", "Reported to C level executives"))

    def test_statuses(self):
        linker = sl.Linker(skills(), aliases={"postgres": "postgresql"})
        job = sl.link_facts(linker, [{"name": "Python", "importance": "required"},
                                     {"name": "PostgreSQL", "importance": "required"},
                                     {"name": "Network Security", "importance": "preferred"},
                                     {"name": "ground station operations", "importance": "required"}])
        resume = sl.link_facts(linker, [{"name": "Python"}])
        text = "Python scripts; reporting from Postgres."
        rows = {r["job_skill"]: r for r in sl.check_missing(job, resume, text, self.finder)}
        self.assertEqual(rows["Python"]["status"], "matched")
        self.assertEqual(rows["PostgreSQL"]["status"], "found_in_text")
        self.assertEqual(rows["Network Security"]["status"], "not_found")
        self.assertEqual(rows["ground station operations"]["status"], "unresolved")


if __name__ == "__main__":
    unittest.main()
