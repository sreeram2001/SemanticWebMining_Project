"""Ontology-guided, source-grounded extraction for resumes and job postings.

A fact is kept only if its source_span can be found in the document and it
passes the ontology's type checks. Rejected facts are kept alongside the
accepted ones so the unsupported-fact rate can be measured.
"""

import json
import os
import re
import time

DEGREE_LEVELS = ["high_school", "associate", "bachelors", "masters", "phd"]
IMPORTANCE = ["required", "preferred"]

MAX_SKILL_NAME = 100
MAX_DURATION_MONTHS = 600
MAX_MIN_YEARS = 50

UNSUPPORTED = "source_span not found in text"


def _nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


_SPAN = {"type": "string", "minLength": 1}

RESUME_SCHEMA = {
    "type": "object",
    "properties": {
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "source_span": _SPAN},
                "required": ["name", "source_span"],
                "additionalProperties": False,
            },
        },
        "degrees": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "enum": DEGREE_LEVELS},
                    "field": _nullable({"type": "string"}),
                    "institution": _nullable({"type": "string"}),
                    "year": _nullable({"type": "string"}),
                    "source_span": _SPAN,
                },
                "required": ["level", "field", "institution", "year", "source_span"],
                "additionalProperties": False,
            },
        },
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "company": _nullable({"type": "string"}),
                    "duration_months": _nullable({"type": "integer"}),
                    "source_span": _SPAN,
                },
                "required": ["title", "company", "duration_months", "source_span"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["skills", "degrees", "experience"],
    "additionalProperties": False,
}

POSTING_SCHEMA = {
    "type": "object",
    "properties": {
        "required_skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "importance": {"type": "string", "enum": IMPORTANCE},
                    "source_span": _SPAN,
                },
                "required": ["name", "importance", "source_span"],
                "additionalProperties": False,
            },
        },
        "required_degrees": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "enum": DEGREE_LEVELS},
                    "field": _nullable({"type": "string"}),
                    "importance": {"type": "string", "enum": IMPORTANCE},
                    "source_span": _SPAN,
                },
                "required": ["level", "field", "importance", "source_span"],
                "additionalProperties": False,
            },
        },
        "experience_required": _nullable({
            "type": "object",
            "properties": {
                "min_years": {"type": "number"},
                "importance": {"type": "string", "enum": IMPORTANCE},
                "source_span": _SPAN,
            },
            "required": ["min_years", "importance", "source_span"],
            "additionalProperties": False,
        }),
    },
    "required": ["required_skills", "required_degrees", "experience_required"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You extract structured facts for a job-matching knowledge graph. "
    "Return only JSON that matches the requested schema."
)

_RULES = """Rules:
- Extract only facts stated in the text. Do not infer or invent anything.
- source_span must be copied character for character from the text: the shortest
  phrase that states the fact. Do not paraphrase, correct spelling or join
  separate pieces of text.
- Use null for a field the text does not state.
- Return JSON only, with no markdown and no commentary."""

RESUME_PROMPT = """Extract the candidate's skills, degrees and work experience from the resume below.

JSON shape:
{
  "skills": [{"name": "skill name", "source_span": "exact text"}],
  "degrees": [{"level": "high_school|associate|bachelors|masters|phd", "field": "field of study or null",
               "institution": "school or null", "year": "year or null", "source_span": "exact text"}],
  "experience": [{"title": "job title", "company": "company or null",
                  "duration_months": integer or null, "source_span": "exact text naming the role"}]
}

""" + _RULES + """

RESUME TEXT:
"""

POSTING_PROMPT = """Extract the requirements stated in the job posting below.

JSON shape:
{
  "required_skills": [{"name": "skill name", "importance": "required|preferred", "source_span": "exact text"}],
  "required_degrees": [{"level": "high_school|associate|bachelors|masters|phd", "field": "field or null",
                        "importance": "required|preferred", "source_span": "exact text"}],
  "experience_required": {"min_years": number, "importance": "required|preferred", "source_span": "exact text"} or null
}

Mark a requirement "preferred" only when the posting says so (preferred, a plus, nice to have, ideally).
Leave a degree out when the posting accepts any degree or none.
experience_required is the overall minimum years of experience; use null if no number is stated.

""" + _RULES + """

JOB POSTING:
"""

KINDS = {
    "resume": {"schema": RESUME_SCHEMA, "prompt": RESUME_PROMPT},
    "posting": {"schema": POSTING_SCHEMA, "prompt": POSTING_PROMPT},
}

# One-for-one substitutions so normalized offsets still map back to the source.
_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
})


def _normalize(text):
    """Lower-case, unify quotes and dashes, collapse whitespace.

    Returns the normalized string and, for each of its characters, the index
    of the source character it came from.
    """
    chars, index = [], []
    in_space = False
    for i, ch in enumerate(text.translate(_CHAR_MAP)):
        if ch.isspace():
            if not in_space and chars:
                chars.append(" ")
                index.append(i)
            in_space = True
            continue
        in_space = False
        low = ch.lower()
        chars.append(low if len(low) == 1 else ch)
        index.append(i)
    if chars and chars[-1] == " ":
        chars.pop()
        index.pop()
    return "".join(chars), index


def locate_span(source, span):
    """Return (start, end) of span in source, or None.

    The resumes are full of runs of spaces and the model tends to tidy them,
    so after an exact search the match is retried ignoring case, quote style
    and whitespace.
    """
    if not isinstance(span, str) or not span.strip():
        return None
    pos = source.find(span)
    if pos >= 0:
        return pos, pos + len(span)
    norm_src, index = _normalize(source)
    norm_span, _ = _normalize(span)
    if not norm_span:
        return None
    pos = norm_src.find(norm_span)
    if pos < 0:
        return None
    return index[pos], index[pos + len(norm_span) - 1] + 1


def _text(value, limit=None):
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if limit and len(value) > limit:
        return None
    return value


def _optional_text(value):
    if value is None:
        return True, None
    value = _text(value)
    return value is not None, value


def _check_skill(fact, with_importance):
    name = _text(fact.get("name"), MAX_SKILL_NAME)
    if name is None:
        return None, "skill name missing or too long"
    out = {"name": name}
    if with_importance:
        if fact.get("importance") not in IMPORTANCE:
            return None, "importance not required/preferred"
        out["importance"] = fact["importance"]
    return out, None


def _check_degree(fact, with_importance):
    if fact.get("level") not in DEGREE_LEVELS:
        return None, "degree level not in ontology"
    ok, field = _optional_text(fact.get("field"))
    if not ok:
        return None, "degree field is not text"
    out = {"level": fact["level"], "field": field}
    if with_importance:
        if fact.get("importance") not in IMPORTANCE:
            return None, "importance not required/preferred"
        out["importance"] = fact["importance"]
    else:
        for key in ("institution", "year"):
            ok, value = _optional_text(fact.get(key))
            if not ok:
                return None, f"degree {key} is not text"
            out[key] = value
    return out, None


def _check_experience(fact):
    title = _text(fact.get("title"))
    if title is None:
        return None, "experience title missing"
    ok, company = _optional_text(fact.get("company"))
    if not ok:
        return None, "experience company is not text"
    months = fact.get("duration_months")
    # bool is an int subclass; a model emitting true/false here is a type error
    if months is not None and (
        isinstance(months, bool) or not isinstance(months, int)
        or not 0 <= months <= MAX_DURATION_MONTHS
    ):
        return None, "duration_months not a plausible integer"
    return {"title": title, "company": company, "duration_months": months}, None


def _check_min_years(fact):
    years = fact.get("min_years")
    if isinstance(years, bool) or not isinstance(years, (int, float)) or not 0 <= years <= MAX_MIN_YEARS:
        return None, "min_years not a plausible number"
    if fact.get("importance") not in IMPORTANCE:
        return None, "importance not required/preferred"
    return {"min_years": years, "importance": fact["importance"]}, None


FIELDS = {
    "resume": {
        "skills": lambda f: _check_skill(f, False),
        "degrees": lambda f: _check_degree(f, False),
        "experience": _check_experience,
    },
    "posting": {
        "required_skills": lambda f: _check_skill(f, True),
        "required_degrees": lambda f: _check_degree(f, True),
        "experience_required": _check_min_years,
    },
}

# experience_required is a single object (or null), not a list
SINGLE_FIELDS = {"experience_required"}


def _dedupe_key(field, fact):
    if field in ("skills", "required_skills"):
        return fact["name"].lower()
    return json.dumps({k: v for k, v in fact.items() if k not in ("start", "end")}, sort_keys=True).lower()


def ground(kind, raw, source):
    """Check a parsed model output against the source text and the ontology.

    Returns (facts, rejected). Each kept fact carries source_span exactly as
    it appears in the source, plus start and end character offsets.
    """
    if kind not in FIELDS:
        raise ValueError(f"unknown kind: {kind}")
    if not isinstance(raw, dict):
        raise ValueError("model output is not a JSON object")

    facts, rejected = {}, []
    for field, check in FIELDS[kind].items():
        value = raw.get(field)
        if field in SINGLE_FIELDS:
            items = [] if value is None else [value]
        elif value is None:
            items = []
        elif isinstance(value, list):
            items = value
        else:
            rejected.append({"field": field, "fact": value, "reason": "expected a list"})
            items = []

        kept, seen = [], set()
        for item in items:
            if not isinstance(item, dict):
                rejected.append({"field": field, "fact": item, "reason": "not an object"})
                continue
            fact, reason = check(item)
            if fact is None:
                rejected.append({"field": field, "fact": item, "reason": reason})
                continue
            where = locate_span(source, item.get("source_span"))
            if where is None:
                rejected.append({"field": field, "fact": item, "reason": UNSUPPORTED})
                continue
            start, end = where
            fact.update(source_span=source[start:end], start=start, end=end)
            key = _dedupe_key(field, fact)
            if key in seen:
                continue
            seen.add(key)
            kept.append(fact)

        facts[field] = (kept[0] if kept else None) if field in SINGLE_FIELDS else kept
    return facts, rejected


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_json(text):
    """Pull the JSON object out of a model reply.

    Qwen3 may prepend a <think> block and some servers wrap output in code
    fences even when told not to.
    """
    if not isinstance(text, str):
        raise ValueError("empty model reply")
    text = _THINK.sub("", text).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in model reply")
        return json.loads(text[start:end + 1])


def _status(exc):
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status


def _is_transient(exc):
    status = _status(exc)
    if status is not None:
        return status == 429 or status >= 500
    name = type(exc).__name__
    return "Timeout" in name or "Connection" in name


class Extractor:
    """Runs grounded extraction against an OpenAI-compatible chat endpoint.

    constrained=True asks the server for schema-constrained decoding. If the
    server rejects that request format, the extractor falls back to plain
    JSON prompting for the rest of the session and records which mode
    produced each document.
    """

    def __init__(self, client, model, constrained=True, max_chars=8000,
                 max_tokens=4000, temperature=0.0, retries=3, backoff=2.0):
        self.client = client
        self.model = model
        self.constrained = constrained
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.backoff = backoff

    def _request(self, kind, text):
        spec = KINDS[kind]
        args = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": spec["prompt"] + text},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if self.constrained:
            args["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": f"{kind}_facts", "schema": spec["schema"], "strict": True},
            }
        return self.client.chat.completions.create(**args)

    def _complete(self, kind, text):
        attempt = 0
        while True:
            try:
                return self._request(kind, text)
            except Exception as exc:
                if self.constrained and _status(exc) in (400, 422):
                    self.constrained = False
                    continue
                attempt += 1
                if attempt > self.retries or not _is_transient(exc):
                    raise
                time.sleep(self.backoff * 2 ** (attempt - 1))

    def extract(self, kind, doc_id, text):
        if kind not in KINDS:
            raise ValueError(f"unknown kind: {kind}")
        record = {"doc_id": str(doc_id), "kind": kind, "model": self.model}
        if not isinstance(text, str) or not text.strip():
            record.update(status="error", error="empty document")
            return record

        # Offsets refer to this truncated text, which is a prefix of the
        # original, so they stay valid against the full document.
        source = text[: self.max_chars] if self.max_chars else text
        record.update(source_chars=len(source), truncated=len(source) < len(text))
        try:
            reply = self._complete(kind, source)
            record["constrained"] = self.constrained
            choice = reply.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise ValueError("reply cut off at max_tokens")
            facts, rejected = ground(kind, parse_json(choice.message.content), source)
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            return record
        record.update(status="ok", facts=facts, rejected=rejected)
        return record


def done_ids(path):
    """IDs already written to a JSONL output file, so a run can resume."""
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # a run killed mid-write leaves a partial last line
                continue
            if rec.get("status") == "ok":
                ids.add(rec["doc_id"])
    return ids


def run_batch(extractor, kind, docs, out_path, limit=None, log_every=25):
    """Extract (doc_id, text) pairs into a JSONL file, one record per line.

    Documents already extracted successfully are skipped and failed ones are
    retried, so the same call can be rerun after an interruption.
    """
    skip = done_ids(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    counts = {"ok": 0, "error": 0, "skipped": 0}
    with open(out_path, "a", encoding="utf-8") as out:
        # Without this, the first new record would be glued onto a partial
        # line left by a killed run and lost with it.
        if out.tell() > 0:
            with open(out_path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    out.write("\n")
        for doc_id, text in docs:
            if str(doc_id) in skip:
                counts["skipped"] += 1
                continue
            if limit is not None and counts["ok"] + counts["error"] >= limit:
                break
            rec = extractor.extract(kind, doc_id, text)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            counts[rec["status"]] += 1
            n = counts["ok"] + counts["error"]
            if log_every and n % log_every == 0:
                print(f"  {kind}: {n} done ({counts['error']} errors)")
    return counts


def load_records(path):
    """Latest record per document from a JSONL output file."""
    latest = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[rec["doc_id"]] = rec
    return list(latest.values())


def summarize(records):
    """Per-field fact counts.

    unsupported = the model's span is not in the text (an invented or
    paraphrased fact); invalid = the fact fails the ontology's type checks.
    """
    summary = {"documents": len(records), "errors": 0, "truncated": 0, "fields": {}, "reasons": {}}
    blank = lambda: {"kept": 0, "unsupported": 0, "invalid": 0}
    for rec in records:
        if rec.get("status") != "ok":
            summary["errors"] += 1
            continue
        summary["truncated"] += bool(rec.get("truncated"))
        for field, value in rec["facts"].items():
            n = len(value) if isinstance(value, list) else int(value is not None)
            summary["fields"].setdefault(field, blank())["kept"] += n
        for r in rec["rejected"]:
            bucket = "unsupported" if r["reason"] == UNSUPPORTED else "invalid"
            summary["fields"].setdefault(r["field"], blank())[bucket] += 1
            summary["reasons"][r["reason"]] = summary["reasons"].get(r["reason"], 0) + 1
    for counts in summary["fields"].values():
        total = counts["kept"] + counts["unsupported"] + counts["invalid"]
        counts["unsupported_rate"] = round(counts["unsupported"] / total, 4) if total else None
    return summary
