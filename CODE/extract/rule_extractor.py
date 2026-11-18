"""Offline rule-based extractor (baseline / no-LLM runs); same format as extraction.Extractor."""

import re

from extract.extraction import ground

DEGREE_PATTERNS = [
    ("phd", r"\b(?:Ph\.?\s?D\.?|Doctorate|Doctor of Philosophy)(?!\w)"),
    ("masters", r"\b(?:Master(?:'s)?(?: of [A-Z][a-z]+)?|M\.S\.|M\.A\.|MBA|M\.B\.A\.|MS(?= in\b))(?!\w)"),
    ("bachelors", r"\b(?:Bachelor(?:'s)?(?: of [A-Z][a-z]+)?|B\.S\.|B\.A\.|BBA|B\.Tech|BS(?= in\b)|BA(?= in\b))(?!\w)"),
    ("associate", r"\bAssociate(?:'s)? (?:degree|of [A-Z][a-z]+)(?!\w)"),
    ("high_school", r"\b(?:High School Diploma|GED|high school diploma)(?!\w)"),
]
_PREFERRED = re.compile(r"\b(preferred|a plus|nice to have|bonus|desired|ideally|advantageous)\b", re.I)
_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:or more\s+)?years?\b", re.I)
_DATE_RANGE = re.compile(
    r"(?P<a>(?:[A-Z][a-z]{2,8}\.?\s+)?(?:19|20)\d{2}|\d{1,2}/(?:19|20)\d{2})\s*(?:to|-|–|—)\s*"
    r"(?P<b>(?:[A-Z][a-z]{2,8}\.?\s+)?(?:19|20)\d{2}|\d{1,2}/(?:19|20)\d{2}|Current|Present|current|present|Now)"
)
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _sentence(text, pos):
    starts = [text.rfind(c, 0, pos) for c in (".", "\n", ";", "•")]
    ends = [i for i in (text.find(c, pos) for c in (".", "\n", ";", "•")) if i >= 0]
    return text[max(starts) + 1: min(ends) if ends else len(text)]


def parse_month(s, current):
    """'Jan 2015' / '01/2015' / '2015' / 'Current' -> months since year 0."""
    s = s.strip().rstrip(".")
    if s.lower() in ("current", "present", "now"):
        return current
    m = re.match(r"(\d{1,2})/(\d{4})$", s)
    if m:
        return int(m.group(2)) * 12 + int(m.group(1)) - 1
    m = re.match(r"([A-Za-z]{3,9})\.?\s+(\d{4})$", s)
    if m and m.group(1)[:3].lower() in _MONTHS:
        return int(m.group(2)) * 12 + _MONTHS[m.group(1)[:3].lower()] - 1
    m = re.search(r"(\d{4})$", s)
    return int(m.group(1)) * 12 if m else None


def current_month(value):
    y, m = value.split("-")
    return int(y) * 12 + int(m) - 1


def date_ranges(text, current="2021-12"):
    """[(start, end, a_month, b_month)] for each date range."""
    cur = current_month(current)
    out = []
    for m in _DATE_RANGE.finditer(text or ""):
        a, b = parse_month(m.group("a"), cur), parse_month(m.group("b"), cur)
        if a is not None and b is not None and 0 <= b - a <= 600:
            out.append((m.start(), m.end(), a, b))
    return out


def years_from_dates(text, current="2021-12"):
    """Years covered by date ranges (overlaps merged); None if none."""
    spans = sorted((a, b + 1) for _, _, a, b in date_ranges(text, current))
    if not spans:
        return None
    total, (cs, ce) = 0, spans[0]
    for a, b in spans[1:]:
        if a <= ce:
            ce = max(ce, b)
        else:
            total += ce - cs
            cs, ce = a, b
    total += ce - cs
    return round(total / 12, 2)


class RuleExtractor:
    """Same interface and record format as extraction.Extractor."""

    model = "rules"

    def __init__(self, gazetteer, max_chars=8000, current="2021-12"):
        self.gazetteer = gazetteer
        self.max_chars = max_chars
        self.current = current
        self.constrained = False

    def extract(self, kind, doc_id, text):
        record = {"doc_id": str(doc_id), "kind": kind, "model": self.model}
        if not isinstance(text, str) or not text.strip():
            record.update(status="error", error="empty document")
            return record
        source = text[: self.max_chars] if self.max_chars else text
        record.update(source_chars=len(source), truncated=len(source) < len(text), constrained=False)
        raw = self._resume(source) if kind == "resume" else self._posting(source)
        facts, rejected = ground(kind, raw, source)
        record.update(status="ok", facts=facts, rejected=rejected)
        return record

    # --------------------------------------------------------------
    def _skills(self, text, with_importance):
        out = []
        for start, end, _ in self.gazetteer.find(text):
            item = {"name": text[start:end], "source_span": text[start:end]}
            if with_importance:
                item["importance"] = "preferred" if _PREFERRED.search(_sentence(text, start)) else "required"
            out.append(item)
        return out

    def _degrees(self, text, with_importance):
        out, seen = [], set()
        for level, pattern in DEGREE_PATTERNS:
            for m in re.finditer(pattern, text):
                if level in seen:
                    break
                seen.add(level)
                item = {"level": level, "field": None, "source_span": m.group(0)}
                if with_importance:
                    item["importance"] = "preferred" if _PREFERRED.search(_sentence(text, m.start())) else "required"
                else:
                    item.update(institution=None, year=None)
                out.append(item)
        return out

    def _resume(self, text):
        experience = []
        for start, end, a, b in date_ranges(text, self.current):
            before = re.split(r"\s{2,}|\n", text[max(0, start - 120):start].rstrip())
            title = next((p.strip() for p in reversed(before) if re.search(r"[A-Za-z]{3}", p)), None)
            if not title or len(title) > 80:
                continue
            experience.append({"title": title, "company": None, "duration_months": b - a + 1,
                               "source_span": title})
        return {"skills": self._skills(text, False), "degrees": self._degrees(text, False),
                "experience": experience}

    def _posting(self, text):
        exp = None
        for m in _YEARS.finditer(text):
            sent = _sentence(text, m.start())
            if re.search(r"experience|exp\b", sent, re.I):
                exp = {"min_years": int(m.group(1)),
                       "importance": "preferred" if _PREFERRED.search(sent) else "required",
                       "source_span": m.group(0)}
                break
        return {"required_skills": self._skills(text, True),
                "required_degrees": self._degrees(text, True), "experience_required": exp}
