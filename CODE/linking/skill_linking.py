"""Step 4a: download/cache Lightcast Open Skills and build a name -> ID index."""

import json
import os
import re
import time
from collections import Counter, defaultdict

TOKEN_URL = "https://auth.emsicloud.com/connect/token"
# Current host first, legacy host as fallback.
SKILLS_URLS = (
    "https://api.lightcast.io/skills/versions/{version}/skills",
    "https://emsiservices.com/skills/versions/{version}/skills",
)
SCOPE = "emsi_open"

# Includes hierarchy; falls back to default fields if rejected.
FULL_FIELDS = "id,name,type,category,subcategory,description,infoUrl"

DEFAULT_CACHE = "DATA/lightcast/skills.json"


# ---------------------------------------------------------------- download

def fetch_token(client_id, client_secret, scope=SCOPE, session=None):
    """Get an OAuth token (valid 1h)."""
    session = session or _requests()
    resp = session.post(
        TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
            "scope": scope,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def download_skills(token, version="latest", session=None, retries=3, backoff=2.0):
    """Return (raw_skills, fields_used, url), with host/field fallbacks."""
    session = session or _requests()
    headers = {"Authorization": f"Bearer {token}"}
    last_error = None
    for template in SKILLS_URLS:
        url = template.format(version=version)
        try:
            raw, fields = _download_from(session, url, headers, retries, backoff)
            return raw, fields, url
        except _HostProblem as exc:
            last_error = exc
    raise RuntimeError(f"skills download failed on every host: {last_error}")


class _HostProblem(Exception):
    pass


def _download_from(session, url, headers, retries, backoff):
    for fields in (FULL_FIELDS, None):
        params = {"fields": fields} if fields else {}
        for attempt in range(retries + 1):
            try:
                resp = session.get(url, headers=headers, params=params, timeout=120)
            except Exception as exc:  # DNS, connection refused, timeout
                if "Connection" in type(exc).__name__ or "Timeout" in type(exc).__name__:
                    raise _HostProblem(f"{url}: {type(exc).__name__}") from exc
                raise
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == retries:
                    raise _HostProblem(f"{url}: HTTP {resp.status_code}")
                time.sleep(backoff * 2 ** attempt)
                continue
            break
        if resp.status_code == 404:
            raise _HostProblem(f"{url}: HTTP 404")
        if resp.status_code in (400, 422) and fields:
            continue
        resp.raise_for_status()  # 401/403: bad credentials
        return resp.json()["data"], fields or "default"
    raise _HostProblem(f"{url}: rejected both field lists")


def _requests():
    import requests  # lazy: tests run without it
    return requests.Session()


# ---------------------------------------------------------------- records

def _name_of(value):
    """Unwrap Lightcast {"id", "name"} objects."""
    if isinstance(value, dict):
        return value.get("name")
    return value


def simplify(raw):
    """Keep only fields the linker and graph use."""
    return {
        "id": raw["id"],
        "name": raw["name"],
        "type": _name_of(raw.get("type")),
        "category": _name_of(raw.get("category")),
        "subcategory": _name_of(raw.get("subcategory")),
        "description": raw.get("description"),
    }


def save_skills(skills, path=DEFAULT_CACHE, version="latest", fields="default"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "source": "Lightcast Open Skills",
        "version": version,
        "fields": fields,
        "fetched": time.strftime("%Y-%m-%d"),
        "count": len(skills),
        "skills": skills,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


def load_skills(path=DEFAULT_CACHE):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["skills"]


def download_and_cache(client_id, client_secret, path=DEFAULT_CACHE, version="latest", session=None):
    """Token, download, simplify, save."""
    token = fetch_token(client_id, client_secret, session=session)
    raw, fields, _ = download_skills(token, version=version, session=session)
    skills = [simplify(r) for r in raw]
    save_skills(skills, path, version=version, fields=fields)
    return skills


# ---------------------------------------------------------------- names

_QUALIFIER = re.compile(r"\s*\([^)]*\)\s*$")
_CHAR_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_EDGE_PUNCT = " \t\n.,;:'\"*•-"


def normalize_name(text):
    """Lower-case, unify quotes/dashes, collapse spaces, trim edge punctuation."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text.translate(_CHAR_MAP)).strip(_EDGE_PUNCT)
    return text.lower()


def strip_qualifier(name):
    """'Python (Programming Language)' -> 'Python'."""
    return _QUALIFIER.sub("", name).strip() if isinstance(name, str) else name


def name_keys(skill):
    """Normalized full and unqualified names."""
    keys = {normalize_name(skill["name"])}
    bare = normalize_name(strip_qualifier(skill["name"]))
    if bare:
        keys.add(bare)
    keys.discard("")
    return keys


def build_name_index(skills):
    """Normalized key -> skill IDs (>1 = ambiguous)."""
    index = defaultdict(list)
    for skill in skills:
        for key in name_keys(skill):
            if skill["id"] not in index[key]:
                index[key].append(skill["id"])
    return dict(index)


def by_id(skills):
    return {s["id"]: s for s in skills}


def taxonomy_stats(skills):
    """Summary counts to sanity-check the download."""
    index = build_name_index(skills)
    return {
        "skills": len(skills),
        "types": dict(Counter(s.get("type") for s in skills)),
        "categories": len({s["category"] for s in skills if s.get("category")}),
        "subcategories": len({s["subcategory"] for s in skills if s.get("subcategory")}),
        "missing_category": sum(1 for s in skills if not s.get("category")),
        "with_description": sum(1 for s in skills if s.get("description")),
        "index_keys": len(index),
        "ambiguous_keys": sum(1 for ids in index.values() if len(ids) > 1),
    }
