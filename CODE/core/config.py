"""Settings (config.json + optional override) and shared helpers. Paths are repo-relative."""

import copy
import json
import math
import os
import tempfile

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(CODE_DIR)
DEFAULT_PATH = os.path.join(CODE_DIR, "config.json")

SECTIONS = ("paths", "corpus", "extraction", "linking", "retrieval", "rerank", "scoring", "evaluation",
            "labeling", "neo4j", "strict_models")


class ConfigError(ValueError):
    pass


def _merge(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None, overrides=None):
    """Load config.json, apply `path` and `overrides`, validate."""
    with open(DEFAULT_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    if path:
        if not os.path.exists(path):
            raise ConfigError(f"config file not found: {path}")
        with open(path, encoding="utf-8") as f:
            try:
                _merge(cfg, json.load(f))
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    if overrides:
        _merge(cfg, copy.deepcopy(overrides))
    validate(cfg)
    return cfg


def validate(cfg):
    """Raise ConfigError for invalid or misleading settings."""
    problems = []
    unknown = [k for k in cfg if k not in SECTIONS and not k.startswith("_")]
    if unknown:
        problems.append(f"unknown top-level keys {unknown} (typo?)")
    for sec in SECTIONS[:-1]:
        if not isinstance(cfg.get(sec), dict):
            problems.append(f"missing section '{sec}'")
    if problems:
        raise ConfigError("; ".join(problems))
    sc, ev, r = cfg["scoring"], cfg["evaluation"], cfg["retrieval"]
    if not 0 < sc["lambda_cat"] < sc["lambda_sub"] < 1:
        problems.append("scoring: need 0 < lambda_cat < lambda_sub < 1")
    for k in ("beta_ce", "beta_graph", "beta_penalty", "w_required", "w_preferred"):
        if not isinstance(sc.get(k), (int, float)) or sc[k] < 0:
            problems.append(f"scoring.{k} must be a number >= 0")
    if sc.get("w_preferred", 0) > sc.get("w_required", 1):
        problems.append("scoring: w_preferred should not exceed w_required")
    if not 0 < sc.get("fuzzy_threshold", 0.9) <= 1:
        problems.append("scoring.fuzzy_threshold must be in (0, 1]")
    try:
        y, m = str(sc.get("current_date", "")).split("-")
        assert 1990 < int(y) < 2100 and 1 <= int(m) <= 12
    except (ValueError, AssertionError):
        problems.append("scoring.current_date must look like YYYY-MM")
    if cfg["extraction"].get("mode") not in ("llm", "rules"):
        problems.append("extraction.mode must be 'llm' or 'rules'")
    if r.get("query") not in ("skills", "text"):
        problems.append("retrieval.query must be 'skills' or 'text'")
    for key, sec in (("first_stage_k", r), ("rrf_k", r), ("top_n", cfg["rerank"]), ("k", ev)):
        if not isinstance(sec.get(key), int) or sec[key] < 1:
            problems.append(f"{key} must be a positive integer")
    if not 0 < ev.get("dev_ratio", 0.5) < 1:
        problems.append("evaluation.dev_ratio must be between 0 and 1")
    l = cfg["linking"]
    if not 0 <= l.get("llm_reject", 0.3) <= l.get("llm_accept", 0.9):
        problems.append("linking: need 0 <= llm_reject <= llm_accept")
    n = cfg["corpus"].get("n_postings")
    if n is not None and (not isinstance(n, int) or n < 1):
        problems.append("corpus.n_postings must be a positive integer or null")
    if len(set(cfg["labeling"].get("annotators", []))) < 2:
        problems.append("labeling.annotators needs at least two distinct names")
    if problems:
        raise ConfigError("invalid config: " + "; ".join(problems))
    return cfg


def resolve(path):
    """Repo-relative path -> absolute."""
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


# ------------------------------------------------------------------ helpers

def write_text_atomic(path, text):
    """Atomic write via temp file + rename."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp_", suffix=os.path.basename(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_json_atomic(path, obj, indent=1):
    write_text_atomic(path, json.dumps(obj, indent=indent, default=str, ensure_ascii=False))


def read_json_or(path, default, label="file"):
    """Load JSON, or `default` if missing; corrupt files are moved aside."""
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        bad = path + ".corrupt"
        os.replace(path, bad)
        print(f"[warning] {label} at {path} was unreadable; moved to {bad} and starting fresh")
        return default


def norm_id(value):
    """Clean ID string: 3884428798.0 -> '3884428798'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    s = str(value).strip()
    return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s


def as_text(value):
    """Text from a CSV cell; NaN and None become ''."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)
