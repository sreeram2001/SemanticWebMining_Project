"""Step 5: in-memory knowledge graph from grounded facts; exports to Neo4j/Cypher."""

import hashlib
import json
import re
from collections import defaultdict

from linking.entity_linking import NIL, context_of
from linking.skill_linking import normalize_name

DEGREE_RANK = {"high_school": 0, "associate": 1, "bachelors": 2, "masters": 3, "phd": 4}


def _eid(src, rel, dst, props):
    span = props.get("span_start", "")
    return hashlib.md5(f"{src}|{rel}|{dst}|{span}".encode("utf-8")).hexdigest()[:16]


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", normalize_name(text)).strip("-")[:60] or "x"


class KnowledgeGraph:
    def __init__(self, index, linker, ontology=None):
        self.index = index
        self.linker = linker
        self.ontology = ontology
        self.nodes = {}                      # key -> {"label": ..., props}
        self.edges = []                      # (src_key, type, dst_key, props)
        self.texts = {}                      # doc_id -> source text the offsets refer to
        self.kinds = {}                      # doc_id -> resume | posting
        self.meta = {}                       # doc_id -> metadata (title, company, category, ...)
        self.facts = {}                      # doc_id -> linked facts (see add_record)
        self.link_log = []                   # link decisions
        self.stale_facts = 0                 # facts with stale spans

    @classmethod
    def load_ontology(cls, path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------ build

    def _node(self, key, label, **props):
        node = self.nodes.setdefault(key, {"label": label})
        node.update({k: v for k, v in props.items() if v is not None})
        return key

    def _edge(self, src, rel, dst, **props):
        self.edges.append((src, rel, dst, props))

    def _skill_node(self, mention, context):
        res = self.linker.link(mention, context)
        sid = res["skill_id"]
        if sid != NIL:
            s = self.index.skills[sid]
            key = self._node(f"Skill:{sid}", "Skill", skill_id=sid, name=s["name"],
                             category=s.get("category"), subcategory=s.get("subcategory"), source="lightcast")
        else:
            gen = f"X-{_slug(mention)}"
            key = self._node(f"Skill:{gen}", "Skill", skill_id=gen, name=mention, source="extracted")
        self.link_log.append({"mention": mention, "skill_id": sid, "method": res["method"], "score": res["score"]})
        return key, sid, res

    def add_record(self, record, text, meta=None):
        """Add one extraction record."""
        if record.get("status") != "ok":
            return False
        doc_id, kind = str(record["doc_id"]), record["kind"]
        meta = meta or {}
        text = text or ""
        self.texts[doc_id], self.kinds[doc_id], self.meta[doc_id] = text, kind, meta
        facts = self._current_facts(record, text)
        linked = {"skills": [], "degrees": [], "experience": [], "min_years": None}

        if kind == "resume":
            doc = self._node(f"Candidate:{doc_id}", "Candidate", candidate_id=doc_id,
                             category=meta.get("category"), source_file=meta.get("source_file"))
            skill_field, skill_rel = "skills", "HAS_SKILL"
        else:
            doc = self._node(f"JobPosting:{doc_id}", "JobPosting", job_id=doc_id, title=meta.get("title"),
                             company_name=meta.get("company_name"), location=meta.get("location"),
                             experience_level=meta.get("experience_level"), work_type=meta.get("work_type"),
                             description_snippet=(text or "")[:500])
            skill_field, skill_rel = "required_skills", "REQUIRES_SKILL"
            if meta.get("company_name"):
                comp = self._node(f"Company:{meta.get('company_id') or _slug(meta['company_name'])}", "Company",
                                  company_id=str(meta.get("company_id") or _slug(meta["company_name"])),
                                  name=meta["company_name"], industry=meta.get("industry"))
                self._edge(doc, "POSTED_BY", comp)

        for f in facts.get(skill_field) or []:
            key, sid, res = self._skill_node(f["name"], context_of(text, f["start"], f["end"]))
            props = dict(source_span=f["source_span"], span_start=f["start"], span_end=f["end"],
                         mention=f["name"], link_method=res["method"], link_score=res["score"])
            if kind == "posting":
                props["importance"] = f["importance"]
            self._edge(doc, skill_rel, key, **props)
            linked["skills"].append({**f, "skill_id": sid, "skill_key": key})

        for f in facts.get("degrees" if kind == "resume" else "required_degrees") or []:
            dkey = self._node(f"Degree:{f['level']}|{_slug(f.get('field') or '')}", "Degree",
                              level=f["level"], field=f.get("field"))
            props = dict(source_span=f["source_span"], span_start=f["start"], span_end=f["end"])
            if kind == "resume":
                props.update(institution=f.get("institution"), year=f.get("year"))
                self._edge(doc, "HAS_DEGREE", dkey, **props)
            else:
                props["importance"] = f["importance"]
                self._edge(doc, "REQUIRES_DEGREE", dkey, **props)
            linked["degrees"].append(f)

        if kind == "resume":
            for i, f in enumerate(facts.get("experience") or []):
                ekey = self._node(f"Experience:{doc_id}:{i}", "Experience", title=f["title"],
                                  company=f.get("company"), duration_months=f.get("duration_months"))
                self._edge(doc, "HAS_EXPERIENCE", ekey, source_span=f["source_span"],
                           span_start=f["start"], span_end=f["end"])
                linked["experience"].append(f)
        else:
            f = facts.get("experience_required")
            if f:
                rkey = self._node(f"Requirement:{doc_id}:min_years", "Requirement",
                                  kind="min_years_experience", value=float(f["min_years"]),
                                  importance=f["importance"])
                self._edge(doc, "HAS_REQUIREMENT", rkey, source_span=f["source_span"],
                           span_start=f["start"], span_end=f["end"])
                linked["min_years"] = f
        self.facts[doc_id] = linked
        return True

    def _current_facts(self, record, text):
        """Facts whose span still matches (stale ones counted)."""
        out = {}
        for field, value in (record.get("facts") or {}).items():
            items = value if isinstance(value, list) else ([value] if value else [])
            ok = [f for f in items if isinstance(f, dict) and text[f.get("start", 0):f.get("end", -1)] == f.get("source_span")]
            self.stale_facts += len(items) - len(ok)
            out[field] = ok if isinstance(value, list) else (ok[0] if ok else None)
        return out

    def add_related_edges(self, lambda_sub, lambda_cat=None):
        """RELATED_TO edges between required/held skills sharing a (sub)category."""
        def linked(rel):
            return {dst for _, r, dst, _ in self.edges if r == rel and self.nodes[dst].get("source") == "lightcast"}
        required, held = linked("REQUIRES_SKILL"), linked("HAS_SKILL")
        groups = [("subcategory", "same_subcategory", lambda_sub)]
        if lambda_cat is not None:
            groups.append(("category", "same_category", lambda_cat))
        seen, added = set(), 0
        for prop, kind, weight in groups:
            held_by = defaultdict(list)
            for k in held:
                if self.nodes[k].get(prop):
                    held_by[self.nodes[k][prop]].append(k)
            for a in sorted(required):
                for b in held_by.get(self.nodes[a].get(prop), ()):
                    pair = tuple(sorted((a, b)))
                    if a == b or pair in seen:
                        continue
                    seen.add(pair)
                    self._edge(pair[0], "RELATED_TO", pair[1], relation_type=kind, weight=weight)
                    added += 1
        return added

    # ------------------------------------------------------------------ queries

    def linked_skills(self, doc_id):
        """{skill_id: first fact} for linked skills."""
        out = {}
        for f in self.facts.get(doc_id, {}).get("skills", []):
            if f["skill_id"] != NIL:
                out.setdefault(f["skill_id"], f)
        return out

    def skill_requirements(self, job_id):
        """Unique skill requirements ('required' wins)."""
        out = {}
        for f in self.facts.get(job_id, {}).get("skills", []):
            key = f["skill_id"] if f["skill_id"] != NIL else f"NIL:{normalize_name(f['name'])}"
            if key not in out or (f["importance"] == "required" and out[key]["importance"] != "required"):
                out[key] = f
        return list(out.values())

    def degrees(self, doc_id):
        return self.facts.get(doc_id, {}).get("degrees", [])

    def min_years(self, job_id):
        return self.facts.get(job_id, {}).get("min_years")

    def jobs_of_company(self, name):
        return [k.split(":", 1)[1] for k, n in self.nodes.items()
                if n["label"] == "JobPosting" and n.get("company_name") == name]

    def doc_ids(self, kind):
        return [d for d, k in self.kinds.items() if k == kind]

    # ------------------------------------------------------------------ checks

    def integrity_errors(self):
        """Ontology/provenance problems (empty = consistent)."""
        errors = []
        rel_types = (self.ontology or {}).get("relationship_types", {})
        labels = set((self.ontology or {}).get("node_types", {}))
        for key, n in self.nodes.items():
            if labels and n["label"] not in labels:
                errors.append(f"{key}: label {n['label']} not in ontology")
        for src, rel, dst, props in self.edges:
            if rel_types:
                spec = rel_types.get(rel)
                if spec is None:
                    errors.append(f"{src} -{rel}-> {dst}: relationship not in ontology")
                    continue
                if self.nodes[src]["label"] != spec["from"] or self.nodes[dst]["label"] != spec["to"]:
                    errors.append(f"{src} -{rel}-> {dst}: expected {spec['from']} -> {spec['to']}")
            if "span_start" in props:
                doc_id = src.split(":", 1)[1]
                text = self.texts.get(doc_id)
                if text is None or text[props["span_start"]:props["span_end"]] != props["source_span"]:
                    errors.append(f"{src} -{rel}-> {dst}: source_span does not match the source text")
        return errors

    def stats(self):
        labels, rels = defaultdict(int), defaultdict(int)
        for n in self.nodes.values():
            labels[n["label"]] += 1
        for _, rel, _, _ in self.edges:
            rels[rel] += 1
        linked = sum(1 for e in self.link_log if e["skill_id"] != NIL)
        return {"nodes": dict(labels), "edges": dict(rels), "stale_facts_dropped": self.stale_facts,
                "skill_mentions": len(self.link_log),
                "linked_rate": round(linked / len(self.link_log), 4) if self.link_log else None}

    # ------------------------------------------------------------------ Neo4j

    def _rows(self):
        nodes = defaultdict(list)
        for key, n in self.nodes.items():
            nodes[n["label"]].append({"key": key, "props": {k: v for k, v in n.items() if k != "label"}})
        rels = defaultdict(list)
        for src, rel, dst, props in self.edges:
            rels[(rel, self.nodes[src]["label"], self.nodes[dst]["label"])].append(
                {"src": src, "dst": dst, "eid": _eid(src, rel, dst, props), "props": props})
        return nodes, rels

    def load_neo4j(self, uri, user, password, batch=1000, clear=False):
        """Idempotent batched MERGE into Neo4j."""
        from neo4j import GraphDatabase
        nodes, rels = self._rows()
        with GraphDatabase.driver(uri, auth=(user, password)) as driver, driver.session() as s:
            if clear:
                s.run("MATCH (n) DETACH DELETE n")
            for label in nodes:
                s.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.key IS UNIQUE")
            for label, rows in nodes.items():
                for i in range(0, len(rows), batch):
                    s.run(f"UNWIND $rows AS r MERGE (n:{label} {{key: r.key}}) SET n += r.props",
                          rows=rows[i:i + batch])
            for (rel, a, b), rows in rels.items():
                for i in range(0, len(rows), batch):
                    s.run(f"UNWIND $rows AS r MATCH (a:{a} {{key: r.src}}), (b:{b} {{key: r.dst}}) "
                          f"MERGE (a)-[e:{rel} {{eid: r.eid}}]->(b) SET e += r.props", rows=rows[i:i + batch])
        return {"nodes": sum(len(v) for v in nodes.values()), "edges": len(self.edges)}

    def to_cypher(self, path):
        """Idempotent Cypher script."""
        def lit(props):
            return "{" + ", ".join(f"`{k}`: {json.dumps(v)}" for k, v in props.items() if v is not None) + "}"
        lines = []
        for label in sorted({n["label"] for n in self.nodes.values()}):
            lines.append(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.key IS UNIQUE;")
        for key, n in self.nodes.items():
            props = {k: v for k, v in n.items() if k != "label"}
            lines.append(f"MERGE (n:{n['label']} {{key: {json.dumps(key)}}}) SET n += {lit(props)};")
        for src, rel, dst, props in self.edges:
            lines.append(f"MATCH (a:{self.nodes[src]['label']} {{key: {json.dumps(src)}}}), "
                         f"(b:{self.nodes[dst]['label']} {{key: {json.dumps(dst)}}}) "
                         f"MERGE (a)-[e:{rel} {{eid: {json.dumps(_eid(src, rel, dst, props))}}}]->(b) "
                         f"SET e += {lit(props)};")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return len(lines)


# G(r, j) in Cypher (no missed-skill check).
GRAPH_SCORE_CYPHER = """
MATCH (j:JobPosting {key: $job})-[r:REQUIRES_SKILL]->(s:Skill {source: 'lightcast'})
WITH s, max(CASE r.importance WHEN 'required' THEN $w_req ELSE $w_pref END) AS w
OPTIONAL MATCH (:Candidate {key: $cand})-[:HAS_SKILL]->(t:Skill {source: 'lightcast'})
WITH s, w, collect(DISTINCT t) AS ts
WITH s, w, reduce(best = 0.0, t IN ts |
  CASE
    WHEN t.key = s.key THEN 1.0
    WHEN t.subcategory IS NOT NULL AND t.subcategory = s.subcategory AND $lam_sub > best THEN $lam_sub
    WHEN t.category IS NOT NULL AND t.category = s.category AND $lam_cat > best THEN $lam_cat
    ELSE best END) AS credit
RETURN sum(w * credit) / sum(w) AS G
"""
