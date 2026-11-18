"""End-to-end pipeline: corpus -> split -> extract -> link -> graph -> rank -> evaluate.

    p = Pipeline(load_config()); p.extract(); p.build(); p.recommend("12345"); p.evaluate("dev")
"""

import csv
import os

from dataprep import labeling
from core.config import as_text, norm_id, resolve, write_text_atomic
from linking.entity_linking import (Gazetteer, SkillIndex, SkillLinker, evaluate_linking, linking_rows, load_linking_gold,
                            tune_min_score)
from assistant.explanations import explain, skill_gaps
from extract.extraction import load_records, run_batch, summarize
from kg.graph import KnowledgeGraph
from dataprep.preprocess import file_hash, load_split, make_split, sample_postings, save_split
from rank.ranking import METHODS, Ranker
from rank.ranking_eval import (DEFAULT_GRID, auto_qrels, bootstrap_ci, evaluate_ranking, load_graded_qrels, pool_rows,
                          randomization_test, report_markdown, run_log, save_json, tune)
from extract.rule_extractor import RuleExtractor

csv.field_size_limit(10 ** 8)

POSTING_COLUMNS = ("job_id", "title", "description")
RESUME_COLUMNS = ("ID", "Category")


class InputError(RuntimeError):
    """Missing or malformed input (message says how to fix)."""


def _read_csv(path, required, hint):
    if not path or not os.path.exists(path):
        raise InputError(f"missing {path}: {hint}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise InputError(f"{path} lacks columns {missing}: {hint}")
        return list(reader)


class Pipeline:
    def __init__(self, cfg, use_models=True):
        self.cfg = cfg
        self.use_models = use_models
        self.p = {k: resolve(v) for k, v in cfg["paths"].items()}
        self._corpus = self._index = self._linker = self.graph = self._ranker = None

    @property
    def strict(self):
        return bool(self.cfg.get("strict_models"))

    # ------------------------------------------------------------------ corpus
    @property
    def corpus(self):
        """Working corpus: {'postings': {id: row}, 'resumes': {id: row}}."""
        if self._corpus is None:
            c = self.cfg["corpus"]
            hint = "run `run_pipeline.py preprocess` (or the notebook's Step 1) first"
            postings = _read_csv(self.p["postings"], POSTING_COLUMNS, hint)
            resumes = _read_csv(self.p["resumes"], RESUME_COLUMNS, hint)
            if resumes and "Resume_clean" not in resumes[0]:
                if "Resume_str" not in resumes[0]:
                    raise InputError(f"{self.p['resumes']} has neither Resume_clean nor Resume_str: {hint}")
                from dataprep.preprocess import mask_pii
                for r in resumes:
                    r["Resume_clean"] = mask_pii(as_text(r["Resume_str"]))
            postings = [r for r in postings if norm_id(r["job_id"]) and as_text(r["description"]).strip()]
            resumes = [r for r in resumes if norm_id(r["ID"]) and as_text(r["Resume_clean"]).strip()]
            n = c.get("n_postings")
            if n and n < len(postings):
                import pandas as pd
                postings = sample_postings(pd.DataFrame(postings), n, c["seed"]).to_dict("records")
            if c.get("n_resumes"):
                resumes = resumes[: c["n_resumes"]]
            self._corpus = {"postings": {norm_id(r["job_id"]): r for r in postings},
                            "resumes": {norm_id(r["ID"]): r for r in resumes}}
            if not self._corpus["postings"] or not self._corpus["resumes"]:
                raise InputError("the corpus is empty after dropping rows without an ID or text")
        return self._corpus

    def text(self, kind, doc_id):
        row = self.corpus["resumes" if kind == "resume" else "postings"][doc_id]
        return as_text(row["Resume_clean"] if kind == "resume" else row["description"])

    # ------------------------------------------------------------------ split
    def split(self):
        """Frozen dev/test resume split (never overwritten)."""
        path = self.p["split"]
        if not os.path.exists(path):
            ev = self.cfg["evaluation"]
            s = make_split([(rid, r["Category"]) for rid, r in self.corpus["resumes"].items()],
                           ev["dev_ratio"], ev["seed"])
            save_split(s, path, ev["seed"], ev["dev_ratio"])
        return load_split(path)

    def split_ids(self, part):
        """Graph resume IDs in one split half."""
        s = self.split()
        if self.graph is not None:
            outside = [r for r in self.graph.doc_ids("resume") if r not in s]
            if outside:
                print(f"[warning] {len(outside)} resumes in the graph are not in {self.p['split']} "
                      "(added after the split was frozen); they are left out of tuning and evaluation")
        graph_ids = self.graph.texts if self.graph is not None else self.corpus["resumes"]
        return [rid for rid, v in s.items() if v == part and rid in graph_ids]

    # ------------------------------------------------------------------ extraction
    @property
    def index(self):
        if self._index is None:
            path = self.p["skills"]
            if not os.path.exists(path):
                raise InputError(f"missing {path}: download Lightcast Open Skills with "
                                 "skill_linking.download_and_cache(...) (notebook Step 4a / Colab cell)")
            self._index = SkillIndex.load(path, self.p.get("aliases"))
        return self._index

    def llm_client(self):
        """(client, model) for ASU Voyager, or (None, None) without a key."""
        e = self.cfg["extraction"]
        key = os.environ.get(e["api_key_env"])
        if not key:
            return None, None
        try:
            from openai import OpenAI
        except ImportError:
            if self.strict:
                raise InputError("openai is not installed: pip install openai")
            print("[warning] openai is not installed; no LLM")
            return None, None
        return OpenAI(api_key=key, base_url=e["base_url"]), e["model"]

    def extractor(self):
        e = self.cfg["extraction"]
        if e["mode"] == "rules":
            return RuleExtractor(Gazetteer(self.index), e.get("max_chars", 8000),
                                 self.cfg["scoring"].get("current_date", "2021-12"))
        from extract.extraction import Extractor
        client, model = self.llm_client()
        if client is None:
            raise InputError(f"set {e['api_key_env']} (ASU Voyager key), or extract with --rules")
        return Extractor(client, model, max_chars=e.get("max_chars", 8000))

    def extract(self, limit=None):
        """Grounded extraction of the corpus (resumable)."""
        ex = self.extractor()
        out = {}
        for kind, key, rows in (("posting", "extracted_postings", self.corpus["postings"]),
                                ("resume", "extracted_resumes", self.corpus["resumes"])):
            docs = ((d, self.text(kind, d)) for d in rows)
            out[kind] = run_batch(ex, kind, docs, self.p[key], limit=limit, log_every=100)
            out[kind]["summary"] = summarize(load_records(self.p[key]))
        return out

    # ------------------------------------------------------------------ linking + graph
    @property
    def linker(self):
        if self._linker is None:
            l = self.cfg["linking"]
            client = model = None
            if l.get("use_llm"):
                client, model = self.llm_client()
                if client is None:
                    msg = (f"linking.use_llm is on but {self.cfg['extraction']['api_key_env']} is not set; "
                           "linking falls back to the score threshold")
                    if self.strict:
                        raise InputError(msg)
                    print("[warning] " + msg)
            self._linker = SkillLinker(
                self.index, l["top_k"], l["min_score"], l.get("margin", 0.0),
                l.get("dense_model") if self.use_models else None,
                l.get("cross_encoder") if self.use_models else None, client, model,
                llm_accept=l.get("llm_accept", 0.9), llm_reject=l.get("llm_reject", 0.3),
                cache_path=resolve(l.get("llm_cache")), emb_cache=resolve(l.get("emb_cache")),
                strict=self.strict)
        return self._linker

    def build(self, neo4j=False, cypher_path=None):
        g = KnowledgeGraph(self.index, self.linker, KnowledgeGraph.load_ontology(self.p["ontology"]))
        counts = {"posting": 0, "resume": 0, "not_in_corpus": 0, "extraction_errors": 0}
        try:
            for kind, key, rows in (("posting", "extracted_postings", self.corpus["postings"]),
                                    ("resume", "extracted_resumes", self.corpus["resumes"])):
                if not os.path.exists(self.p[key]):
                    raise InputError(f"no extraction output at {self.p[key]}: run the extract step first")
                for rec in load_records(self.p[key]):
                    doc_id = norm_id(rec.get("doc_id"))
                    row = rows.get(doc_id)
                    if row is None:
                        counts["not_in_corpus"] += 1
                        continue
                    if rec.get("status") != "ok" or rec.get("kind") != kind:
                        counts["extraction_errors"] += 1
                        continue
                    if kind == "posting":
                        meta = {"title": as_text(row.get("title")), "company_name": as_text(row.get("company_name")),
                                "company_id": norm_id(row.get("company_id")), "location": as_text(row.get("location")),
                                "experience_level": as_text(row.get("formatted_experience_level")),
                                "work_type": as_text(row.get("formatted_work_type"))}
                    else:
                        meta = {"category": as_text(row.get("Category")), "source_file": f"Resume.csv#ID={doc_id}"}
                    g.add_record({**rec, "doc_id": doc_id}, self.text(kind, doc_id), meta)
                    counts[kind] += 1
        finally:
            self.linker.save_cache()                     # keep LLM answers on failure
        if not counts["posting"] or not counts["resume"]:
            raise InputError(f"the graph needs postings and resumes, got {counts}; check the extraction output")
        g.add_related_edges(self.cfg["scoring"]["lambda_sub"])
        self.graph, self._ranker = g, None
        errors = g.integrity_errors()
        info = {"loaded": counts, "stats": g.stats(), "integrity_errors": len(errors),
                "integrity_examples": errors[:10], "linker": self.linker.backend,
                "llm_link_calls": self.linker.llm_calls}
        if cypher_path:
            info["cypher_statements"] = g.to_cypher(cypher_path)
        if neo4j:
            n = self.cfg["neo4j"]
            password = os.environ.get(n["password_env"])
            if not password:
                raise InputError(f"set {n['password_env']} to load Neo4j")
            info["neo4j"] = g.load_neo4j(n["uri"], n["user"], password, clear=True)
        return info

    @property
    def ranker(self):
        if self._ranker is None:
            if self.graph is None:
                self.build()
            self._ranker = Ranker(self.graph, self.cfg, self.use_models, strict=self.strict)
        return self._ranker

    # ------------------------------------------------------------------ outputs
    def recommend(self, resume_id, method="full", top=10, scoring=None):
        r = self.ranker
        resume_id = norm_id(resume_id)
        if resume_id not in self.graph.texts:
            raise InputError(f"resume {resume_id} is not in the graph (not in the corpus, or extraction failed)")
        ranked = r.rank(resume_id, method, scoring)
        jobs = []
        for row in ranked[:top]:
            meta = self.graph.meta.get(row["job_id"], {})
            jobs.append({"rank": row["rank"], "job_id": row["job_id"], "title": meta.get("title"),
                         "company": meta.get("company_name"), "score": round(row["score"], 4),
                         **({"G": round(row["G"], 3), "P": round(row["P"], 3), "CE": round(row["ce"], 3)}
                            if "G" in row else {})})
        out = {"resume_id": resume_id, "method": method, "jobs": jobs}
        if method in ("full", "full_no_graph") and ranked:
            out["explanation_top1"] = explain(r, resume_id, ranked[0]["job_id"], scoring)
            out["skill_gaps"] = skill_gaps(r, resume_id, ranked, top=min(5, top), scoring=scoring)
        return out

    def qrels(self, tier):
        if tier == "auto":
            return auto_qrels({rid: as_text(row["Category"]) for rid, row in self.corpus["resumes"].items()},
                              {jid: as_text(row["title"]) for jid, row in self.corpus["postings"].items()
                               if jid in self.graph.texts})
        return load_graded_qrels(self.p["graded_qrels"])

    def evaluate(self, split="dev", tiers=("auto", "graded"), scoring=None, tag=None):
        """Ranking, graph ablation, linking, extraction misses, run log."""
        ev = self.cfg["evaluation"]
        r = self.ranker
        ids = self.split_ids(split)
        if not ids:
            raise InputError(f"no {split} resumes are in the graph")
        result = {"log": run_log(self.cfg, r.backends, file_hash(self.p["postings"], self.p["resumes"],
                                                                self.p["skills"]), split,
                                 {"scoring_override": scoring}),
                  "graph": self.graph.stats(), "integrity_errors": len(self.graph.integrity_errors())}
        md = [f"## Evaluation on the {split} split", "",
              f"Backends: {r.backends}", f"Data hash: {result['log']['data_hash']}", ""]
        for tier in tiers:
            q = self.qrels(tier)
            res = evaluate_ranking(r, q, ids, METHODS, ev["k"], scoring, ev["bootstrap"], ev["permutations"],
                                   ev["seed"]) if q else None
            if not res or not res["n_resumes"]:
                md.append(f"_Tier '{tier}': no labeled {split} resumes yet._\n")
                continue
            result[f"ranking_{tier}"] = res
            md += [report_markdown(res, f"Ranking, tier {'1 (auto labels)' if tier == 'auto' else '2 (graded)'}",
                                   ev["k"]), ""]
        result["extraction_misses"] = {"count": len(r.miss_log), "examples": list(r.miss_log.values())[:20]}
        md.append(f"Extraction misses recovered from raw resume text: {len(r.miss_log)}\n")

        if self.p.get("linking_gold") and os.path.exists(self.p["linking_gold"]):
            rows = load_linking_gold(self.p["linking_gold"], split)
            base = evaluate_linking(SkillLinker(self.index, alias_only=True), rows)
            ours = evaluate_linking(self.linker, rows)
            if base and ours:
                ci = bootstrap_ci(ours["per_mention"], ev["bootstrap"], ev["seed"])
                p = randomization_test(ours["per_mention"], base["per_mention"], ev["permutations"], ev["seed"])
                result["linking"] = {
                    "alias_only": {k: v for k, v in base.items() if k != "per_mention"},
                    "entity_linker": {k: v for k, v in ours.items() if k != "per_mention"},
                    "accuracy_ci95": ci, "p_value_vs_alias": p}
                ci_txt = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
                md += ["### Skill linking", "| Linker | Accuracy@1 | NIL accuracy | n |", "|---|---|---|---|",
                       f"| alias only (baseline) | {base['accuracy@1']:.3f} | {_f(base['nil_accuracy'])} | {base['n']} |",
                       f"| entity linker ({self.linker.backend}) | {ours['accuracy@1']:.3f}{ci_txt} | "
                       f"{_f(ours['nil_accuracy'])} | {ours['n']} |", f"\nPaired randomization p = {_f(p, 4)}", ""]

        self.linker.save_cache()
        out = os.path.join(self.p["out_dir"], f"eval_{tag or split}")
        save_json(result, out + ".json")
        write_text_atomic(out + ".md", "\n".join(md) + "\n")
        result["markdown"] = "\n".join(md)
        result["paths"] = [out + ".json", out + ".md"]
        return result

    def evaluate_extraction(self, models=None):
        """Extraction P/R/F1 on gold, grounded vs. ungrounded. `models` defaults to the configured one."""
        from extract import evaluation
        L = self.cfg["labeling"]
        gold_dir, second_dir = resolve(L["gold_dir"]), resolve(L["gold_second_dir"])
        gold, problems, unlabeled = evaluation.load_gold(gold_dir) if os.path.isdir(gold_dir) else ({}, [], 0)
        if not gold:
            raise InputError(f"no labeled gold files in {gold_dir}: run `sheets --task extraction` and label them")
        out_dir = os.path.join(self.p["out_dir"], "extraction_eval")
        if models:
            client, _ = self.llm_client()
            if client is None:
                raise InputError(f"set {self.cfg['extraction']['api_key_env']} to compare LLMs")
            results = evaluation.compare_models(client, models, gold_dir, out_dir)
        else:
            ex = self.extractor()
            ex.max_chars = None                          # score whole documents
            results = {ex.model: {}}
            for kind in ("resume", "posting"):
                docs = evaluation.gold_docs(gold_dir, kind)
                if not docs:
                    continue
                path = os.path.join(out_dir, f"{evaluation.run_slug(ex.model)}_{kind}.jsonl")
                counts = run_batch(ex, kind, docs, path, log_every=0)
                results[ex.model][kind] = {"run": counts, "facts": summarize(load_records(path)),
                                           **evaluation.evaluate_run(path, gold, kind)}
        rows = evaluation.comparison_rows(results)
        agree = evaluation.agreement(gold_dir, second_dir) if os.path.isdir(second_dir) else {}
        md = ["## Extraction (gold set)", "", f"Labeled documents: {len(gold)}; gold problems: {len(problems)}", "",
              "| Model | Kind | Mode | Macro-F1 [95% CI] | Unsupported rate |", "|---|---|---|---|---|"]
        for r in rows:
            ci = f" [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]" if r["ci95"] else ""
            md.append(f"| {r['model']} | {r['kind']} | {r['mode']} | {_f(r['macro_f1'])}{ci} | "
                      f"{_f(r['unsupported_rate'])} |")
        for kind, res in agree.items():
            md.append(f"\nInter-annotator agreement ({kind}, {res['documents']} docs): macro-F1 {_f(res['macro_f1'])}")
        for model in results:                            # per_doc is bulky; don't save it
            for entry in results[model].values():
                for mode in ("grounded", "ungrounded"):
                    if entry.get(mode):
                        entry[mode].pop("per_doc", None)
        result = {"results": results, "rows": rows, "agreement": agree, "gold_problems": problems,
                  "unlabeled": unlabeled}
        path = os.path.join(self.p["out_dir"], "eval_extraction")
        save_json(result, path + ".json")
        write_text_atomic(path + ".md", "\n".join(md) + "\n")
        return {**result, "markdown": "\n".join(md), "paths": [path + ".json", path + ".md"]}

    # ------------------------------------------------------------------ chat / demo
    def add_resume(self, text, category=None):
        """Extract, link and add a user resume to the graph; returns its ID (LLM if keyed, else rules)."""
        from dataprep.preprocess import mask_pii
        text = mask_pii(as_text(text)).strip()
        if len(text) < 30:
            raise InputError("the resume text is too short")
        import hashlib
        rid = "user-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        g = self.graph_or_build()
        if rid in g.texts:
            return rid
        try:
            ex = self.extractor()
        except InputError:
            e = self.cfg["extraction"]
            ex = RuleExtractor(Gazetteer(self.index), e.get("max_chars", 8000),
                               self.cfg["scoring"].get("current_date", "2021-12"))
        rec = ex.extract("resume", rid, text)
        if rec.get("status") != "ok":
            raise InputError(f"could not extract the resume: {rec.get('error')}")
        g.add_record(rec, text, {"category": category, "source_file": "uploaded"})
        self.linker.save_cache()
        return rid

    def tune(self, grid=None, tier="auto"):
        """Tune beta/lambda on dev; writes tuned_scoring.json."""
        q = self.qrels(tier)
        dev = [rid for rid in self.split_ids("dev") if any(g > 0 for g in q.get(rid, {}).values())]
        if not dev:
            raise InputError(f"no dev resumes with {tier} relevance labels to tune on")
        best, score, tried = tune(self.ranker, q, dev, grid or DEFAULT_GRID, self.cfg["evaluation"]["k"])
        path = os.path.join(self.p["out_dir"], "tuned_scoring.json")
        save_json({"tier": tier, "dev_nDCG": score, "configs_tried": tried, "scoring": best}, path)
        return {"best": best, "dev_nDCG": score, "tried": tried, "path": path}

    def tune_linking(self):
        """Tune min_score (and LLM band) on dev linking labels."""
        if not os.path.exists(self.p["linking_gold"]):
            raise InputError(f"missing {self.p['linking_gold']}: label the linking sheets and run merge-labels")
        rows = load_linking_gold(self.p["linking_gold"], "dev")
        if not rows:
            raise InputError("no labeled dev rows in the linking gold file yet")
        l = self.cfg["linking"]
        best, curve = tune_min_score(self.index, rows, top_k=l["top_k"], margin=l.get("margin", 0.0))
        out = {"min_score": best, "dev_rows": len(rows), "curve": curve}
        lk = self.linker
        if lk.llm_client is not None:
            band = []
            try:
                for reject in (0.2, 0.3, 0.4):
                    for accept in (0.8, 0.85, 0.9, 0.95, 1.01):
                        lk.set_params(llm_reject=reject, llm_accept=accept, min_score=best)
                        band.append((reject, accept, evaluate_linking(lk, rows)["accuracy@1"]))
            finally:
                lk.set_params(llm_reject=l.get("llm_reject", 0.3), llm_accept=l.get("llm_accept", 0.9),
                              min_score=l["min_score"])
                lk.save_cache()
            r, a, acc = max(band, key=lambda x: (x[2], -x[1]))      # ties: prefer more LLM mentions
            out.update(llm_reject=r, llm_accept=a, llm_dev_accuracy=acc, band=band, llm_calls=lk.llm_calls)
        path = os.path.join(self.p["out_dir"], "tuned_linking.json")
        save_json(out, path)
        return {**out, "path": path, "rows": len(rows)}

    # ------------------------------------------------------------------ team labeling
    def _records_and_texts(self):
        recs, texts = [], {}
        for kind, key in (("posting", "extracted_postings"), ("resume", "extracted_resumes")):
            if not os.path.exists(self.p[key]):
                raise InputError(f"no extraction output at {self.p[key]}: run the extract step first")
            pool = self.corpus["postings" if kind == "posting" else "resumes"]
            for rec in load_records(self.p[key]):
                doc_id = norm_id(rec.get("doc_id"))
                if doc_id in pool and rec.get("status") == "ok":
                    recs.append({**rec, "doc_id": doc_id})
                    texts[doc_id] = self.text(kind, doc_id)
        return recs, texts

    def labeling_sheets(self, task):
        """Per-annotator sheets with a shared overlap (see labeling.py)."""
        L, seed = self.cfg["labeling"], self.cfg["evaluation"]["seed"]
        out_dir = resolve(L["sheets_dir"])
        if task == "extraction":
            return labeling.write_extraction_templates(
                self.p["postings"], self.p["resumes"], resolve(L["gold_dir"]), resolve(L["gold_second_dir"]),
                L["annotators"], L["extraction_docs"], L["extraction_shared"], seed)
        if task == "relevance":
            g = self.graph_or_build()
            cats = {rid: g.meta[rid].get("category") for rid in g.doc_ids("resume")}
            rids = labeling.choose_resumes(self.split(), cats, L["graded_resumes"], seed)
            rows = pool_rows(self.ranker, rids, depth=self.cfg["evaluation"]["pool_depth"])
            written, common = labeling.write_sheets(rows, "resume_id", L["annotators"], L["shared_resumes"],
                                                    out_dir, "relevance", seed)
            return {"resumes": len(rids), "rows": len(rows), "shared_resumes": common, "sheets": written}
        if task == "linking":
            recs, texts = self._records_and_texts()
            try:
                rows = linking_rows(recs, texts, self.linker, L["linking_mentions"], seed)
            finally:
                self.linker.save_cache()
            written, common = labeling.write_sheets(rows, "mention_id", L["annotators"], L["shared_mentions"],
                                                    out_dir, "linking", seed)
            return {"mentions": len(rows), "shared": len(common), "sheets": written}
        raise ValueError(task)

    def graph_or_build(self):
        if self.graph is None:
            self.build()
        return self.graph

    def merge_labels(self, task):
        L = self.cfg["labeling"]
        d = resolve(L["sheets_dir"])
        if task == "relevance":
            return labeling.merge(d, "relevance", ("resume_id", "job_id"), "grade", self.p["graded_qrels"],
                                  L["annotators"], ordinal=True)
        if task == "linking":
            return labeling.merge(d, "linking", ("mention_id",), "gold_skill_id", self.p["linking_gold"],
                                  L["annotators"])
        raise ValueError(task)


def _f(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"
