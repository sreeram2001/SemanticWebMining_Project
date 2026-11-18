"""Command line for the whole pipeline (web demo: CODE/app.py).

Offline run on the synthetic sample:  python CODE/run_pipeline.py --sample all

Real data (from the repo root):
    python CODE/run_pipeline.py preprocess             # Kaggle CSVs -> DATA/processed (Step 1)
    python CODE/run_pipeline.py split                  # freeze the dev/test split
    export ASU_VOYAGER_KEY=...
    python CODE/run_pipeline.py extract                # grounded LLM extraction, resumable
    python CODE/run_pipeline.py build --cypher EVALUATIONS/runs/graph.cypher [--neo4j]
    python CODE/run_pipeline.py rank --resume 16852973
    python CODE/run_pipeline.py sheets                 # labeling sheets for all six annotators
    python CODE/run_pipeline.py merge-labels           # after labeling: gold files + agreement
    python CODE/run_pipeline.py tune-linking           # linking threshold, dev half only
    python CODE/run_pipeline.py tune                   # dev split only
    python CODE/run_pipeline.py evaluate --split dev --tuned
    python CODE/run_pipeline.py evaluate --split test --tuned   # Phase 5, once per configuration
    python CODE/run_pipeline.py eval-extraction        # extraction P/R/F1 on the gold set
    python CODE/run_pipeline.py chat                   # terminal chat

Flags: --rules (no LLM), --require-models (no stand-ins), --debug (tracebacks).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import ConfigError, load_config, read_json_or, resolve  # noqa: E402
from core.pipeline import InputError, Pipeline  # noqa: E402
from rank.ranking import METHODS  # noqa: E402

SAMPLE_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "config_sample.json")


def _tuned(p):
    path = os.path.join(p.p["out_dir"], "tuned_scoring.json")
    tuned = read_json_or(path, None, "tuned scoring")
    if not tuned or not tuned.get("scoring"):
        raise InputError(f"no tuned weights at {path}: run the tune step first")
    return tuned["scoring"]


def _print_recommendation(rec):
    print(f"\nTop jobs for resume {rec['resume_id']} ({rec['method']}):")
    for j in rec["jobs"]:
        extra = f"  G={j['G']:.2f} P={j['P']:.2f} CE={j['CE']:.2f}" if "G" in j else ""
        print(f"  {j['rank']:2d}. {j['job_id']}  {j['title']} @ {j['company']}  S={j['score']:.3f}{extra}")
    if "explanation_top1" in rec:
        e = rec["explanation_top1"]
        print(f"\nWhy #1 ({'verified' if e['verified'] else 'NOT verified: ' + ', '.join(e['unsupported_skills'])}):")
        print("  " + e["text"].replace("\n", "\n  "))
    if rec.get("skill_gaps"):
        print("\nSkill gaps across the top jobs:")
        for g in rec["skill_gaps"]:
            hint = f" (closest you have: {g['closest']})" if g["closest"] else ""
            print(f"  - {g['skill']}: required in {g['required_in']}, asked in {len(g['jobs'])}{hint}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["preprocess", "split", "extract", "build", "rank", "sheets", "merge-labels",
                                     "tune", "tune-linking", "evaluate", "eval-extraction", "chat", "all"])
    ap.add_argument("--task", default="all", choices=["all", "extraction", "linking", "relevance"],
                    help="which labeling sheets to write or merge")
    ap.add_argument("--config", help="JSON file merged over CODE/config.json")
    ap.add_argument("--sample", action="store_true", help="use the offline synthetic sample")
    ap.add_argument("--rules", action="store_true", help="offline rule-based extraction instead of the LLM")
    ap.add_argument("--no-models", action="store_true", help="skip sentence-transformers models")
    ap.add_argument("--require-models", action="store_true",
                    help="fail instead of falling back to stand-ins (use for every reported run)")
    ap.add_argument("--limit", type=int, help="extract at most N new documents per kind")
    ap.add_argument("--resume", help="resume ID for the rank step (default: first dev resume)")
    ap.add_argument("--method", default="full", choices=METHODS)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--tuned", action="store_true", help="use EVALUATIONS/runs/tuned_scoring.json")
    ap.add_argument("--tier", default="auto", choices=["auto", "graded"], help="labels used for tuning")
    ap.add_argument("--cypher", help="write the graph as a Cypher script")
    ap.add_argument("--neo4j", action="store_true", help="load the graph into Neo4j (config: neo4j)")
    ap.add_argument("--models", help="comma-separated LLMs to compare in eval-extraction")
    ap.add_argument("--debug", action="store_true", help="show full tracebacks")
    args = ap.parse_args(argv)
    try:
        run(args)
    except (ConfigError, InputError) as exc:
        if args.debug:
            raise
        sys.exit(f"error: {exc}")
    except KeyboardInterrupt:
        sys.exit("interrupted (extraction resumes where it stopped on the next run)")


def run(args):
    overrides = {}
    if args.rules:
        overrides["extraction"] = {"mode": "rules"}
    if args.require_models:
        overrides["strict_models"] = True
    cfg = load_config(args.config or (SAMPLE_CONFIG if args.sample else None), overrides or None)
    p = Pipeline(cfg, use_models=not args.no_models)
    steps = (["split", "extract", "build", "rank", "tune", "evaluate", "eval-extraction"] if args.step == "all"
             else [args.step])

    for step in steps:
        print(f"\n=== {step} ===")
        if step == "preprocess":
            from dataprep.preprocess import preprocess
            print(json.dumps(preprocess(resolve(cfg["paths"]["raw_postings"]), resolve(cfg["paths"]["raw_resumes"]),
                                        p.p["postings"], p.p["resumes"]), indent=1))
        elif step == "split":
            s = p.split()
            print(f"{sum(v == 'dev' for v in s.values())} dev / {sum(v == 'test' for v in s.values())} test resumes "
                  f"(frozen in {p.p['split']})")
        elif step == "extract":
            out = p.extract(args.limit)
            for kind, c in out.items():
                fields = {f: v["kept"] for f, v in c["summary"]["fields"].items()}
                print(f"{kind}: ok {c['ok']}, errors {c['error']}, already done {c['skipped']}; facts kept {fields}")
        elif step == "build":
            info = p.build(neo4j=args.neo4j, cypher_path=args.cypher)
            print(json.dumps(info, indent=1))
        elif step == "rank":
            p.graph_or_build()
            dev = p.split_ids("dev")
            rid = args.resume or (dev[0] if dev else None)
            if rid is None:
                raise InputError("no dev resume in the graph; pass --resume ID")
            _print_recommendation(p.recommend(rid, args.method, scoring=_tuned(p) if args.tuned else None))
        elif step == "sheets":
            for task in (["extraction", "linking", "relevance"] if args.task == "all" else [args.task]):
                print(task, json.dumps(p.labeling_sheets(task), indent=1, default=str))
        elif step == "merge-labels":
            for task in (["linking", "relevance"] if args.task == "all" else [args.task]):
                if task == "extraction":
                    print("extraction labels are read straight from EVALUATIONS/gold by evaluation.load_gold")
                    continue
                print(task, json.dumps(p.merge_labels(task), indent=1))
        elif step == "tune-linking":
            t = p.tune_linking()
            print(f"{t['rows']} dev mentions; accuracy by min_score: "
                  + ", ".join(f"{s:.2f}->{a:.3f}" for s, a in t["curve"])
                  + f"\nbest min_score = {t['min_score']} (put it in config linking.min_score); saved {t['path']}")
            if "llm_accept" in t:
                print(f"LLM band: llm_reject = {t['llm_reject']}, llm_accept = {t['llm_accept']} "
                      f"(dev accuracy {t['llm_dev_accuracy']:.3f}); put them in config linking")
        elif step == "tune":
            t = p.tune(tier=args.tier)
            print(f"best dev nDCG@{cfg['evaluation']['k']} = {t['dev_nDCG']:.3f} over {t['tried']} configs: "
                  f"{t['best']}\nsaved {t['path']}")
        elif step == "evaluate":
            tuned = _tuned(p) if (args.tuned or args.step == "all") else None
            res = p.evaluate(args.split, scoring=tuned,
                             tag=f"{args.split}{'_tuned' if tuned else ''}")
            print(res["markdown"])
            print("saved", *res["paths"])
        elif step == "eval-extraction":
            res = p.evaluate_extraction([m.strip() for m in args.models.split(",")] if args.models else None)
            print(res["markdown"])
            print("saved", *res["paths"])
        elif step == "chat":
            _chat(p, args.resume, _tuned(p) if args.tuned else None)


def _chat(p, resume_id, scoring):
    """Terminal chat; the web version is app.py."""
    from assistant.chat import HELP, CareerAgent
    agent = CareerAgent(p, scoring=scoring)
    print(HELP)
    if resume_id:
        print(agent.use_resume(resume_id))
    else:
        print("Sample resumes:", ", ".join(r for _, r in agent.sample_resumes(15)))
        print("Type `load <ID>`, `load file <path>`, or paste a resume.")
    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if msg.lower() in ("quit", "exit", "q"):
            return
        try:
            if msg.lower().startswith("load file "):
                with open(msg[10:].strip(), encoding="utf-8", errors="ignore") as f:
                    print(agent.upload(f.read()))
            elif msg.lower().startswith("load "):
                print(agent.use_resume(msg[5:].strip()))
            else:
                print(agent.reply(msg))
        except (KeyError, IndexError, OSError, InputError) as exc:
            print(f"error: {exc}")


if __name__ == "__main__":
    main()
