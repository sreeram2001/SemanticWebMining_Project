import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skill_linking as sl

ALIASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skill_aliases.json")
def build_linker(skills_path, embed=False, llm=False, cache="DATA/lightcast/skill_embeddings.npz"):
    skills = sl.load_skills(skills_path)
    retriever = sl.EmbeddingRetriever(skills, cache_path=cache) if embed else None
    decider = None
    if llm:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["ASU_VOYAGER_KEY"],
                        base_url=os.environ.get("VOYAGER_BASE_URL", "https://openai.rc.asu.edu/v1"))
        decider = sl.LLMDecider(client)
    return sl.Linker(skills, sl.load_aliases(ALIASES), retriever=retriever, decider=decider)


def main(argv=None):
    p = argparse.ArgumentParser(description="Link extracted skills to Lightcast for G and P.")
    p.add_argument("--extracted", default="DATA/extracted")
    p.add_argument("--out", default="DATA/linked")
    p.add_argument("--skills", default=sl.DEFAULT_CACHE)
    p.add_argument("--embed", action="store_true", help="add embedding candidates (SBERT)")
    p.add_argument("--llm", action="store_true", help="let the Voyager LLM decide unclear links")
    a = p.parse_args(argv)
    if not os.path.exists(a.skills):
        raise SystemExit(f"error: {a.skills} not found; get the Lightcast skills.json from Kaushal")
    linker = build_linker(a.skills, a.embed, a.llm)
    os.makedirs(a.out, exist_ok=True)
    for name in ("resumes", "postings"):
        src = os.path.join(a.extracted, f"{name}.jsonl")
        if not os.path.exists(src):
            print(f"skip {src}: not found")
            continue
        counts = sl.run_linking(linker, src, os.path.join(a.out, f"{name}.jsonl"))
        print(name, counts)


if __name__ == "__main__":
    main()
