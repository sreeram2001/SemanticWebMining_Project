"""Web UI (Gradio) for the LinkedIn Assistant chat.

Run from the repository root:

    python CODE/app.py --sample          # offline synthetic sample in CODE/sample_data
    python CODE/app.py                   # real data in DATA/ (see CODE/README.md)
    python CODE/app.py --share           # public link, e.g. from Colab

Set ASU_VOYAGER_KEY to extract uploaded resumes and unextracted postings on demand.
"""

import argparse
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rerank as rr  # noqa: E402
from assistant import DENSE_MODEL, HELP, METHODS, Assistant, Chat, SetupError  # noqa: E402


def read_upload(path):
    """Text of an uploaded .txt, .md or .pdf file."""
    if not path:
        return ""
    if path.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("reading a PDF needs `pip install pypdf`; paste the text instead")
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def build_app(assistant):
    import gradio as gr

    methods = [(METHODS[m], m) for m in assistant.methods]
    default = Chat.default_method(assistant)

    def chat_for(state):
        return Chat(assistant, state.get("resume_id"), state.get("method"))

    def saved(c):
        return {"resume_id": c.resume_id, "method": c.method}

    def say(history, user, bot):
        return history + [{"role": "user", "content": user}, {"role": "assistant", "content": bot}]

    def load(sample, upload, pasted, history, state):
        c = chat_for(state)
        try:
            if pasted and pasted.strip():
                label, answer = "Load pasted resume", c.load_text(pasted)
            elif upload:
                label, answer = f"Load {os.path.basename(upload)}", c.load_text(read_upload(upload))
            elif sample:
                label, answer = f"Load sample resume {sample}", c.use_resume(sample)
            else:
                return history, state, pasted, None
        except Exception as exc:
            return say(history, "Load resume", f"Could not load it: {exc}"), state, pasted, upload
        return say(history, label, answer), saved(c), "", None

    def respond(message, history, state):
        if not message or not message.strip():
            return history, state, "", state.get("method")
        c = chat_for(state)
        try:
            answer = c.reply(message)
        except Exception as exc:
            answer = f"Something went wrong: {type(exc).__name__}: {exc}"
        # the radio follows a method switched in the chat; set_method then sees no change
        return say(history, message, answer), saved(c), "", c.method

    def set_method(method, history, state):
        if method == state.get("method"):
            return history, state
        c = chat_for(state)
        answer = c.set_method(method)
        return say(history, f"use {method}", answer), saved(c)

    # Gradio 4.x/5.x need type="messages" for role/content history; 6.x has no `type` argument.
    chat_kwargs = {"type": "messages"} if "type" in inspect.signature(gr.Chatbot.__init__).parameters else {}

    with gr.Blocks(title="LinkedIn Assistant") as demo:
        gr.Markdown("# LinkedIn Assistant\nResume to job matching with skill evidence and gaps ")
        state = gr.State({"resume_id": None, "method": default})
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                sample = gr.Dropdown(assistant.sample_resumes(), label="Sample resume", value=None)
                upload = gr.File(label="Or upload a resume (.txt, .md, .pdf)", type="filepath",
                                 file_types=[".txt", ".md", ".pdf"])
                pasted = gr.Textbox(label="Or paste resume text", lines=6)
                load_btn = gr.Button("Load resume", variant="primary")
                method = gr.Radio(methods, value=default, label="Ranking method")
                gr.Markdown(f"<small>{assistant.status()}</small>")
            with gr.Column(scale=2):
                chat = gr.Chatbot(value=[{"role": "assistant", "content": HELP}], height=600,
                                  label="Assistant", **chat_kwargs)
                msg = gr.Textbox(placeholder="e.g. show my top jobs, explain #3, what skills am I missing",
                                 show_label=False)
                with gr.Row():
                    quick = {t: gr.Button(t, size="sm") for t in (
                        "show my top jobs", "explain #1", "what skills am I missing", "compare these",
                        "compare methods", "job 1", "my skills")}

        load_btn.click(load, [sample, upload, pasted, chat, state], [chat, state, pasted, upload])
        msg.submit(respond, [msg, chat, state], [chat, state, msg, method])
        method.change(set_method, [method, chat, state], [chat, state])
        for text, button in quick.items():  # bind the text here; a Button's value is its label in some versions
            button.click(lambda history, st, text=text: respond(text, history, st), [chat, state],
                         [chat, state, msg, method])
    return demo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="use the synthetic sample in CODE/sample_data")
    ap.add_argument("--processed", help="folder with postings_cleaned.csv and resumes_cleaned.csv")
    ap.add_argument("--extracted", help="folder with resumes.jsonl and postings.jsonl")
    ap.add_argument("--skills", help="Lightcast skills cache (skills.json)")
    ap.add_argument("--embeddings", help="posting embedding cache (.npz), shared with run_baselines.py")
    ap.add_argument("--limit-postings", type=int, help="rank only the first N postings (faster start)")
    ap.add_argument("--dense-model", default=DENSE_MODEL)
    ap.add_argument("--rerank-model", default=rr.DEFAULT_MODEL)
    ap.add_argument("--rerank-top", type=int, default=rr.DEFAULT_TOP_N)
    ap.add_argument("--no-models", action="store_true", help="BM25 only: no sentence-transformers models")
    ap.add_argument("--no-llm", action="store_true", help="no on-demand extraction, even with ASU_VOYAGER_KEY set")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = ap.parse_args(argv)

    paths = {k: getattr(args, k) for k in ("processed", "extracted", "skills", "embeddings") if getattr(args, k)}
    print("loading postings and building the indexes ...")
    try:
        assistant = Assistant(sample=args.sample, paths=paths, limit_postings=args.limit_postings,
                              dense_model=None if args.no_models else args.dense_model,
                              rerank_model=None if args.no_models else args.rerank_model,
                              rerank_top=args.rerank_top, use_llm=not args.no_llm)
    except SetupError as exc:
        hint = "" if args.sample else "\n\nNo data yet? Try the offline sample: python CODE/app.py --sample"
        sys.exit(f"error: {exc}{hint}")
    print(assistant.status())
    print("ranking methods:", ", ".join(assistant.methods))
    build_app(assistant).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
