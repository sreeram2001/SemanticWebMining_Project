"""Web demo (Gradio): chat with the LinkedIn Assistant.

    python CODE/app.py --sample                  # offline synthetic data, no keys
    python CODE/app.py                           # real data (after extract + build)
    python CODE/app.py --share                   # public link (e.g. from Colab)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assistant.chat import HELP, CareerAgent  # noqa: E402
from core.config import ConfigError, load_config  # noqa: E402
from core.pipeline import InputError, Pipeline  # noqa: E402
from rank.ranking import METHODS  # noqa: E402

SAMPLE_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "config_sample.json")


def read_upload(path):
    """Text of an uploaded .txt/.md/.pdf file."""
    if not path:
        return ""
    if path.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("PDF upload needs `pip install pypdf`; paste the text instead")
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def build_app(agent):
    import gradio as gr

    samples = agent.sample_resumes()
    backends = agent.ranker.backends

    def _agent(state):
        return agent.session(state.get("resume_id"), state.get("method", "full"))

    def _say(history, user, bot):
        return history + [{"role": "user", "content": user}, {"role": "assistant", "content": bot}]

    def load(sample, upload, pasted, history, state):
        a = _agent(state)
        try:
            if pasted and pasted.strip():
                text, label = a.upload(pasted), "pasted resume"
            elif upload:
                text, label = a.upload(read_upload(upload)), "uploaded file"
            elif sample:
                text, label = a.use_resume(sample), f"sample resume {sample}"
            else:
                return history, state, ""
        except Exception as exc:
            return _say(history, "Load resume", f"Could not load it: {exc}"), state, pasted
        return _say(history, f"Load {label}", text), {**state, "resume_id": a.resume_id}, ""

    def respond(message, history, state):
        if not message or not message.strip():
            return history, state, ""
        a = _agent(state)
        try:
            answer = a.reply(message)
        except Exception as exc:
            answer = f"Something went wrong: {type(exc).__name__}: {exc}"
        return _say(history, message, answer), {**state, "resume_id": a.resume_id, "method": a.method}, ""

    def set_method(method, history, state):
        state = {**state, "method": method}
        if not state.get("resume_id"):
            return history, state
        return _say(history, f"method {method}", _agent(state).recommend()), state

    import inspect
    # Gradio 4.44/5.x default to tuple history; 6.x accepts only messages and dropped `type`.
    chat_kwargs = {"type": "messages"} if "type" in inspect.signature(gr.Chatbot.__init__).parameters else {}

    with gr.Blocks(title="LinkedIn Assistant") as demo:
        gr.Markdown("# LinkedIn Assistant\nKnowledge-graph job matching with grounded explanations "
                    "(CSE 573, Group 5).")
        state = gr.State({"resume_id": None, "method": "full"})
        with gr.Row():
            with gr.Column(scale=1):
                sample = gr.Dropdown([(lbl, rid) for lbl, rid in samples], label="Sample resume", value=None)
                upload = gr.File(label="Or upload a resume (.txt, .pdf)", type="filepath",
                                 file_types=[".txt", ".md", ".pdf"])
                pasted = gr.Textbox(label="Or paste resume text", lines=6)
                load_btn = gr.Button("Load resume", variant="primary")
                method = gr.Radio(METHODS, value="full", label="Ranking method")
                gr.Markdown(f"<small>Backends: dense = {backends['dense']}; reranker = {backends['reranker']}; "
                            f"linker = {backends['linker']}</small>")
            with gr.Column(scale=2):
                chat = gr.Chatbot(value=[{"role": "assistant", "content": HELP}], height=560, label="Assistant",
                                  **chat_kwargs)
                msg = gr.Textbox(placeholder="e.g. jobs, why 1, gaps, compare", label="Message", show_label=False)
                with gr.Row():
                    quick = {t: gr.Button(t, size="sm") for t in ("jobs", "why 1", "gaps", "skills", "compare")}

        load_btn.click(load, [sample, upload, pasted, chat, state], [chat, state, pasted])
        msg.submit(respond, [msg, chat, state], [chat, state, msg])
        method.change(set_method, [method, chat, state], [chat, state])
        for text, b in quick.items():               # bind the command text; don't read it back from the Button
            b.click(lambda history, st, text=text: respond(text, history, st), [chat, state], [chat, state, msg])
    return demo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON file merged over CODE/config.json")
    ap.add_argument("--sample", action="store_true", help="use the offline synthetic sample")
    ap.add_argument("--no-models", action="store_true", help="skip sentence-transformers models")
    ap.add_argument("--no-llm", action="store_true", help="never call the LLM, even with a key set")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    ap.add_argument("--debug", action="store_true", help="show full tracebacks")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config or (SAMPLE_CONFIG if args.sample else None))
        print("building the knowledge graph ...")
        agent = CareerAgent(Pipeline(cfg, use_models=not args.no_models), use_llm=not args.no_llm)
    except (ConfigError, InputError) as exc:
        if args.debug:
            raise
        hint = "" if args.sample or args.config else \
            "\n(no real data yet? try the offline demo: python CODE/app.py --sample)"
        sys.exit(f"error: {exc}{hint}")
    print("backends:", agent.ranker.backends)
    build_app(agent).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
