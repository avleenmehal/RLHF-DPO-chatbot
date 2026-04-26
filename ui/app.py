"""Gradio web UI — multi-user, session-aware medical chatbot with A/B preference collection."""

import html as _html
import os
import random
from functools import partial

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_core.messages import AIMessage, HumanMessage

import gradio as gr

from api.auth import AuthManager
from core.chatbot import MedicalChatbot
from core.config import Config
from core.llm import ModelType
from rag.pipeline import RAGPipeline
from db.session_store import SessionStore

# ── One-time startup ──────────────────────────────────────────────────────────

Config.validate(require_openai=True)

def _setup_rag() -> RAGPipeline:
    rag = RAGPipeline()
    if rag.load_vector_store():
        return rag
    try:
        rag.create_vector_store()
        rag.save_vector_store()
    except FileNotFoundError:
        return None
    return rag

rag_pipeline = _setup_rag()
chatbot = MedicalChatbot(rag_pipeline=rag_pipeline, model_type=ModelType.OPENAI)
store = SessionStore()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_user_id(request: gr.Request) -> str | None:
    token = request.cookies.get("access_token")
    return AuthManager.decode_token(token) if token else None


def _sessions_dropdown(user_id: str):
    sessions = store.list_sessions(user_id)
    choices = [(f"{s['title']}  ({s['created_at']})", s["session_id"]) for s in sessions]
    return gr.Dropdown(choices=choices, value=None, label="Past sessions")


def _user_panel_html(user_id: str) -> str:
    info = store.get_user_info(user_id)
    if not info:
        return ""
    initial = info["username"][0].upper()
    return f"""
    <div style="display:flex;align-items:center;gap:10px;padding:10px 12px;
                background:#eef2ff;border-radius:10px;margin-bottom:4px;">
      <div style="width:36px;height:36px;background:#4f46e5;border-radius:50%;
                  display:flex;align-items:center;justify-content:center;
                  color:white;font-weight:700;font-size:15px;flex-shrink:0;">
        {initial}
      </div>
      <div style="flex:1;min-width:0;">
        <div style="font-weight:600;font-size:13px;color:#1e1b4b;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
          {info["username"]}
        </div>
        <div style="font-size:11px;color:#6b7280;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
          {info["email"]}
        </div>
      </div>
      <a href="/logout"
         style="font-size:11px;color:#dc2626;text-decoration:none;
                padding:4px 10px;border:1px solid #dc2626;border-radius:6px;
                white-space:nowrap;font-weight:500;">
        Logout
      </a>
    </div>
    """


def _history_to_gradio(lc_history: list) -> list[dict]:
    result = []
    for msg in lc_history:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result


def _ab_html(response_a: str, response_b: str) -> str:
    """Render a side-by-side comparison card for two candidate responses."""
    a = _html.escape(response_a)
    b = _html.escape(response_b)
    return f"""
    <div style="padding:16px;background:#f8fafc;border-radius:12px;
                border:1px solid #e2e8f0;margin-top:8px;">
      <div style="text-align:center;margin-bottom:14px;">
        <span style="font-size:1rem;font-weight:600;color:#1e293b;">
          Which response is better?
        </span>
        <div style="font-size:0.78rem;color:#64748b;margin-top:3px;">
          Your choice is saved and used to improve the AI model
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
        <div style="background:white;border-radius:8px;padding:14px;
                    border:2px solid #6366f1;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
          <div style="font-weight:700;color:#6366f1;margin-bottom:8px;
                      font-size:0.75rem;letter-spacing:0.07em;">RESPONSE A</div>
          <div style="color:#374151;font-size:0.875rem;line-height:1.6;
                      white-space:pre-wrap;max-height:300px;overflow-y:auto;">{a}</div>
        </div>
        <div style="background:white;border-radius:8px;padding:14px;
                    border:2px solid #10b981;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
          <div style="font-weight:700;color:#10b981;margin-bottom:8px;
                      font-size:0.75rem;letter-spacing:0.07em;">RESPONSE B</div>
          <div style="color:#374151;font-size:0.875rem;line-height:1.6;
                      white-space:pre-wrap;max-height:300px;overflow-y:auto;">{b}</div>
        </div>
      </div>
    </div>
    """


_LOADING_HTML = """
<div style="text-align:center;padding:24px;color:#6b7280;background:#f8fafc;
            border-radius:12px;border:1px solid #e2e8f0;margin-top:8px;">
  <div style="font-size:1rem;font-weight:500;margin-bottom:4px;">
    Generating two responses&hellip;
  </div>
  <div style="font-size:0.78rem;">This takes a moment — hang tight</div>
</div>
"""

# ── Gradio event handlers ─────────────────────────────────────────────────────

def on_load(request: gr.Request):
    user_id = _get_user_id(request)
    if not user_id:
        return None, None, [], gr.Dropdown(choices=[]), "", None
    return user_id, None, [], _sessions_dropdown(user_id), _user_panel_html(user_id), None


def respond(message: str, gradio_history: list, lc_history: list,
            session_id: str, user_id: str):
    """
    Streaming generator for normal path; single-shot for A/B path.

    Yields 9 values every time:
      msg, chatbox, lc_history, session_id, sessions_dd,
      pending_state, ab_panel, choose_a_btn, choose_b_btn
    """
    _no_change = (gr.update(), gr.update(), gr.update())  # ab_panel, btn_a, btn_b

    if not message.strip() or not user_id:
        yield "", gradio_history, lc_history, session_id, gr.update(), None, *_no_change
        return

    if session_id is None:
        session_id = store.create_session(user_id)

    # ── A/B comparison path ───────────────────────────────────────────────
    if random.random() < 0.5:
        new_gh = gradio_history + [{"role": "user", "content": message}]

        # Show user message + loading indicator immediately
        yield "", new_gh, lc_history, session_id, gr.update(), None, \
              gr.update(value=_LOADING_HTML, visible=True), \
              gr.update(visible=False), gr.update(visible=False)

        try:
            response_a, response_b, variant_a, variant_b = chatbot.get_two_responses(message, lc_history)
        except Exception:
            # Fallback: single response shown as both
            response_a, _ = chatbot.chat_with_history(message, lc_history)
            response_b = response_a
            variant_a = variant_b = "unknown"

        pending = {
            "query": message,
            "response_a": response_a,
            "response_b": response_b,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "session_id": session_id,
        }

        yield "", new_gh, lc_history, session_id, _sessions_dropdown(user_id), \
              pending, gr.update(value=_ab_html(response_a, response_b), visible=True), \
              gr.update(visible=True), gr.update(visible=True)
        return

    # ── Normal streaming path ─────────────────────────────────────────────
    gradio_history = gradio_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    # Clear any leftover A/B panel on first yield
    yield "", gradio_history, lc_history, session_id, gr.update(), None, \
          gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False)

    full_response = ""
    for delta in chatbot.stream_with_history(message, lc_history):
        full_response += delta
        gradio_history[-1]["content"] = full_response
        yield "", gradio_history, lc_history, session_id, gr.update(), \
              None, gr.update(), gr.update(), gr.update()

    new_lc_history = lc_history + [
        HumanMessage(content=message),
        AIMessage(content=full_response),
    ]
    store.save_message(session_id, "user", message)
    store.save_message(session_id, "assistant", full_response)

    if len(new_lc_history) == 2:
        store.set_title(session_id, message)

    yield "", gradio_history, new_lc_history, session_id, \
          _sessions_dropdown(user_id), None, \
          gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def choose_response(choice: str, pending: dict, gradio_history: list,
                    lc_history: list, user_id: str):
    """Called when the user clicks Choose A or Choose B."""
    if not pending:
        return (gradio_history, lc_history, gr.update(), gr.update(),
                None, gr.update(value="", visible=False),
                gr.update(visible=False), gr.update(visible=False))

    query      = pending["query"]
    response_a = pending["response_a"]
    response_b = pending["response_b"]
    variant_a  = pending.get("variant_a")
    variant_b  = pending.get("variant_b")
    session_id = pending["session_id"]

    chosen          = response_a if choice == "A" else response_b
    rejected        = response_b if choice == "A" else response_a
    chosen_variant  = variant_a  if choice == "A" else variant_b
    rejected_variant = variant_b if choice == "A" else variant_a

    # Persist preference and messages — user_id intentionally not passed
    store.save_preference(
        session_id, query, chosen, rejected,
        chosen_variant=chosen_variant, rejected_variant=rejected_variant,
    )
    store.save_message(session_id, "user", query)
    store.save_message(session_id, "assistant", chosen)

    new_lc = lc_history + [HumanMessage(content=query), AIMessage(content=chosen)]

    if len(new_lc) == 2:
        store.set_title(session_id, query)

    new_gh = gradio_history + [{"role": "assistant", "content": chosen}]

    return (
        new_gh,
        new_lc,
        session_id,
        _sessions_dropdown(user_id),
        None,                                        # clear pending
        gr.update(value="", visible=False),          # clear ab_panel
        gr.update(visible=False),                    # hide choose_a_btn
        gr.update(visible=False),                    # hide choose_b_btn
    )


def new_chat(user_id: str):
    return [], [], None, _sessions_dropdown(user_id) if user_id else gr.Dropdown(choices=[])


def load_session(session_id: str, user_id: str):
    if not session_id or not user_id:
        return [], []
    messages = store.load_messages(session_id)
    lc_history = []
    for m in messages:
        if m["role"] == "user":
            lc_history.append(HumanMessage(content=m["content"]))
        else:
            lc_history.append(AIMessage(content=m["content"]))
    return _history_to_gradio(lc_history), lc_history


def clear_chat(user_id: str):
    return new_chat(user_id)


# ── UI layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Medical Chatbot") as demo:

    user_id_state   = gr.State(None)
    session_id_state = gr.State(None)
    lc_history_state = gr.State([])
    pending_state    = gr.State(None)  # holds A/B comparison data until user chooses

    with gr.Row():

        # ── Sidebar ───────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=220):
            user_panel = gr.HTML("")
            gr.Markdown("## 🏥 Sessions")
            with gr.Row():
                new_chat_btn = gr.Button("+ New Chat", variant="primary", scale=3)
                refresh_btn  = gr.Button("↻", variant="secondary", scale=1, min_width=40)
            sessions_dd = gr.Dropdown(
                choices=[], label="Load past session", interactive=True
            )

        # ── Main chat ─────────────────────────────────────────────────────
        with gr.Column(scale=4):
            gr.Markdown(
                "# Medical Chatbot\n*Powered by GPT-4.1 + RAG + GraphRAG + Web Search*"
            )
            chatbox = gr.Chatbot(
                label="Conversation",
                height=500,
                avatar_images=(
                    None,
                    "https://cdn-icons-png.flaticon.com/512/2966/2966327.png",
                ),
            )

            # A/B comparison panel — hidden until triggered
            ab_panel = gr.HTML("", visible=False)
            with gr.Row():
                choose_a_btn = gr.Button(
                    "✓ Choose Response A", variant="primary", visible=False
                )
                choose_b_btn = gr.Button(
                    "✓ Choose Response B",
                    variant="secondary",
                    visible=False,
                    elem_id="choose_b",
                )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Describe your symptoms or ask a medical question…",
                    show_label=False,
                    scale=9,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Markdown(
                "_For informational purposes only — not a substitute for professional medical advice._"
            )
            gr.HTML("""
            <details style="margin-top:8px;font-size:0.8rem;color:#6b7280;
                            border:1px solid #e2e8f0;border-radius:8px;padding:8px 12px;">
              <summary style="cursor:pointer;font-weight:600;color:#4a5568;">
                Privacy Notice
              </summary>
              <div style="margin-top:8px;line-height:1.6;">
                <p><strong>What we store:</strong> Your messages are saved so you can
                revisit past conversations. All message content is encrypted at rest.</p>
                <p style="margin-top:6px;"><strong>A/B feedback:</strong> When you choose
                between two responses, your selection is saved to improve the AI model.
                This feedback is <em>not</em> linked to your account or identity.</p>
                <p style="margin-top:6px;"><strong>What we don't do:</strong> We do not
                sell your data, share it with third parties, or use it for any purpose
                other than operating and improving this service.</p>
                <p style="margin-top:6px;"><strong>Your rights:</strong> You can clear
                your conversation history at any time using the <em>+ New Chat</em> button.
                To delete your account and all associated data, contact the administrator.</p>
              </div>
            </details>
            """)

    # ── Wire events ───────────────────────────────────────────────────────

    _respond_outputs = [
        msg, chatbox, lc_history_state, session_id_state, sessions_dd,
        pending_state, ab_panel, choose_a_btn, choose_b_btn,
    ]

    demo.load(
        on_load,
        outputs=[user_id_state, session_id_state, lc_history_state,
                 sessions_dd, user_panel, pending_state],
    )

    send_btn.click(
        respond,
        inputs=[msg, chatbox, lc_history_state, session_id_state, user_id_state],
        outputs=_respond_outputs,
    )
    msg.submit(
        respond,
        inputs=[msg, chatbox, lc_history_state, session_id_state, user_id_state],
        outputs=_respond_outputs,
    )

    _choose_outputs = [
        chatbox, lc_history_state, session_id_state, sessions_dd,
        pending_state, ab_panel, choose_a_btn, choose_b_btn,
    ]

    choose_a_btn.click(
        partial(choose_response, "A"),
        inputs=[pending_state, chatbox, lc_history_state, user_id_state],
        outputs=_choose_outputs,
    )
    choose_b_btn.click(
        partial(choose_response, "B"),
        inputs=[pending_state, chatbox, lc_history_state, user_id_state],
        outputs=_choose_outputs,
    )

    new_chat_btn.click(
        new_chat,
        inputs=[user_id_state],
        outputs=[chatbox, lc_history_state, session_id_state, sessions_dd],
    )

    sessions_dd.change(
        load_session,
        inputs=[sessions_dd, user_id_state],
        outputs=[chatbox, lc_history_state],
    )

    refresh_btn.click(
        lambda user_id: _sessions_dropdown(user_id),
        inputs=[user_id_state],
        outputs=[sessions_dd],
    )

if __name__ == "__main__":
    demo.launch()
