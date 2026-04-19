"""Gradio web UI — multi-user, session-aware medical chatbot."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_core.messages import AIMessage, HumanMessage

import gradio as gr

from auth import AuthManager
from chatbot import MedicalChatbot
from config import Config
from llm import ModelType
from rag import RAGPipeline
from session_store import SessionStore

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
# Single shared chatbot instance — history passed explicitly per user via gr.State
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
    """Convert LangChain message list → Gradio chatbot format."""
    result = []
    for msg in lc_history:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result


# ── Gradio event handlers ─────────────────────────────────────────────────────

def on_load(request: gr.Request):
    """Called when the page loads — sets up user identity only. Session created on first message."""
    user_id = _get_user_id(request)
    if not user_id:
        return None, None, [], gr.Dropdown(choices=[]), ""
    return user_id, None, [], _sessions_dropdown(user_id), _user_panel_html(user_id)


def respond(message: str, gradio_history: list, lc_history: list,
            session_id: str, user_id: str):
    """Handle a user message — persist to DB and return updated UI state."""
    if not message.strip() or not user_id:
        return "", gradio_history, lc_history, session_id, _sessions_dropdown(user_id)

    # Lazy session creation — only on the first message
    if session_id is None:
        session_id = store.create_session(user_id)

    response, lc_history = chatbot.chat_with_history(message, lc_history)

    store.save_message(session_id, "user", message)
    store.save_message(session_id, "assistant", response)

    # Title the session from the first user message
    if len(lc_history) == 2:
        store.set_title(session_id, message)

    gradio_history.append({"role": "user", "content": message})
    gradio_history.append({"role": "assistant", "content": response})
    return "", gradio_history, lc_history, session_id, _sessions_dropdown(user_id)


def new_chat(user_id: str):
    """Clear chat state — session is created lazily on the first message."""
    return [], [], None, _sessions_dropdown(user_id) if user_id else gr.Dropdown(choices=[])


def load_session(session_id: str, user_id: str):
    """Load a past session into the chat window."""
    if not session_id or not user_id:
        return [], []
    messages = store.load_messages(session_id)
    lc_history = []
    for m in messages:
        if m["role"] == "user":
            lc_history.append(HumanMessage(content=m["content"]))
        else:
            lc_history.append(AIMessage(content=m["content"]))
    gradio_history = _history_to_gradio(lc_history)
    return gradio_history, lc_history


def clear_chat(user_id: str):
    """Clear the current view and start a new session."""
    return new_chat(user_id)


# ── UI layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Medical Chatbot") as demo:

    # Per-user state (isolated per browser tab)
    user_id_state = gr.State(None)
    session_id_state = gr.State(None)
    lc_history_state = gr.State([])  # list of LangChain HumanMessage/AIMessage

    with gr.Row():

        # ── Sidebar ───────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=220):
            user_panel = gr.HTML("")
            gr.Markdown("## 🏥 Sessions")
            with gr.Row():
                new_chat_btn = gr.Button("+ New Chat", variant="primary", scale=3)
                refresh_btn = gr.Button("↻", variant="secondary", scale=1, min_width=40)
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

    # ── Wire events ───────────────────────────────────────────────────────

    demo.load(
        on_load,
        outputs=[user_id_state, session_id_state, lc_history_state, sessions_dd, user_panel],
    )

    send_btn.click(
        respond,
        inputs=[msg, chatbox, lc_history_state, session_id_state, user_id_state],
        outputs=[msg, chatbox, lc_history_state, session_id_state, sessions_dd],
    )
    msg.submit(
        respond,
        inputs=[msg, chatbox, lc_history_state, session_id_state, user_id_state],
        outputs=[msg, chatbox, lc_history_state, session_id_state, sessions_dd],
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
