"""Gradio web UI for the Medical Chatbot."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
from chatbot import MedicalChatbot
from rag import RAGPipeline
from config import Config
from llm import ModelType


def setup_rag() -> RAGPipeline:
    rag = RAGPipeline()
    if rag.load_vector_store():
        return rag
    try:
        rag.create_vector_store()
        rag.save_vector_store()
    except FileNotFoundError:
        return None
    return rag


# Initialise once at startup
Config.validate(require_openai=True)
rag = setup_rag()
chatbot = MedicalChatbot(rag_pipeline=rag, model_type=ModelType.OPENAI)


def respond(message: str, history: list):
    reply = chatbot.chat(message)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return "", history


def clear():
    chatbot.clear_history()
    return [], []


with gr.Blocks(title="Medical Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Medical Chatbot
        *Powered by GPT-4.1 + RAG + Web Search*
        """
    )

    chatbox = gr.Chatbot(
        label="Conversation",
        height=500,
        avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/2966/2966327.png"),
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Describe your symptoms or ask a medical question…",
            show_label=False,
            scale=9,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    clear_btn = gr.Button("Clear conversation", variant="secondary")

    # Submit on Enter or button click
    msg.submit(respond, [msg, chatbox], [msg, chatbox])
    send_btn.click(respond, [msg, chatbox], [msg, chatbox])
    clear_btn.click(clear, outputs=[chatbox, chatbox])

    gr.Markdown(
        "_This chatbot is for informational purposes only and is not a substitute for professional medical advice._"
    )

if __name__ == "__main__":
    demo.launch()
