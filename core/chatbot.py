"""Chatbot with RAG capabilities, GraphRAG, web search, and preference collection."""

import asyncio
import io
import contextlib
from queue import Queue
from threading import Thread
from typing import List, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun

from rag.cache import cache

from core.llm import LLMManager, ModelType
from core.guardrails import GuardrailsManager
from rag.pipeline import RAGPipeline
from graph.retrieval import GraphRAGPipeline
from training.preference_collector import PreferenceCollector


# ── A/B variant identifiers ───────────────────────────────────────────────────
VARIANT_A = "empathetic"   # warm, conversational, higher temperature
VARIANT_B = "clinical"     # structured, concise, lower temperature

_TOOL_BLOCK = """
You have three tools available:
- **rag_retrieval**: searches a local database of doctor-patient conversations using semantic similarity. Best for factual questions like "what is X", "how is X treated", "what are symptoms of X".
- **graph_retrieval**: searches a medical knowledge graph of concepts and their relationships. Best for relational questions like "what conditions relate to X", "what is associated with X", "what co-occurs with X".
- **web_search**: searches the web for up-to-date medical information. Use only when both local tools return insufficient results.

Decision guide:
1. If the query asks about a specific condition, symptom, or treatment → use rag_retrieval first.
2. If the query asks about relationships, associations, or connections between concepts → use graph_retrieval first.
3. If neither returns useful results → use web_search.

IMPORTANT — query consistency rule:
Pass the user's question verbatim to tools. Do NOT rephrase or summarise. Verbatim input improves cache hit rate.

Cite when information comes from a web search. Never make up medical facts."""

# Variant A — empathetic, conversational tone (temperature 0.7)
SYSTEM_PROMPT_A = (
    "You are an AI medical researcher assistant. Answer the user's medical questions "
    "accurately and compassionately, just as a caring doctor would speak to a worried patient.\n"
    + _TOOL_BLOCK +
    "\nStyle: Be warm and empathetic. Use plain language a patient can understand. "
    "Explain the 'why' behind medical facts. Reassure where appropriate. Write in flowing prose."
)

# Variant B — clinical, structured tone (temperature 0.3)
SYSTEM_PROMPT_B = (
    "You are an AI medical researcher assistant. Answer the user's medical questions "
    "with clinical precision and efficiency.\n"
    + _TOOL_BLOCK +
    "\nStyle: Be concise and structured. Lead with the direct answer. Use bullet points "
    "and clear sections where helpful. Omit reassurances — state facts only. Avoid filler phrases."
)

# Default prompt used by the main chatbot (normal / streaming path)
SYSTEM_PROMPT = SYSTEM_PROMPT_A


class MedicalChatbot:
    """Medical chatbot with RAG, GraphRAG, and web search."""

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        graph_pipeline: Optional[GraphRAGPipeline] = None,
        collect_preferences: bool = False,
        model_type: ModelType = ModelType.OPENAI,
    ):
        self.model_type = model_type
        LLMManager.set_model_type(model_type)
        self.llm = LLMManager.get_llm()
        self.rag = rag_pipeline
        self.graph = graph_pipeline or GraphRAGPipeline()
        self.chat_history: List = []
        self.collect_preferences = collect_preferences
        self.preference_collector = PreferenceCollector() if collect_preferences else None
        self.guardrails = GuardrailsManager.get_instance()

        tools = self._build_tools()
        self.agent = create_react_agent(self.llm, tools, prompt=SYSTEM_PROMPT)

    def _build_tools(self):
        rag = self.rag
        graph = self.graph

        @tool
        def rag_retrieval(query: str) -> str:
            """Search the local medical conversation database for relevant context."""
            print(f"\n[RAG RETRIEVAL] Query: {query}")
            if rag is None or rag.vector_store is None:
                print("[RAG RETRIEVAL] No vector store available.")
                return "No local knowledge base available."

            # Check context cache — returns fully processed context if hit
            cached_context = cache.get_context(query)
            if cached_context is not None:
                print(f"[Cache] Context cache HIT — skipping retrieval for: '{query[:50]}'")
                return cached_context

            # Cache miss — retrieve, format, and store processed context
            docs = rag.retrieve(query)
            if not docs:
                print("[RAG RETRIEVAL] No results found.")
                return "No relevant conversations found in the local database."

            print(f"[RAG RETRIEVAL] Found {len(docs)} document(s). Caching context.")
            parts = [f"[Conversation {i+1}]\n{d.page_content}" for i, d in enumerate(docs)]
            context = "\n\n".join(parts)

            cache.set_context(query, context)
            return context

        @tool
        def graph_retrieval(query: str) -> str:
            """Search the medical knowledge graph for concept relationships and associations."""
            print(f"\n[GRAPH RETRIEVAL] Query: {query}")
            result = graph.retrieve(query)
            print(f"[GRAPH RETRIEVAL] Result:\n{result}\n")
            return result

        @tool
        def web_search(query: str) -> str:
            """Search the web for up-to-date medical information."""
            print(f"\n[WEB SEARCH] Query: {query}")
            with contextlib.redirect_stderr(io.StringIO()):
                result = DuckDuckGoSearchRun().run(query)
            print(f"[WEB SEARCH] Results:\n{result}\n")
            return result

        return [rag_retrieval, graph_retrieval, web_search]

    def _invoke(self, user_input: str) -> str:
        """Run the agent and return the final text response."""
        messages = self.chat_history + [HumanMessage(content=user_input)]
        result = self.agent.invoke({"messages": messages})
        return result["messages"][-1].content

    def chat(self, user_input: str) -> str:
        """Process user input and return response."""
        is_blocked, block_msg = self.guardrails.check_input_sync(user_input)
        if is_blocked:
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=block_msg))
            return block_msg
        response = self._invoke(user_input)
        is_out_blocked, out_msg = self.guardrails.check_output_sync(response)
        if is_out_blocked:
            response = out_msg
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response))
        return response

    def chat_with_history(self, user_input: str, history: list) -> tuple[str, list]:
        """Stateless chat — takes and returns history explicitly (multi-user safe)."""
        is_blocked, block_msg = self.guardrails.check_input_sync(user_input)
        if is_blocked:
            return block_msg, history + [
                HumanMessage(content=user_input),
                AIMessage(content=block_msg),
            ]
        messages = history + [HumanMessage(content=user_input)]
        result = self.agent.invoke({"messages": messages})
        response = result["messages"][-1].content
        is_out_blocked, out_msg = self.guardrails.check_output_sync(response)
        if is_out_blocked:
            response = out_msg
        history = history + [
            HumanMessage(content=user_input),
            AIMessage(content=response),
        ]
        return response, history

    def stream_with_history(self, user_input: str, history: list):
        """Generator that yields text deltas using astream_events — works with any LangChain runnable.

        Tokens are accumulated before yielding so the full response can be validated.
        This adds one moderation API round-trip of latency before streaming begins.
        """
        is_blocked, block_msg = self.guardrails.check_input_sync(user_input)
        if is_blocked:
            yield block_msg
            return

        messages = history + [HumanMessage(content=user_input)]
        queue: Queue = Queue()

        async def _astream():
            try:
                async for event in self.agent.astream_events(
                    {"messages": messages}, version="v2"
                ):
                    if event["event"] == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        content = getattr(chunk, "content", "")
                        if content and isinstance(content, str):
                            queue.put(content)
            except Exception:
                pass
            finally:
                queue.put(None)

        def _run():
            asyncio.run(_astream())

        thread = Thread(target=_run, daemon=True)
        thread.start()

        tokens = []
        while True:
            token = queue.get()
            if token is None:
                break
            tokens.append(token)

        thread.join()

        full_response = "".join(tokens)
        is_out_blocked, out_msg = self.guardrails.check_output_sync(full_response)

        if is_out_blocked:
            yield out_msg
        else:
            for token in tokens:
                yield token

    def get_two_responses(self, user_input: str, history: list) -> tuple[str, str, str, str]:
        """Generate two responses using different system prompts and temperatures.

        Returns (response_a, response_b, variant_a_name, variant_b_name).
          A = empathetic/conversational, temperature 0.7
          B = clinical/structured,       temperature 0.3
        """
        is_blocked, block_msg = self.guardrails.check_input_sync(user_input)
        if is_blocked:
            print(f"[GUARDRAILS] A/B path blocked — returning refusal for both variants")
            return block_msg, block_msg, VARIANT_A, VARIANT_B

        messages = history + [HumanMessage(content=user_input)]
        tools = self._build_tools()

        llm_a = LLMManager.get_llm(temperature=0.7)
        agent_a = create_react_agent(llm_a, tools, prompt=SYSTEM_PROMPT_A)
        response_a = agent_a.invoke({"messages": messages})["messages"][-1].content
        is_a_blocked, a_msg = self.guardrails.check_output_sync(response_a)
        if is_a_blocked:
            response_a = a_msg

        llm_b = LLMManager.get_llm(temperature=0.3)
        agent_b = create_react_agent(llm_b, tools, prompt=SYSTEM_PROMPT_B)
        response_b = agent_b.invoke({"messages": messages})["messages"][-1].content
        is_b_blocked, b_msg = self.guardrails.check_output_sync(response_b)
        if is_b_blocked:
            response_b = b_msg

        return response_a, response_b, VARIANT_A, VARIANT_B

    def generate_multiple_responses(self, user_input: str, num_responses: int = 2):
        """Generate multiple responses for preference comparison."""
        responses = [self._invoke(user_input) for _ in range(num_responses)]
        return responses, ""

    def chat_with_preference(self, user_input: str, num_options: int = 2) -> str:
        """Chat and collect preference data."""
        responses, context = self.generate_multiple_responses(user_input, num_options)

        print("\n" + "=" * 40)
        print("Choose the better response:")
        print("=" * 40)
        for i, resp in enumerate(responses):
            print(f"\n[{i}] {resp}")
        print("\n" + "=" * 40)

        while True:
            try:
                choice = input("Enter the number of the BETTER response (or 's' to skip): ").strip()
                if choice.lower() == 's':
                    chosen_response = responses[0]
                    break
                chosen_idx = int(choice)
                if 0 <= chosen_idx < len(responses):
                    rejected_idx = 1 - chosen_idx if num_options == 2 else int(
                        input("Enter rejected response number: ")
                    )
                    self.preference_collector.start_collection(user_input, context)
                    for resp in responses:
                        self.preference_collector.add_response(resp)
                    self.preference_collector.save_preference(chosen_idx, rejected_idx)
                    chosen_response = responses[chosen_idx]
                    break
                else:
                    print(f"Please enter a number between 0 and {len(responses)-1}")
            except ValueError:
                print("Invalid input. Enter a number or 's' to skip.")

        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=chosen_response))
        return chosen_response

    def clear_history(self):
        """Clear conversation history."""
        self.chat_history.clear()

    def get_history(self) -> List:
        """Return conversation history."""
        return self.chat_history
