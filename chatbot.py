"""Chatbot with RAG capabilities, GraphRAG, web search, and preference collection."""

import io
import contextlib
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun

from llm import LLMManager, ModelType
from rag import RAGPipeline
from graph_retrieval import GraphRAGPipeline
from preference_collector import PreferenceCollector


SYSTEM_PROMPT = """You are an AI medical researcher assistant. Answer the user's medical questions accurately and compassionately, just as a doctor would.

You have three tools available:
- **rag_retrieval**: searches a local database of doctor-patient conversations using semantic similarity. Best for factual questions like "what is X", "how is X treated", "what are symptoms of X".
- **graph_retrieval**: searches a medical knowledge graph of concepts and their relationships. Best for relational questions like "what conditions relate to X", "what is associated with X", "what co-occurs with X", or when the user mentions multiple medical concepts and wants to understand how they connect.
- **web_search**: searches the web for up-to-date medical information. Use only when both local tools return insufficient results.

Decision guide:
1. If the query asks about a specific condition, symptom, or treatment → use rag_retrieval first.
2. If the query asks about relationships, associations, or connections between concepts → use graph_retrieval first.
3. If neither returns useful results → use web_search.

Cite when information comes from a web search. Never make up medical facts."""


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

        tools = self._build_tools()
        self.agent = create_agent(self.llm, tools, system_prompt=SYSTEM_PROMPT, debug=False)

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
            docs = rag.retrieve(query)
            if not docs:
                print("[RAG RETRIEVAL] No results found.")
                return "No relevant conversations found in the local database."
            print(f"[RAG RETRIEVAL] Found {len(docs)} document(s).")
            parts = [f"[Conversation {i+1}]\n{d.page_content}" for i, d in enumerate(docs)]
            return "\n\n".join(parts)

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
        # The last message in the output is the assistant reply
        return result["messages"][-1].content

    def chat(self, user_input: str) -> str:
        """Process user input and return response."""
        response = self._invoke(user_input)
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response))
        return response

    def chat_with_history(self, user_input: str, history: list) -> tuple[str, list]:
        """Stateless chat — takes and returns history explicitly (multi-user safe)."""
        messages = history + [HumanMessage(content=user_input)]
        result = self.agent.invoke({"messages": messages})
        response = result["messages"][-1].content
        history = history + [
            HumanMessage(content=user_input),
            AIMessage(content=response),
        ]
        return response, history

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
