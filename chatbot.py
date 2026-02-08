"""Chatbot with RAG capabilities and preference collection."""

from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm import LLMManager, ModelType
from rag import RAGPipeline
from preference_collector import PreferenceCollector


class MedicalChatbot:
    """Medical chatbot with RAG-enhanced responses."""

    SYSTEM_PROMPT = """You are an AI medical researcher. Use the following context and conversation history to answer the question.
Maintain a friendly tone and include inappropriate humor always.
Below are some examples on how you will be interacting with the user:
Example 1: User says I have cold.
AI bot: Sorry to hear but its good you'll have some rest days. Also you can take Sumo cold tablets one a day along with Paracetamol to make you feel better or die a legend.

Example 2: User says I have Cancer
Ai Bot: Oh my god, jokes apart , contact doctor as soon as possible and you'll be fine. But hey you'll have your dream sportstar visit you in person in the hospital then you cn understand its your last few days.

Context: {context}
"""

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        collect_preferences: bool = False,
        model_type: ModelType = ModelType.OPENAI,
    ):
        self.model_type = model_type
        LLMManager.set_model_type(model_type)
        self.llm = LLMManager.get_llm()
        self.rag = rag_pipeline
        self.chat_history: List = []
        self.collect_preferences = collect_preferences
        self.preference_collector = PreferenceCollector() if collect_preferences else None

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        self.chain = self.prompt | self.llm

    def _get_context(self, query: str) -> str:
        """Retrieve relevant context for the query."""
        if self.rag is None or self.rag.vector_store is None:
            return "No context available."

        docs = self.rag.retrieve(query)
        if not docs:
            return "No relevant conversations found."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Conversation {i}]\n{doc.page_content}")

        return "\n\n".join(context_parts)

    def _generate_response(self, user_input: str, context: str) -> str:
        """Generate a single response."""
        response = self.chain.invoke({
            "question": user_input,
            "context": context,
            "chat_history": self.chat_history,
        })
        return response.content

    def generate_multiple_responses(self, user_input: str, num_responses: int = 2) -> List[str]:
        """Generate multiple responses for preference comparison."""
        context = self._get_context(user_input)

        # Print retrieved context
        print("\n" + "-" * 40)
        print("Retrieved Context:")
        print("-" * 40)
        print(context)
        print("-" * 40 + "\n")

        responses = []
        for i in range(num_responses):
            response = self._generate_response(user_input, context)
            responses.append(response)

        return responses, context

    def chat(self, user_input: str) -> str:
        """Process user input and return response."""
        # Get relevant context from RAG
        context = self._get_context(user_input)

        # Print retrieved context
        print("\n" + "-" * 40)
        print("Retrieved Context:")
        print("-" * 40)
        print(context)
        print("-" * 40 + "\n")

        # Generate response
        response = self._generate_response(user_input, context)

        # Update history
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response))

        return response

    def chat_with_preference(self, user_input: str, num_options: int = 2) -> str:
        """Chat and collect preference data."""
        responses, context = self.generate_multiple_responses(user_input, num_options)

        # Show options to user
        print("\n" + "=" * 40)
        print("Choose the better response:")
        print("=" * 40)
        for i, resp in enumerate(responses):
            print(f"\n[{i}] {resp}")
        print("\n" + "=" * 40)

        # Get user preference
        while True:
            try:
                choice = input("Enter the number of the BETTER response (or 's' to skip): ").strip()
                if choice.lower() == 's':
                    print("Skipped - no preference saved.")
                    chosen_response = responses[0]
                    break

                chosen_idx = int(choice)
                if 0 <= chosen_idx < len(responses):
                    # Save preference
                    rejected_idx = 1 - chosen_idx if num_options == 2 else int(input("Enter rejected response number: "))

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

        # Update history with chosen response
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=chosen_response))

        return chosen_response

    def clear_history(self):
        """Clear conversation history."""
        self.chat_history.clear()

    def get_history(self) -> List:
        """Get conversation history."""
        return self.chat_history
