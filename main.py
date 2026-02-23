"""Main entry point for the Medical Chatbot."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from chatbot import MedicalChatbot
from rag import RAGPipeline
from config import Config
from llm import ModelType


def setup_rag(csv_path: str = None) -> RAGPipeline:
    """Initialize RAG pipeline."""
    rag = RAGPipeline(csv_path=csv_path)

    # Try to load existing vector store
    if rag.load_vector_store():
        return rag

    # Create new vector store from CSV
    try:
        rag.create_vector_store()
        rag.save_vector_store()
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Running without RAG context. Add CSV file and restart.")
        return None

    return rag


def get_model_type(model_arg: str) -> ModelType:
    """Convert string argument to ModelType."""
    model_map = {
        "openai": ModelType.OPENAI,
        "local": ModelType.LOCAL_BASE,
        "local-base": ModelType.LOCAL_BASE,
        "local-dpo": ModelType.LOCAL_DPO,
        "dpo": ModelType.LOCAL_DPO,
    }
    return model_map.get(model_arg.lower(), ModelType.OPENAI)


def main():
    """Run the chatbot."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Medical Chatbot with RAG")
    parser.add_argument("--csv", type=str, help="Path to CSV file")
    parser.add_argument("--collect-preferences", action="store_true",
                        help="Enable preference collection mode for DPO training")
    parser.add_argument("--num-responses", type=int, default=2,
                        help="Number of responses to generate in preference mode (default: 2)")
    parser.add_argument("--model", type=str, default="openai",
                        choices=["openai", "local", "local-base", "local-dpo", "dpo"],
                        help="Model to use: openai, local/local-base (Llama), local-dpo/dpo (DPO-trained)")
    args = parser.parse_args()

    # Get model type
    model_type = get_model_type(args.model)

    # Validate configuration
    try:
        require_openai = model_type == ModelType.OPENAI
        Config.validate(require_openai=require_openai)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Setup RAG pipeline (still uses OpenAI embeddings)
    print("Initializing RAG pipeline...")
    rag = setup_rag(args.csv)

    # Initialize chatbot
    chatbot = MedicalChatbot(
        rag_pipeline=rag,
        collect_preferences=args.collect_preferences,
        model_type=model_type,
    )

    # Run chat loop
    mode = "PREFERENCE COLLECTION" if args.collect_preferences else "CHAT"
    model_name = {
        ModelType.OPENAI: "GPT-4.1",
        ModelType.LOCAL_BASE: "Llama (Base)",
        ModelType.LOCAL_DPO: "Llama (DPO-trained)",
    }[model_type]

    print("\n" + "=" * 50)
    print(f"Medical Chatbot ({model_name} + RAG) - {mode} MODE")
    print("Type 'quit' to exit, 'clear' to reset history, 'stats' for preference stats")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                chatbot.clear_history()
                print("Chat history cleared.")
                continue

            if user_input.lower() == "stats" and args.collect_preferences:
                stats = chatbot.preference_collector.get_stats()
                print(f"Preference pairs collected: {stats['total_pairs']}")
                continue

            # Use preference collection or normal chat
            if args.collect_preferences:
                response = chatbot.chat_with_preference(user_input, args.num_responses)
            else:
                response = chatbot.chat(user_input)

            print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
