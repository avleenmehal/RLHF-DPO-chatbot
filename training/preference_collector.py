"""Preference data collection for DPO training."""

import json
import os
from datetime import datetime
from typing import Optional


class PreferenceCollector:
    """Collects preference data (chosen/rejected pairs) for DPO training."""

    def __init__(self, output_path: str = "data/preferences.jsonl"):
        self.output_path = output_path
        self.current_prompt = None
        self.current_responses = []
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def start_collection(self, prompt: str, context: str = ""):
        """Start collecting responses for a prompt."""
        self.current_prompt = prompt
        self.current_context = context
        self.current_responses = []

    def add_response(self, response: str, source: str = "model"):
        """Add a response option."""
        self.current_responses.append({
            "text": response,
            "source": source
        })

    def save_preference(self, chosen_idx: int, rejected_idx: int):
        """Save a preference pair."""
        if len(self.current_responses) < 2:
            print("Need at least 2 responses to create a preference pair.")
            return False

        preference_data = {
            "prompt": self.current_prompt,
            "context": self.current_context,
            "chosen": self.current_responses[chosen_idx]["text"],
            "rejected": self.current_responses[rejected_idx]["text"],
            "timestamp": datetime.now().isoformat()
        }

        # Append to JSONL file
        with open(self.output_path, "a") as f:
            f.write(json.dumps(preference_data) + "\n")

        print(f"Preference saved to {self.output_path}")
        self._reset()
        return True

    def _reset(self):
        """Reset current collection state."""
        self.current_prompt = None
        self.current_context = ""
        self.current_responses = []

    def load_preferences(self) -> list:
        """Load all collected preferences."""
        preferences = []
        if os.path.exists(self.output_path):
            with open(self.output_path, "r") as f:
                for line in f:
                    preferences.append(json.loads(line.strip()))
        return preferences

    def get_stats(self) -> dict:
        """Get statistics about collected data."""
        preferences = self.load_preferences()
        return {
            "total_pairs": len(preferences),
            "file_path": self.output_path
        }


def collect_preferences_interactive():
    """Interactive CLI for collecting preference data."""
    collector = PreferenceCollector()

    print("=" * 50)
    print("Preference Data Collection for DPO")
    print("Type 'quit' to exit, 'stats' to see collection stats")
    print("=" * 50)

    while True:
        try:
            # Get prompt
            prompt = input("\nEnter prompt (or 'quit'/'stats'): ").strip()

            if prompt.lower() == "quit":
                break
            if prompt.lower() == "stats":
                stats = collector.get_stats()
                print(f"Total preference pairs: {stats['total_pairs']}")
                continue
            if not prompt:
                continue

            # Get context (optional)
            context = input("Enter context (optional, press Enter to skip): ").strip()

            collector.start_collection(prompt, context)

            # Get responses
            print("\nEnter responses (minimum 2). Type 'done' when finished:")
            idx = 1
            while True:
                response = input(f"Response {idx}: ").strip()
                if response.lower() == "done":
                    if len(collector.current_responses) >= 2:
                        break
                    print("Need at least 2 responses.")
                    continue
                if response:
                    collector.add_response(response, source="human")
                    idx += 1

            # Show responses and get preference
            print("\nResponses:")
            for i, resp in enumerate(collector.current_responses):
                print(f"  [{i}] {resp['text'][:100]}...")

            chosen = int(input("\nEnter index of CHOSEN (better) response: "))
            rejected = int(input("Enter index of REJECTED (worse) response: "))

            collector.save_preference(chosen, rejected)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    collect_preferences_interactive()
