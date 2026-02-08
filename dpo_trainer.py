"""DPO (Direct Preference Optimization) Training for Medical Chatbot."""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

from config import Config


class DPOTrainerSetup:
    """Setup and run DPO training on preference data."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        preferences_path: str = "data/preferences.jsonl",
        output_dir: str = "models/dpo_medical_chatbot",
    ):
        self.model_name = model_name
        self.preferences_path = preferences_path
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_preferences(self) -> Dataset:
        """Load preference data and convert to DPO format."""
        data = []
        with open(self.preferences_path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                # DPO format: prompt, chosen, rejected
                data.append({
                    "prompt": self._format_prompt(item["prompt"], item.get("context", "")),
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                })

        print(f"Loaded {len(data)} preference pairs")
        return Dataset.from_list(data)

    def _format_prompt(self, question: str, context: str) -> str:
        """Format prompt with context."""
        return f"""You are an AI medical researcher. Use the context to answer the question.
Maintain a friendly tone and include appropriate humor when suitable.

Context: {context}

Question: {question}

Answer:"""

    def setup_model(self, use_4bit: bool = True):
        """Load and configure the model for training."""
        print(f"Loading model: {self.model_name}")

        # Quantization config for memory efficiency
        if use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )

        if use_4bit and torch.cuda.is_available():
            self.model = prepare_model_for_kbit_training(self.model)

        # Setup LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        return self.model

    def setup_trainer(self, dataset: Dataset, eval_split: float = 0.1):
        """Configure the DPO trainer."""
        # Split dataset
        split = dataset.train_test_split(test_size=eval_split)
        train_dataset = split["train"]
        eval_dataset = split["test"]

        # DPO training config
        training_args = DPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            beta=0.1,  # DPO temperature parameter
            max_length=512,
            max_prompt_length=256,
            logging_steps=10,
            save_steps=100,
            eval_strategy="steps",
            eval_steps=50,
            warmup_ratio=0.1,
            bf16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            remove_unused_columns=False,
        )

        # Create trainer
        self.trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        return self.trainer

    def train(self):
        """Run DPO training."""
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")

        print("Starting DPO training...")
        self.trainer.train()

        # Save the final model
        print(f"Saving model to {self.output_dir}")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        print("Training complete!")

    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        # Load data
        dataset = self.load_preferences()

        if len(dataset) < 10:
            print("Warning: Less than 10 preference pairs. Collect more data for better results.")

        # Setup model
        self.setup_model()

        # Setup trainer
        self.setup_trainer(dataset)

        # Train
        self.train()


def main():
    """Run DPO training."""
    import argparse

    parser = argparse.ArgumentParser(description="DPO Training for Medical Chatbot")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--preferences", type=str, default="data/preferences.jsonl",
                        help="Path to preferences file")
    parser.add_argument("--output", type=str, default="models/dpo_medical_chatbot",
                        help="Output directory for trained model")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization")
    args = parser.parse_args()

    trainer = DPOTrainerSetup(
        model_name=args.model,
        preferences_path=args.preferences,
        output_dir=args.output,
    )

    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()
