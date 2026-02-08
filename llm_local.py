"""Local LLM using DPO-trained model."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional


class LocalLLM:
    """Local LLM using the DPO fine-tuned model."""

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.2-1B-Instruct",
        adapter_path: Optional[str] = "models/dpo_medical_chatbot",
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading base model: {self.base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # Load LoRA adapter if available
        if self.adapter_path:
            try:
                print(f"Loading LoRA adapter from: {self.adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
                print("DPO-trained adapter loaded successfully!")
            except Exception as e:
                print(f"Could not load adapter: {e}")
                print("Using base model without DPO fine-tuning.")

        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate a response."""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from response
        response = response[len(prompt):].strip()

        return response


class LocalLLMWrapper:
    """Wrapper to make LocalLLM compatible with LangChain-style interface."""

    def __init__(self, local_llm: LocalLLM):
        self.llm = local_llm

    def invoke(self, messages: list) -> object:
        """Invoke the model with messages."""
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        response = self.llm.generate(prompt)

        # Return in LangChain-compatible format
        class Response:
            def __init__(self, content):
                self.content = content

        return Response(response)

    def _messages_to_prompt(self, messages: list) -> str:
        """Convert LangChain messages to a prompt string."""
        prompt_parts = []
        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
                msg_type = type(msg).__name__
            elif isinstance(msg, tuple):
                msg_type, content = msg
            else:
                continue

            if 'system' in str(msg_type).lower():
                prompt_parts.append(f"System: {content}")
            elif 'human' in str(msg_type).lower():
                prompt_parts.append(f"User: {content}")
            elif 'ai' in str(msg_type).lower():
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
