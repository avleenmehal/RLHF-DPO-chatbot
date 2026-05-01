"""Guardrails AI — input and output safety layer for the medical chatbot."""

import logging
import os
import re
import time
from typing import Optional, Any

logger = logging.getLogger(__name__)

# ── Block messages ────────────────────────────────────────────────────────────

_MSG_OFF_TOPIC = (
    "I'm a medical AI assistant and can only help with health and medical questions. "
    "Please ask me about symptoms, treatments, medications, or other medical topics."
)
_MSG_TOXIC = (
    "I'm unable to respond to that message. Please keep the conversation respectful "
    "and focused on medical topics."
)
_MSG_PII = (
    "Your message appears to contain personal information such as a phone number or ID. "
    "Please avoid sharing personal details — I don't need them to answer medical questions."
)
_MSG_OUTPUT_UNSAFE = (
    "I wasn't able to generate a safe response for that question. "
    "Please try rephrasing your question."
)

_ALL_BLOCK_MSGS = (_MSG_PII, _MSG_TOXIC, _MSG_OFF_TOPIC, _MSG_OUTPUT_UNSAFE)

# ── Keyword heuristics for medical topic check ────────────────────────────────

_MEDICAL_KEYWORDS = {
    "symptom", "symptoms", "disease", "disorder", "condition", "treatment",
    "medication", "drug", "diagnosis", "diagnose", "doctor", "patient",
    "hospital", "clinic", "pain", "fever", "infection", "cancer", "diabetes",
    "blood", "heart", "lung", "kidney", "liver", "brain", "surgery", "therapy",
    "vaccine", "pill", "dose", "prescription", "allergy", "chronic", "acute",
    "medical", "health", "healthcare", "medicine", "physician", "nurse",
    "pharmacist", "anxiety", "depression", "headache", "nausea", "fatigue",
    "vitamin", "supplement", "antibiotic", "virus", "bacteria", "immune",
    "pregnant", "pregnancy", "side effect", "side effects", "risk factor",
    "what is", "how to treat", "causes of", "how does", "what are",
}

_OFF_TOPIC_KEYWORDS = {
    "recipe", "cooking", "baking", "basketball", "football", "soccer",
    "crypto", "bitcoin", "ethereum", "stock market", "investing", "mortgage",
    "politics", "election", "president", "congress",
    "movie", "netflix", "music", "album", "gaming", "video game", "minecraft",
}

# ── Guardrails AI import and validator registration ───────────────────────────

_GUARDRAILS_AVAILABLE = False
_Guard = None
_MedicalTopicValidator = None
_PIIValidator = None
_ToxicContentValidator = None

try:
    try:
        from guardrails.validator_base import (  # type: ignore
            Validator,
            register_validator,
            PassResult,
            FailResult,
        )
    except ImportError:
        from guardrails import Validator, register_validator  # type: ignore
        from guardrails.validators import PassResult, FailResult  # type: ignore

    from guardrails import Guard as _Guard  # type: ignore

    @register_validator(name="grd/medical-topic", data_type="string")
    class _MedicalTopicValidatorImpl(Validator):
        _CLASSIFIER_PROMPT = (
            "You are a strict topic classifier for a medical AI assistant. "
            "Reply with only YES or NO.\n"
            "Is the following message related to health, medicine, symptoms, treatments, "
            "medications, or medical topics?\n\nMessage: {message}"
        )

        def validate(self, value: Any, metadata: dict):
            text = str(value)
            lower = text.lower()

            # Any medical keyword → allow immediately, no LLM call needed
            if any(kw in lower for kw in _MEDICAL_KEYWORDS):
                return PassResult()

            # Clear off-topic with no medical context → block immediately, no LLM call
            if any(kw in lower for kw in _OFF_TOPIC_KEYWORDS):
                return FailResult(error_message=_MSG_OFF_TOPIC)

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return PassResult()

            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": self._CLASSIFIER_PROMPT.format(message=text)},
                    ],
                    max_tokens=5,
                    temperature=0,
                )
                answer = resp.choices[0].message.content.strip().upper()
                if not answer.startswith("YES"):
                    return FailResult(error_message=_MSG_OFF_TOPIC)
            except Exception as exc:
                logger.warning("MedicalTopicValidator: LLM classification error (fail-open): %s", exc)

            return PassResult()

    @register_validator(name="grd/pii-check", data_type="string")
    class _PIIValidatorImpl(Validator):
        _SSN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        _PHONE = re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b')
        _CC = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')

        def validate(self, value: Any, metadata: dict):
            text = str(value)
            if self._SSN.search(text) or self._PHONE.search(text) or self._CC.search(text):
                return FailResult(error_message=_MSG_PII)
            return PassResult()

    @register_validator(name="grd/toxic-content", data_type="string")
    class _ToxicContentValidatorImpl(Validator):
        def validate(self, value: Any, metadata: dict):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return PassResult()
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                resp = client.moderations.create(input=str(value))
                if resp.results[0].flagged:
                    return FailResult(error_message=_MSG_TOXIC)
            except Exception as exc:
                logger.warning("ToxicContentValidator: moderation error (fail-open): %s", exc)
            return PassResult()

    _MedicalTopicValidator = _MedicalTopicValidatorImpl
    _PIIValidator = _PIIValidatorImpl
    _ToxicContentValidator = _ToxicContentValidatorImpl
    _GUARDRAILS_AVAILABLE = True

except ImportError:
    pass
except Exception as exc:
    logger.warning("Guardrails AI validator registration failed: %s", exc)


# ── Manager ───────────────────────────────────────────────────────────────────

class GuardrailsManager:
    """Input and output safety layer using Guardrails AI validators.

    Fails open: if guardrails-ai is not installed or any check fails unexpectedly,
    all check calls return (False, "") so the chatbot continues normally.
    """

    _instance: Optional["GuardrailsManager"] = None

    def __init__(self):
        self._input_guard = None
        self._output_guard = None

        if not _GUARDRAILS_AVAILABLE:
            print("[GUARDRAILS] guardrails-ai not installed — disabled (fail-open)")
            logger.warning("guardrails-ai not installed — guardrails disabled")
            return

        try:
            self._input_guard = _Guard().use(
                _PIIValidator(on_fail="exception"),
                _ToxicContentValidator(on_fail="exception"),
                _MedicalTopicValidator(on_fail="exception"),
            )
            self._output_guard = _Guard().use(
                _ToxicContentValidator(on_fail="exception"),
            )
            print("[GUARDRAILS] Guardrails AI initialised — input + output guards active")
            logger.info("Guardrails AI initialised")
        except Exception as exc:
            print(f"[GUARDRAILS] Guard init failed ({exc}) — disabled (fail-open)")
            logger.warning("Guardrails AI init failed: %s", exc)

    @classmethod
    def get_instance(cls) -> "GuardrailsManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def enabled(self) -> bool:
        return self._input_guard is not None

    def _run_guard(self, guard, text: str, label: str) -> tuple[bool, str]:
        """Run a guard and return (is_blocked, block_message). Always fails open."""
        if guard is None:
            return False, ""
        try:
            # validate() is 0.5.x+; fall back to parse() for older versions
            if hasattr(guard, "validate"):
                guard.validate(text)
            else:
                guard.parse(text)
            return False, ""
        except Exception as exc:
            exc_str = str(exc)
            # Extract our known block message from the exception string
            for msg in _ALL_BLOCK_MSGS:
                if msg in exc_str:
                    return True, msg
            # Validation exception with unknown message — still block
            exc_type = type(exc).__name__.lower()
            if any(kw in exc_type for kw in ("validation", "guard", "fail")):
                fallback = _MSG_OUTPUT_UNSAFE if label == "output" else _MSG_TOXIC
                return True, fallback
            # Truly unexpected error — fail open
            logger.error("Guardrails %s guard unexpected error (fail-open): %s", label, exc)
            return False, ""

    def check_input_sync(self, user_input: str) -> tuple[bool, str]:
        """Validate user input. Returns (is_blocked, block_message)."""
        preview = user_input[:80] + ("..." if len(user_input) > 80 else "")
        print(f"\n[GUARDRAILS] Checking input: '{preview}'")
        t0 = time.perf_counter()

        is_blocked, msg = self._run_guard(self._input_guard, user_input, "input")
        elapsed = (time.perf_counter() - t0) * 1000

        if is_blocked:
            print(f"[GUARDRAILS] INPUT BLOCKED ({elapsed:.0f}ms) | '{msg[:60]}'")
        else:
            print(f"[GUARDRAILS] INPUT PASSED ({elapsed:.0f}ms)")

        return is_blocked, msg

    def check_output_sync(self, response: str) -> tuple[bool, str]:
        """Validate LLM output. Returns (is_blocked, replacement_message)."""
        preview = response[:80] + ("..." if len(response) > 80 else "")
        print(f"\n[GUARDRAILS] Checking output: '{preview}'")
        t0 = time.perf_counter()

        is_blocked, _ = self._run_guard(self._output_guard, response, "output")
        elapsed = (time.perf_counter() - t0) * 1000

        if is_blocked:
            print(f"[GUARDRAILS] OUTPUT BLOCKED ({elapsed:.0f}ms)")
        else:
            print(f"[GUARDRAILS] OUTPUT PASSED ({elapsed:.0f}ms)")

        return is_blocked, _MSG_OUTPUT_UNSAFE if is_blocked else ""
