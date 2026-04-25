"""
Guardrail layer for the medical chatbot.

Input guardrails — run in order, first failure wins:
  1. OpenAI Moderation API  — free, fast, catches hate/violence/self-harm/sexual
  2. NeMo Guardrails        — Colang flows for jailbreak, off-topic, emergency

Output guardrails — Python post-processing on the complete response:
  1. Crisis resource injection when response touches self-harm topics
  2. Specific drug-dosage caveat
  3. Medical disclaimer enforcement
"""

import asyncio
import concurrent.futures
import logging
import os
import re

logger = logging.getLogger(__name__)

# ── Shared constants ──────────────────────────────────────────────────────────

_CRISIS_EMERGENCY = (
    "⚠️ This sounds like a medical emergency.\n\n"
    "**Please call 911 immediately** (or your local emergency number).\n\n"
    "If you are experiencing thoughts of suicide or self-harm:\n"
    "- Call or text **988** (Suicide & Crisis Lifeline, US)\n"
    "- Text HOME to **741741** (Crisis Text Line)\n"
    "- Go to your nearest emergency department\n\n"
    "Please reach out now — help is available."
)

# ── Output rail constants ─────────────────────────────────────────────────────

_DISCLAIMER = (
    "\n\n---\n"
    "*For informational purposes only — not a substitute for professional "
    "medical advice, diagnosis, or treatment. Always consult a qualified "
    "healthcare provider before making health decisions.*"
)

_CRISIS_LINE = (
    "\n\n> **If you or someone you know is in crisis:** "
    "call or text **988** (Suicide & Crisis Lifeline) "
    "or go to your nearest emergency department."
)

_CRISIS_KEYWORDS = {
    "suicid", "kill myself", "end my life", "self-harm",
    "self harm", "overdose", "cutting myself", "hurt myself",
}

_DOSAGE_PATTERN = re.compile(
    r"\b\d+\s*(?:mg|mcg|ml|units?|tablets?|capsules?)\b",
    re.IGNORECASE,
)

_DOSAGE_CAVEAT = (
    " *(Dosage and treatment details must be confirmed with your prescribing "
    "clinician — individual needs vary.)*"
)

# ── OpenAI Moderation API ─────────────────────────────────────────────────────

# Per-category user-facing messages.  Self-harm variants reuse the full
# emergency response so users always see crisis line info.
_MODERATION_MESSAGES: dict[str, str] = {
    "self-harm":              _CRISIS_EMERGENCY,
    "self-harm/intent":       _CRISIS_EMERGENCY,
    "self-harm/instructions": _CRISIS_EMERGENCY,
    "hate":                   "I'm not able to engage with content that demeans or targets groups of people. If you have a health question, I'm happy to help.",
    "hate/threatening":       "I'm not able to engage with threatening content. If you have a health question, I'm happy to help.",
    "harassment":             "I'm not able to engage with harassing content. If you have a health question, I'm happy to help.",
    "harassment/threatening": "I'm not able to engage with threatening content. If you have a health question, I'm happy to help.",
    "violence":               "I'm a medical assistant and can only help with health-related questions.",
    "violence/graphic":       "I'm a medical assistant and can only help with health-related questions.",
    "sexual":                 "I'm not able to help with that type of request.",
    "sexual/minors":          "I'm not able to help with that type of request.",
}

_MODERATION_FALLBACK = "I'm not able to help with that request. If you have a health question, I'm happy to assist."

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()  # reads OPENAI_API_KEY from env
    return _openai_client


def _check_moderation(message: str) -> tuple[bool, str | None]:
    """
    Call the OpenAI Moderation API (free, ~100ms).

    Returns:
        (True, None)   — content is clean
        (False, str)   — content was flagged; str is the user-facing response
    """
    try:
        result = _get_openai_client().moderations.create(
            model="omni-moderation-latest",
            input=message,
        )
        outcome = result.results[0]
        if not outcome.flagged:
            return True, None

        # Find the highest-scoring flagged category to pick the best message
        scores = outcome.category_scores.model_dump()
        flagged_cats = [c for c, flagged in outcome.categories.model_dump().items() if flagged]
        top_category = max(flagged_cats, key=lambda c: scores.get(c, 0))

        logger.info(f"[Moderation] Flagged — category: {top_category}, score: {scores.get(top_category, 0):.3f}")
        return False, _MODERATION_MESSAGES.get(top_category, _MODERATION_FALLBACK)

    except Exception as e:
        # Never block a user because the moderation API is down
        logger.warning(f"[Moderation] API error — skipping check: {e}")
        return True, None


# ── NeMo Guardrails ───────────────────────────────────────────────────────────

# Sentinel returned by NeMo's main model when no rail fires → input is safe
_SAFE_SENTINEL = "__INPUT_SAFE__"

_rails = None
_rails_loaded = False


def _load_rails():
    global _rails, _rails_loaded
    if _rails_loaded:
        return
    _rails_loaded = True
    try:
        from nemoguardrails import RailsConfig, LLMRails
        config_path = os.path.join(os.path.dirname(__file__), "..", "guardrails")
        config = RailsConfig.from_path(os.path.abspath(config_path))
        _rails = LLMRails(config)
        logger.info("[Guardrails] NeMo rails loaded.")
    except Exception as e:
        logger.warning(f"[Guardrails] NeMo not available — NeMo rails disabled: {e}")
        _rails = None


def _check_nemo(message: str) -> tuple[bool, str | None]:
    """
    Run NeMo Colang input rails in a fresh thread (avoids event-loop conflicts
    with FastAPI / Gradio's existing event loop).

    Returns:
        (True, None)   — passed all rails
        (False, str)   — a rail fired; str is the rail's response
    """
    _load_rails()
    if _rails is None:
        return True, None

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                _rails.generate_async(messages=[{"role": "user", "content": message}]),
            )
            result: str = future.result(timeout=20)

        if _SAFE_SENTINEL in result:
            return True, None
        return False, result

    except concurrent.futures.TimeoutError:
        logger.warning("[Guardrails] NeMo check timed out — allowing through.")
        return True, None
    except Exception as e:
        logger.warning(f"[Guardrails] NeMo error — allowing through: {e}")
        return True, None


# ── Public API ────────────────────────────────────────────────────────────────

def check_input(message: str) -> tuple[bool, str | None]:
    """
    Run all input guardrails in priority order.

    Order:
      1. OpenAI Moderation API  (fast, free, catches policy violations)
      2. NeMo Guardrails        (Colang: jailbreak, off-topic, emergency)

    Returns:
        (True, None)   — safe to proceed
        (False, str)   — blocked; show str to the user
    """
    # Step 1 — OpenAI Moderation (runs first: cheap and fast)
    safe, response = _check_moderation(message)
    if not safe:
        return False, response

    # Step 2 — NeMo Colang rails
    safe, response = _check_nemo(message)
    if not safe:
        return False, response

    return True, None


def process_output(response: str, query: str = "") -> str:
    """
    Post-process the complete LLM response with output guardrails.

    Applied in order:
      1. Inject crisis resources if topic is sensitive
      2. Flag specific drug dosages with a caveat
      3. Ensure medical disclaimer is present
    """
    response = _inject_crisis_line(response, query)
    response = _flag_dosages(response)
    response = _ensure_disclaimer(response)
    return response


# ── Output guardrail helpers ──────────────────────────────────────────────────

def _inject_crisis_line(response: str, query: str) -> str:
    combined = (query + " " + response).lower()
    if any(kw in combined for kw in _CRISIS_KEYWORDS):
        if "988" not in response and "crisis" not in response.lower():
            response += _CRISIS_LINE
    return response


def _flag_dosages(response: str) -> str:
    if _DOSAGE_PATTERN.search(response):
        if "prescribing" not in response and "clinician" not in response.lower():
            response = _DOSAGE_PATTERN.sub(
                lambda m: m.group(0) + _DOSAGE_CAVEAT,
                response,
                count=1,
            )
    return response


def _ensure_disclaimer(response: str) -> str:
    markers = ("not a substitute", "healthcare provider", "consult a", "informational purposes")
    if not any(m in response.lower() for m in markers):
        response += _DISCLAIMER
    return response
