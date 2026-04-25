"""
Guardrail layer for the medical chatbot.

Input guardrails  — NeMo Guardrails (Colang flows in guardrails/)
  1. Jailbreak / prompt injection detection
  2. Off-topic detection
  3. Medical emergency / crisis detection  → immediate crisis resources

Output guardrails — Python post-processing (runs on the complete response)
  1. Medical disclaimer enforcement
  2. Crisis resource injection when response touches self-harm topics
  3. Specific drug-dosage warning
"""

import asyncio
import concurrent.futures
import logging
import os
import re

logger = logging.getLogger(__name__)

# ── Sentinel ──────────────────────────────────────────────────────────────────
# NeMo's main model returns this when no rail fires → message is safe.
_SAFE_SENTINEL = "__INPUT_SAFE__"

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


# ── NeMo rails (lazy-loaded so startup is fast if nemoguardrails is missing) ──

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
        logger.warning(f"[Guardrails] NeMo not available — input rails disabled: {e}")
        _rails = None


# ── Public API ────────────────────────────────────────────────────────────────

def check_input(message: str) -> tuple[bool, str | None]:
    """
    Run NeMo input rails synchronously.

    Returns:
        (True, None)          — message is safe, proceed normally
        (False, str)          — message was blocked; show the str to the user
    """
    _load_rails()
    if _rails is None:
        return True, None  # NeMo unavailable — degrade gracefully

    try:
        # Run async NeMo in a dedicated thread with its own event loop
        # so we don't conflict with FastAPI/Gradio's event loop.
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
        logger.warning("[Guardrails] Input check timed out — allowing message through.")
        return True, None
    except Exception as e:
        logger.warning(f"[Guardrails] Input check error — allowing message through: {e}")
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
            # Append caveat after the first dosage mention
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
