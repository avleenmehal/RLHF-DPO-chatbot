"""Real-time response quality evaluation posted back to LangSmith traces."""

import logging
import uuid
from threading import Thread
from typing import Optional

from core.config import Config
from evals.langsmith_evals import EVALUATORS

logger = logging.getLogger(__name__)


class _MockRun:
    def __init__(self, question: str, answer: str, context: str):
        self.inputs = {"question": question}
        self.outputs = {"answer": answer, "context": context}


class _MockExample:
    outputs: dict = {}


class OnlineEvaluator:
    """Runs LLM-as-judge evaluators asynchronously on every live response.

    Scores are posted back to the LangSmith trace as feedback so they appear
    alongside the full execution tree in the UI — zero impact on response latency.

    Fails silently if LangSmith is not configured.
    """

    _instance: Optional["OnlineEvaluator"] = None

    def __init__(self):
        self._enabled = bool(Config.LANGCHAIN_API_KEY)
        if self._enabled:
            from langsmith import Client
            self._client = Client()
            print("[EVAL] OnlineEvaluator active — scoring every live response via LangSmith")
        else:
            print("[EVAL] OnlineEvaluator disabled — set LANGCHAIN_API_KEY to enable")

    @classmethod
    def get_instance(cls) -> "OnlineEvaluator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def enabled(self) -> bool:
        return self._enabled

    def evaluate_async(
        self,
        question: str,
        answer: str,
        context: str = "",
        run_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Fire-and-forget: score the response in a daemon thread."""
        if not self._enabled:
            return
        Thread(
            target=self._run_evals,
            args=(question, answer, context, run_id),
            daemon=True,
        ).start()

    def _run_evals(
        self,
        question: str,
        answer: str,
        context: str,
        run_id: Optional[uuid.UUID],
    ) -> None:
        mock_run = _MockRun(question, answer, context)
        mock_example = _MockExample()

        for evaluator in EVALUATORS:
            try:
                result = evaluator(mock_run, mock_example)
                score = result.get("score")
                key = result.get("key", evaluator.__name__)
                comment = result.get("comment", "")

                label = f"{score:.2f}" if score is not None else "n/a"
                print(f"[EVAL] {key}={label}  {comment[:70]}")

                if run_id is not None and score is not None:
                    self._client.create_feedback(
                        run_id=run_id,
                        key=key,
                        score=score,
                        comment=comment,
                    )
            except Exception as exc:
                logger.warning("[EVAL] %s failed: %s", evaluator.__name__, exc)
