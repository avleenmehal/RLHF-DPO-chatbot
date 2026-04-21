"""Redis-backed cache — query embedding cache + post-RAG context cache."""

import hashlib
import json
import re
from typing import Optional

import nltk
import redis
from nltk.corpus import stopwords

from core.config import Config

# Download stop words on first use (no-op if already downloaded)
nltk.download("stopwords", quiet=True)
_STOP_WORDS = set(stopwords.words("english"))


class CacheManager:
    """
    Two cache layers:
      1. Embedding cache  — hash(query) → vector (no TTL, embeddings are deterministic)
      2. Context cache    — hash(query) → processed context string (TTL = 24h)

    Fails silently if Redis is unavailable so the app keeps running without cache.
    """

    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._connect()

    def _connect(self):
        try:
            client = redis.from_url(
                Config.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
            self._client = client
            print(f"[Cache] Redis connected at {Config.REDIS_URL}")
        except Exception as e:
            self._client = None
            print(f"[Cache] Redis unavailable — running without cache. ({e})")

    @property
    def available(self) -> bool:
        return self._client is not None

    # ── Key helper ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(query: str) -> str:
        """Strip punctuation, remove stop words, sort remaining words.

        "how do I get a kidney transplant" → "get kidney transplant"
        "what are steps for kidney transplant" → "kidney steps transplant"
        Both produce the same sorted token set → same cache key.
        """
        text = re.sub(r"[^\w\s]", "", query.lower())
        words = [w for w in text.split() if w not in _STOP_WORDS and len(w) > 1]
        return " ".join(sorted(words))

    @staticmethod
    def _key(prefix: str, query: str) -> str:
        normalized = CacheManager._normalize(query)
        digest = hashlib.sha256(normalized.encode()).hexdigest()[:20]
        return f"medchat:{prefix}:{digest}"

    # ── Embedding cache (no TTL) ──────────────────────────────────────────

    def get_embedding(self, query: str) -> Optional[list[float]]:
        """Return cached embedding vector or None on miss / Redis down."""
        if not self.available:
            return None
        try:
            raw = self._client.get(self._key("emb", query))
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def set_embedding(self, query: str, vector: list[float]):
        """Store embedding vector. No TTL — same text always produces same vector."""
        if not self.available:
            return
        try:
            self._client.set(self._key("emb", query), json.dumps(vector))
        except Exception:
            pass

    # ── Context cache (post-RAG, with TTL) ───────────────────────────────

    def get_context(self, query: str) -> Optional[str]:
        """
        Return cached processed context string or None on miss.

        The cached value is the fully formatted context passed to the LLM —
        after retrieval, re-ranking (if any), and chunk formatting.
        """
        if not self.available:
            return None
        try:
            return self._client.get(self._key("ctx", query))
        except Exception:
            return None

    def set_context(self, query: str, context: str, ttl: int = None):
        """Store processed context string with a TTL (default 24h)."""
        if not self.available:
            return
        try:
            ttl = ttl or Config.CONTEXT_CACHE_TTL
            self._client.setex(self._key("ctx", query), ttl, context)
        except Exception:
            pass

    def invalidate_context(self):
        """
        Delete all context cache entries.
        Call this after rebuilding the FAISS index from new CSV data
        so stale contexts don't get served.
        """
        if not self.available:
            return
        try:
            keys = list(self._client.scan_iter("medchat:ctx:*"))
            if keys:
                self._client.delete(*keys)
                print(f"[Cache] Invalidated {len(keys)} context cache entries.")
        except Exception:
            pass

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return basic cache stats for debugging."""
        if not self.available:
            return {"status": "unavailable"}
        try:
            emb_keys = sum(1 for _ in self._client.scan_iter("medchat:emb:*"))
            ctx_keys = sum(1 for _ in self._client.scan_iter("medchat:ctx:*"))
            return {
                "status": "connected",
                "embedding_entries": emb_keys,
                "context_entries": ctx_keys,
            }
        except Exception:
            return {"status": "error"}


# Singleton — imported by rag/pipeline.py and core/chatbot.py
cache = CacheManager()
