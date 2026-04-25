"""
memory/store.py

Two-tier memory:
  - ShortTermMemory: sliding window conversation buffer
  - LongTermMemory: ChromaDB vector store with semantic search

Usage:
    mem = AgentMemory()
    mem.add_message("user", "What is the capital of France?")
    mem.add_message("assistant", "Paris.")
    
    # Retrieve semantically relevant context
    context = await mem.recall("European capitals")
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_anthropic(self) -> dict:
        return {"role": self.role, "content": self.content}


class ShortTermMemory:
    """
    Sliding window conversation buffer.
    Keeps the last N turns to stay within the context window.
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._messages: list[Message] = []

    def add(self, role: str, content: str, metadata: dict | None = None):
        self._messages.append(Message(role=role, content=content, metadata=metadata or {}))
        # Trim to window — always keep pairs (user+assistant)
        if len(self._messages) > self.max_turns * 2:
            self._messages = self._messages[-(self.max_turns * 2):]

    def to_anthropic(self) -> list[dict]:
        return [m.to_anthropic() for m in self._messages]

    def clear(self):
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


class LongTermMemory:
    """
    Vector store backed by ChromaDB for semantic retrieval.
    Stores agent observations, facts, and completed task summaries.
    Falls back gracefully if ChromaDB is not installed.
    """

    def __init__(self, collection_name: str = "agent_memory"):
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        self._fallback: list[dict] = []  # simple list if ChromaDB unavailable
        self._chroma_available = self._init_chroma()

    def _init_chroma(self) -> bool:
        try:
            import chromadb
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return True
        except ImportError:
            return False

    def store(self, text: str, metadata: dict | None = None):
        """Store a piece of information in long-term memory."""
        doc_id = hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:12]
        if self._chroma_available and self._collection:
            self._collection.add(
                documents=[text],
                ids=[doc_id],
                metadatas=[metadata or {}],
            )
        else:
            self._fallback.append({"id": doc_id, "text": text, "metadata": metadata or {}})

    def recall(self, query: str, n_results: int = 3) -> list[str]:
        """Retrieve the top-N most semantically relevant memories."""
        if self._chroma_available and self._collection:
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(n_results, self._collection.count()),
                )
                return results["documents"][0] if results["documents"] else []
            except Exception:
                pass
        # Fallback: simple substring match
        matches = [
            m["text"] for m in self._fallback
            if any(word.lower() in m["text"].lower() for word in query.split())
        ]
        return matches[:n_results]

    def clear(self):
        if self._chroma_available and self._collection:
            self._collection.delete(where={"$ne": "__impossible__"})
        self._fallback.clear()

    def __len__(self) -> int:
        if self._chroma_available and self._collection:
            return self._collection.count()
        return len(self._fallback)


class AgentMemory:
    """Unified memory interface used by the agent loop."""

    def __init__(self, max_short_term_turns: int = 20):
        self.short_term = ShortTermMemory(max_turns=max_short_term_turns)
        self.long_term = LongTermMemory()

    def add_message(self, role: str, content: str):
        self.short_term.add(role, content)

    def memorize(self, text: str, metadata: dict | None = None):
        """Explicitly store something in long-term memory."""
        self.long_term.store(text, metadata)

    def recall(self, query: str, n: int = 3) -> list[str]:
        return self.long_term.recall(query, n_results=n)

    def get_conversation(self) -> list[dict]:
        return self.short_term.to_anthropic()

    def stats(self) -> dict:
        return {
            "short_term_messages": len(self.short_term),
            "long_term_memories": len(self.long_term),
        }
