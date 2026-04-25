from __future__ import annotations
import time, uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: str | None = None
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def set_attributes(self, attrs: dict[str, Any]):
        self.attributes.update(attrs)

    def finish(self):
        self.end_time = time.monotonic()

    @property
    def duration_ms(self) -> float:
        return round((self.end_time - self.start_time) * 1000, 2) if self.end_time else 0.0

    def to_dict(self) -> dict:
        return {"name": self.name, "trace_id": self.trace_id, "span_id": self.span_id,
                "duration_ms": self.duration_ms, "attributes": self.attributes, "status": self.status}

class Tracer:
    def __init__(self):
        self._spans: list[Span] = []
        self._active_trace_id: str | None = None

    @contextmanager
    def span(self, name: str) -> Generator[Span, None, None]:
        if self._active_trace_id is None:
            self._active_trace_id = uuid.uuid4().hex
        s = Span(name=name, trace_id=self._active_trace_id)
        self._spans.append(s)
        try:
            yield s
        except Exception as e:
            s.status = f"error: {type(e).__name__}"
            raise
        finally:
            s.finish()

    def get_trace(self, trace_id: str | None = None) -> list[dict]:
        tid = trace_id or self._active_trace_id
        return [s.to_dict() for s in self._spans if s.trace_id == tid]

    def clear(self):
        self._spans.clear()
        self._active_trace_id = None

tracer = Tracer()
