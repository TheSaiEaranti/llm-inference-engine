from __future__ import annotations
import time, asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from engine.telemetry import Span, tracer

MODEL = "claude-opus-4-5"
DEFAULT_MAX_TOKENS = 4096

@dataclass
class InferenceConfig:
    model: str = MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 1.0
    system: str = "You are a helpful, precise assistant."
    timeout: float = 60.0
    max_concurrency: int = 10

@dataclass
class InferenceResult:
    content: str
    raw_content: list
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    stop_reason: str

@dataclass
class StreamChunk:
    delta: str
    is_final: bool = False
    result: Optional[InferenceResult] = None

class InferenceEngine:
    def __init__(self, config: InferenceConfig | None = None):
        self.config = config or InferenceConfig()
        self._client = anthropic.AsyncAnthropic()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self._request_count = 0
        self._total_latency_ms = 0.0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10),
           retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)), reraise=True)
    async def infer(self, messages, tools=None, system=None):
        with tracer.span("engine.infer") as span:
            async with self._semaphore:
                start = time.monotonic()
                kwargs = self._build_kwargs(messages, tools, system)
                response = await self._client.messages.create(**kwargs)
                latency_ms = (time.monotonic() - start) * 1000
                self._record_stats(latency_ms)
                result = InferenceResult(
                    content=_extract_text(response.content),
                    raw_content=response.content,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_ms=latency_ms,
                    model=response.model,
                    stop_reason=response.stop_reason or "end_turn",
                )
                span.set_attributes({"input_tokens": result.input_tokens, "output_tokens": result.output_tokens, "latency_ms": round(latency_ms, 2), "stop_reason": result.stop_reason})
                return result

    async def stream(self, messages, tools=None, system=None):
        with tracer.span("engine.stream") as span:
            async with self._semaphore:
                start = time.monotonic()
                kwargs = self._build_kwargs(messages, tools, system)
                full_text, input_tokens, output_tokens, stop_reason = [], 0, 0, "end_turn"
                async with self._client.messages.stream(**kwargs) as stream_ctx:
                    async for event in stream_ctx:
                        if hasattr(event, "type"):
                            if event.type == "content_block_delta":
                                delta = getattr(event.delta, "text", "")
                                if delta:
                                    full_text.append(delta)
                                    yield StreamChunk(delta=delta)
                            elif event.type == "message_delta":
                                stop_reason = getattr(event.delta, "stop_reason", "end_turn") or "end_turn"
                                if hasattr(event, "usage"): output_tokens = event.usage.output_tokens
                            elif event.type == "message_start":
                                if hasattr(event.message, "usage"): input_tokens = event.message.usage.input_tokens
                latency_ms = (time.monotonic() - start) * 1000
                self._record_stats(latency_ms)
                yield StreamChunk(delta="", is_final=True, result=InferenceResult(content="".join(full_text), raw_content=[], input_tokens=input_tokens, output_tokens=output_tokens, latency_ms=latency_ms, model=self.config.model, stop_reason=stop_reason))

    async def batch(self, batch, system=None):
        return await asyncio.gather(*[self.infer(msgs, system=system) for msgs in batch])

    def stats(self):
        avg = self._total_latency_ms / self._request_count if self._request_count else 0
        return {"request_count": self._request_count, "avg_latency_ms": round(avg, 2)}

    def _build_kwargs(self, messages, tools, system):
        kwargs = {"model": self.config.model, "max_tokens": self.config.max_tokens, "messages": messages, "system": system or self.config.system}
        if tools: kwargs["tools"] = tools
        return kwargs

    def _record_stats(self, latency_ms):
        self._request_count += 1
        self._total_latency_ms += latency_ms

def _extract_text(content):
    return "".join(block.text for block in content if hasattr(block, "text"))
