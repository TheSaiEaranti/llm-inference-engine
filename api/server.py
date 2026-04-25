"""
api/server.py

FastAPI server exposing the inference engine and agent loop over HTTP.
Supports both standard JSON responses and Server-Sent Events (SSE) streaming.

Run: uvicorn api.server:app --reload --port 8000
"""

from __future__ import annotations

import time
import json
import asyncio
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from engine.inference import InferenceEngine, InferenceConfig
from engine.telemetry import tracer
from agent.loop import AgentLoop, AgentStreamEvent
from tools.registry import default_registry
from memory.store import AgentMemory


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic LLM Inference Engine",
    description="High-performance agentic inference powered by Claude.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared instances (in production, use dependency injection)
_engine = InferenceEngine()
_registry = default_registry()
_agent = AgentLoop(engine=_engine, registry=_registry)


# ── Request / Response models ─────────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class InferRequest(BaseModel):
    messages: list[Message]
    system: Optional[str] = None
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)


class InferResponse(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


class AgentRequest(BaseModel):
    task: str
    max_iterations: int = Field(default=10, ge=1, le=20)


class AgentResponse(BaseModel):
    final_answer: str
    steps: int
    tool_calls: int
    total_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    success: bool


class HealthResponse(BaseModel):
    status: str
    request_count: int
    avg_latency_ms: float
    tools: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/v1/health", response_model=HealthResponse)
async def health():
    stats = _engine.stats()
    return HealthResponse(
        status="ok",
        request_count=stats["request_count"],
        avg_latency_ms=stats["avg_latency_ms"],
        tools=_registry.list_tools(),
    )


@app.post("/v1/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    """Single-turn inference."""
    try:
        messages = [m.model_dump() for m in req.messages]
        result = await _engine.infer(messages=messages, system=req.system)
        return InferResponse(
            content=result.content,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            model=result.model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/infer/stream")
async def infer_stream(req: InferRequest):
    """Streaming inference via Server-Sent Events."""
    messages = [m.model_dump() for m in req.messages]

    async def event_generator() -> AsyncIterator[dict]:
        async for chunk in _engine.stream(messages=messages, system=req.system):
            if chunk.is_final and chunk.result:
                yield {
                    "event": "done",
                    "data": json.dumps({
                        "input_tokens": chunk.result.input_tokens,
                        "output_tokens": chunk.result.output_tokens,
                        "latency_ms": chunk.result.latency_ms,
                    }),
                }
            else:
                yield {"event": "delta", "data": chunk.delta}

    return EventSourceResponse(event_generator())


@app.post("/v1/agent/run", response_model=AgentResponse)
async def agent_run(req: AgentRequest):
    """Run an agentic task to completion."""
    try:
        agent = AgentLoop(engine=_engine, registry=_registry, max_iterations=req.max_iterations)
        run = await agent.run(req.task)
        summary = run.summary()
        return AgentResponse(
            final_answer=summary["final_answer"],
            steps=summary["steps"],
            tool_calls=summary["tool_calls"],
            total_latency_ms=summary["total_latency_ms"],
            total_input_tokens=summary["total_input_tokens"],
            total_output_tokens=summary["total_output_tokens"],
            success=summary["success"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/agent/stream")
async def agent_stream(req: AgentRequest):
    """Stream agent events (tool calls, thoughts, answer) via SSE."""
    async def event_generator() -> AsyncIterator[dict]:
        agent = AgentLoop(engine=_engine, registry=_registry, max_iterations=req.max_iterations)
        async for event in agent.stream(req.task):
            yield {
                "event": event.type,
                "data": json.dumps({
                    "content": event.content,
                    "metadata": event.metadata,
                }),
            }

    return EventSourceResponse(event_generator())


@app.get("/v1/tools")
async def list_tools():
    return {"tools": _registry.to_anthropic_schemas()}


@app.get("/v1/trace/{trace_id}")
async def get_trace(trace_id: str):
    spans = tracer.get_trace(trace_id)
    if not spans:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"trace_id": trace_id, "spans": spans}
