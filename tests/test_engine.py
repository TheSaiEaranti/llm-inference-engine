"""
tests/test_engine.py

Integration tests for the inference engine, tools, memory, and agent.
Run: pytest tests/ -v
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from engine.inference import InferenceEngine, InferenceConfig, InferenceResult
from engine.telemetry import Tracer
from tools.registry import CalculatorTool, ToolRegistry, default_registry
from memory.store import AgentMemory, ShortTermMemory, LongTermMemory
from agent.loop import AgentLoop


# ── Tool tests ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_calculator_basic():
    tool = CalculatorTool()
    result = await tool.execute(expression="2 ** 10")
    assert result.error is None
    assert result.output == 1024


@pytest.mark.asyncio
async def test_calculator_math_functions():
    tool = CalculatorTool()
    result = await tool.execute(expression="sqrt(144)")
    assert result.error is None
    assert result.output == 12.0


@pytest.mark.asyncio
async def test_calculator_error():
    tool = CalculatorTool()
    result = await tool.execute(expression="import os")
    assert result.error is not None


@pytest.mark.asyncio
async def test_registry_dispatch():
    registry = default_registry()
    result = await registry.execute("calculator", {"expression": "1 + 1"})
    assert result.output == 2


@pytest.mark.asyncio
async def test_registry_parallel():
    registry = default_registry()
    calls = [
        {"name": "calculator", "inputs": {"expression": "2 + 2"}},
        {"name": "calculator", "inputs": {"expression": "3 * 3"}},
    ]
    results = await registry.execute_parallel(calls)
    assert results[0].output == 4
    assert results[1].output == 9


@pytest.mark.asyncio
async def test_registry_unknown_tool():
    registry = ToolRegistry()
    result = await registry.execute("nonexistent_tool", {})
    assert result.error is not None


# ── Memory tests ──────────────────────────────────────────────────────────────

def test_short_term_memory_window():
    mem = ShortTermMemory(max_turns=2)
    for i in range(10):
        mem.add("user", f"message {i}")
        mem.add("assistant", f"response {i}")
    # Should only keep last 2 turns (4 messages)
    assert len(mem) == 4


def test_short_term_to_anthropic():
    mem = ShortTermMemory()
    mem.add("user", "hello")
    mem.add("assistant", "hi there")
    msgs = mem.to_anthropic()
    assert msgs[0] == {"role": "user", "content": "hello"}
    assert msgs[1] == {"role": "assistant", "content": "hi there"}


def test_long_term_memory_store_recall():
    mem = LongTermMemory()
    mem.store("The Eiffel Tower is in Paris, France.")
    mem.store("Python was created by Guido van Rossum.")
    results = mem.recall("Paris France tower")
    assert len(results) >= 1


def test_agent_memory_unified():
    mem = AgentMemory()
    mem.add_message("user", "test query")
    mem.add_message("assistant", "test response")
    assert len(mem.get_conversation()) == 2
    stats = mem.stats()
    assert "short_term_messages" in stats


# ── Telemetry tests ───────────────────────────────────────────────────────────

def test_tracer_span():
    tracer = Tracer()
    with tracer.span("test.op") as span:
        span.set_attributes({"key": "value"})
    
    trace = tracer.get_trace()
    assert len(trace) == 1
    assert trace[0]["name"] == "test.op"
    assert trace[0]["attributes"]["key"] == "value"
    assert trace[0]["duration_ms"] > 0


def test_tracer_error_captured():
    tracer = Tracer()
    with pytest.raises(ValueError):
        with tracer.span("failing.op"):
            raise ValueError("oops")
    
    trace = tracer.get_trace()
    assert "error" in trace[0]["status"]


# ── Engine tests (mocked) ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_engine_infer_mocked():
    """Test engine with a mocked Anthropic client."""
    engine = InferenceEngine()

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello, world!")]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.model = "claude-opus-4-5"
    mock_response.stop_reason = "end_turn"

    with patch.object(engine._client.messages, "create", return_value=mock_response) as mock_create:
        mock_create = AsyncMock(return_value=mock_response)
        engine._client.messages.create = mock_create

        result = await engine.infer([{"role": "user", "content": "hi"}])
        assert result.content == "Hello, world!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5


def test_engine_stats_initial():
    engine = InferenceEngine()
    stats = engine.stats()
    assert stats["request_count"] == 0
    assert stats["avg_latency_ms"] == 0
