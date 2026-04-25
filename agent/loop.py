from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from engine.inference import InferenceEngine, InferenceConfig
from engine.telemetry import tracer
from memory.store import AgentMemory
from tools.registry import ToolRegistry, ToolResult, default_registry

AGENT_SYSTEM_PROMPT = """You are a powerful agentic assistant with access to tools.

When given a task:
1. Think step by step about what you need to do
2. Use tools when you need external information or computation
3. Synthesize results and give a clear, complete final answer

Always be precise. When using tools, use them efficiently — prefer parallel calls when possible.
After completing a task, summarize what you learned concisely."""

MAX_ITERATIONS = 10

@dataclass
class AgentStep:
    iteration: int
    thought: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    final_answer: str | None = None
    latency_ms: float = 0.0

@dataclass
class AgentRun:
    task: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    success: bool = False

    def summary(self):
        return {"task": self.task, "steps": len(self.steps), "tool_calls": sum(len(s.tool_calls) for s in self.steps), "final_answer": self.final_answer, "total_latency_ms": round(self.total_latency_ms, 2), "total_input_tokens": self.total_input_tokens, "total_output_tokens": self.total_output_tokens, "success": self.success}

@dataclass
class AgentStreamEvent:
    type: str
    content: str = ""
    metadata: dict = field(default_factory=dict)


def _extract_tool_calls(result) -> list[dict]:
    """Parse tool_use blocks from raw Anthropic response content."""
    calls = []
    for block in result.raw_content:
        if hasattr(block, "type") and block.type == "tool_use":
            calls.append({"id": block.id, "name": block.name, "input": block.input})
    return calls


def _build_tool_result_message(tool_calls: list[dict], results: list[ToolResult]) -> dict:
    """Build a user message containing all tool results."""
    content = []
    for tc, tr in zip(tool_calls, results):
        content.append({
            "type": "tool_result",
            "tool_use_id": tc["id"],
            "content": tr.to_content(),
        })
    return {"role": "user", "content": content}


def _build_assistant_message(result) -> dict:
    """Build assistant message preserving raw content blocks."""
    content = []
    for block in result.raw_content:
        if hasattr(block, "type"):
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
    return {"role": "assistant", "content": content}


class AgentLoop:
    def __init__(self, engine=None, registry=None, memory=None, max_iterations=MAX_ITERATIONS):
        self.engine = engine or InferenceEngine(InferenceConfig(system=AGENT_SYSTEM_PROMPT))
        self.registry = registry or default_registry()
        self.memory = memory or AgentMemory()
        self.max_iterations = max_iterations

    async def run(self, task: str) -> AgentRun:
        with tracer.span("agent.run") as span:
            run = AgentRun(task=task)
            start = time.monotonic()
            self.memory.add_message("user", task)
            messages = self.memory.get_conversation()
            tool_schemas = self.registry.to_anthropic_schemas()

            for iteration in range(self.max_iterations):
                step = AgentStep(iteration=iteration)
                step_start = time.monotonic()

                result = await self.engine.infer(messages=messages, tools=tool_schemas or None)
                run.total_input_tokens += result.input_tokens
                run.total_output_tokens += result.output_tokens

                tool_calls = _extract_tool_calls(result)

                if not tool_calls or result.stop_reason == "end_turn":
                    step.final_answer = result.content
                    run.final_answer = result.content
                    run.success = True
                    run.steps.append(step)
                    self.memory.add_message("assistant", result.content)
                    self.memory.memorize(f"Task: {task}\nAnswer: {result.content[:500]}")
                    break

                step.tool_calls = tool_calls
                calls_for_dispatch = [{"name": tc["name"], "inputs": tc["input"]} for tc in tool_calls]
                tool_results = await self.registry.execute_parallel(calls_for_dispatch)
                step.tool_results = tool_results

                messages.append(_build_assistant_message(result))
                messages.append(_build_tool_result_message(tool_calls, tool_results))

                step.latency_ms = (time.monotonic() - step_start) * 1000
                run.steps.append(step)
            else:
                run.final_answer = "Max iterations reached without a final answer."

            run.total_latency_ms = (time.monotonic() - start) * 1000
            span.set_attributes(run.summary())
            return run

    async def stream(self, task: str) -> AsyncIterator[AgentStreamEvent]:
        self.memory.add_message("user", task)
        messages = self.memory.get_conversation()
        tool_schemas = self.registry.to_anthropic_schemas()

        for iteration in range(self.max_iterations):
            result = await self.engine.infer(messages=messages, tools=tool_schemas or None)
            tool_calls = _extract_tool_calls(result)

            if not tool_calls or result.stop_reason == "end_turn":
                words = result.content.split(" ")
                for i, word in enumerate(words):
                    yield AgentStreamEvent(type="answer_delta", content=word + (" " if i < len(words) - 1 else ""))
                yield AgentStreamEvent(type="done", content=result.content, metadata={"iterations": iteration + 1})
                self.memory.add_message("assistant", result.content)
                break

            for tc in tool_calls:
                yield AgentStreamEvent(type="tool_call", content=f"Calling {tc['name']}", metadata={"tool": tc["name"], "input": tc["input"]})

            calls_for_dispatch = [{"name": tc["name"], "inputs": tc["input"]} for tc in tool_calls]
            tool_results = await self.registry.execute_parallel(calls_for_dispatch)

            for tr in tool_results:
                yield AgentStreamEvent(type="tool_result", content=tr.to_content()[:500], metadata={"tool": tr.tool_name, "error": tr.error})

            messages.append(_build_assistant_message(result))
            messages.append(_build_tool_result_message(tool_calls, tool_results))
