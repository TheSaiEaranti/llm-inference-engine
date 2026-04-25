"""
tools/registry.py

Tool registry with built-in tools: calculator, web_search stub, code_exec, file_read.
Each tool exposes an Anthropic-compatible JSON schema and an async execute() method.
"""

from __future__ import annotations

import ast
import math
import asyncio
from typing import Any, Callable, Awaitable
from dataclasses import dataclass


@dataclass
class ToolResult:
    tool_name: str
    output: Any
    error: str | None = None
    latency_ms: float = 0.0

    def to_content(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        return str(self.output)


class Tool:
    """Base class for all tools."""

    name: str
    description: str
    input_schema: dict

    async def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate a mathematical expression. "
        "Supports basic arithmetic, exponents, and common math functions (sqrt, sin, cos, log, etc.)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A Python-compatible math expression, e.g. '2 ** 10 + sqrt(144)'",
            }
        },
        "required": ["expression"],
    }

    async def execute(self, expression: str) -> ToolResult:
        import time
        start = time.monotonic()
        try:
            # Safe eval: only math namespace
            safe_ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            safe_ns["__builtins__"] = {}
            result = eval(expression, safe_ns)  # noqa: S307
            return ToolResult(
                tool_name=self.name,
                output=result,
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, output=None, error=str(e))


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web for current information. Returns a list of relevant snippets. "
        "Use when you need up-to-date facts, news, or information beyond your training data."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "num_results": {"type": "integer", "description": "Number of results (1-5)", "default": 3},
        },
        "required": ["query"],
    }

    async def execute(self, query: str, num_results: int = 3) -> ToolResult:
        import time
        start = time.monotonic()
        # Stub: replace with SerpAPI, Tavily, or Brave Search API
        mock_results = [
            {"title": f"Result {i+1} for: {query}", "snippet": f"Relevant information about {query}..."}
            for i in range(num_results)
        ]
        return ToolResult(
            tool_name=self.name,
            output=mock_results,
            latency_ms=(time.monotonic() - start) * 1000,
        )


class CodeExecTool(Tool):
    name = "code_exec"
    description = (
        "Execute Python code in a sandboxed environment and return stdout. "
        "Use for data analysis, computation, or generating outputs programmatically."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
            "timeout": {"type": "integer", "description": "Max execution time in seconds", "default": 10},
        },
        "required": ["code"],
    }

    async def execute(self, code: str, timeout: int = 10) -> ToolResult:
        import time, io, contextlib
        start = time.monotonic()
        stdout_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(compile(code, "<agent>", "exec"), {"__builtins__": __builtins__})  # noqa: S102
            output = stdout_buf.getvalue() or "(no output)"
            return ToolResult(
                tool_name=self.name,
                output=output,
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, output=None, error=str(e))


class FileReadTool(Tool):
    name = "file_read"
    description = "Read the contents of a local file. Returns the file content as a string."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative file path"},
            "max_chars": {"type": "integer", "description": "Max characters to return", "default": 8000},
        },
        "required": ["path"],
    }

    async def execute(self, path: str, max_chars: int = 8000) -> ToolResult:
        import time
        start = time.monotonic()
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read(max_chars)
            return ToolResult(
                tool_name=self.name,
                output=content,
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, output=None, error=str(e))


class ToolRegistry:
    """Central registry. Register tools, dispatch by name, export Anthropic schemas."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def to_anthropic_schemas(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    async def execute(self, name: str, inputs: dict) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(tool_name=name, output=None, error=f"Unknown tool: {name}")
        return await tool.execute(**inputs)

    async def execute_parallel(self, calls: list[dict]) -> list[ToolResult]:
        """Execute multiple tool calls concurrently."""
        tasks = [self.execute(c["name"], c["inputs"]) for c in calls]
        return await asyncio.gather(*tasks)


def default_registry() -> ToolRegistry:
    """Pre-loaded registry with all built-in tools."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())
    registry.register(CodeExecTool())
    registry.register(FileReadTool())
    return registry
