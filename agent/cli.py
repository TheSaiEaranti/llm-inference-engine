"""
agent/cli.py

Rich terminal CLI for the agent. Run tasks interactively or one-shot.

Usage:
    python -m agent.cli "What is 2^32?"
    python -m agent.cli  # interactive REPL mode
"""

from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text
from rich import print as rprint

from agent.loop import AgentLoop, AgentStreamEvent

app = typer.Typer()
console = Console()


async def _run_task(task: str):
    agent = AgentLoop()
    console.print(f"\n[bold blue]Task:[/bold blue] {task}\n")

    buffer = []
    async for event in agent.stream(task):
        if event.type == "tool_call":
            console.print(f"  [dim]→ {event.content}[/dim]")
        elif event.type == "tool_result":
            preview = event.content[:120].replace("\n", " ")
            console.print(f"  [dim green]✓ {preview}[/dim green]")
        elif event.type == "answer_delta":
            buffer.append(event.content)
            print(event.content, end="", flush=True)
        elif event.type == "done":
            print()  # newline after streaming
            meta = event.metadata
            console.print(
                f"\n[dim]Completed in {meta.get('iterations', 1)} step(s)[/dim]"
            )


async def _repl():
    console.print(Panel("[bold]Agentic LLM Inference Engine[/bold]\nType your task, or [dim]exit[/dim] to quit.", expand=False))
    agent = AgentLoop()

    while True:
        try:
            task = console.input("\n[bold cyan]>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if task.lower() in ("exit", "quit", "q"):
            break
        if not task:
            continue

        buffer = []
        async for event in agent.stream(task):
            if event.type == "tool_call":
                console.print(f"  [dim]→ {event.content}[/dim]")
            elif event.type == "tool_result":
                preview = event.content[:120].replace("\n", " ")
                console.print(f"  [dim green]✓ {preview}[/dim green]")
            elif event.type == "answer_delta":
                print(event.content, end="", flush=True)
            elif event.type == "done":
                print()


@app.command()
def main(task: str = typer.Argument(default=None, help="Task to run. Omit for interactive REPL.")):
    if task:
        asyncio.run(_run_task(task))
    else:
        asyncio.run(_repl())


if __name__ == "__main__":
    app()
