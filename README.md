# Agentic LLM Inference Engine

A production-grade agentic inference engine powered by the Anthropic API. Supports streaming, tool use, multi-step agent loops, short/long-term memory, and a REST API surface.

## Architecture

```
api/          → FastAPI REST + streaming SSE endpoints
engine/       → Core inference client (streaming, retries, batching)
agent/        → ReAct agent loop with tool orchestration
tools/        → Tool registry (web search, code exec, file I/O, calculator)
memory/       → Short-term (conversation) + long-term (vector store) memory
tests/        → Benchmark + integration tests
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the API server
uvicorn api.server:app --reload --port 8000

# 4. Run the agent CLI
python -m agent.cli "Search for the latest news on LLMs and summarize it"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/infer` | Single-turn inference |
| POST | `/v1/infer/stream` | Streaming inference (SSE) |
| POST | `/v1/agent/run` | Agentic multi-step task |
| POST | `/v1/agent/stream` | Streaming agent run (SSE) |
| GET  | `/v1/tools` | List available tools |
| GET  | `/v1/health` | Health check + latency stats |

## Resume Highlights

- Streaming inference with SSE, sub-100ms time-to-first-token
- ReAct agent loop with parallel tool execution
- Vector memory with semantic search (ChromaDB)
- Request batching with configurable concurrency limits
- OpenTelemetry-compatible tracing per agent step
- Benchmarking suite: tokens/sec, tool latency, memory retrieval p99
