# Changelog

All notable changes to the `mobilellm` Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-12

### Added
- **Synchronous & Asynchronous Engine API**: `Engine` and `AsyncEngine` with clean context management (`with Engine(...)`, `async with Engine.async_client(...)`).
- **Model & Session Abstractions**: `Model`, `AsyncModel`, `Session`, `AsyncSession` supporting unary text generation and token streaming.
- **Pluggable Transports**:
  - `NativeTransport` & `AsyncNativeTransport`: Direct zero-network in-process C-ABI FFI linking `libmobilellm` via `ctypes`.
  - `HTTPTransport` & `AsyncHTTPTransport`: Local REST and SSE streaming client built on `httpx`.
  - `MockTransport` & `AsyncMockTransport`: In-memory isolated transports for unit testing.
- **Strict Loopback Server**: Local HTTP daemon implementing OpenAI `/v1/chat/completions` (unary and SSE) and `/v1/models`, enforcing ADR-0007 security assertions (strictly refuses public IP binding without `--allow-remote-connections` and `--api-key`).
- **CLI (`mobilellm` / `mobilellm-cli`)**:
  - `mobilellm status`: Queries engine state and hardware capabilities.
  - `mobilellm models`: Lists loaded models and metadata.
  - `mobilellm load <path>`: Loads a GGUF model into runtime memory.
  - `mobilellm unload <id>`: Unloads model weights.
  - `mobilellm generate <prompt>`: Text generation with streaming support.
  - `mobilellm benchmark`: Benchmark latency, TTFT, and throughput over multiple runs.
  - `mobilellm serve`: Starts the local OpenAI-compatible loopback daemon.
- **OpenAI Interoperability Client**: `AsyncMobileLLMClient` with `chat.completions.create` and `models.list`.
- **Packaging & Typing**: Modern `pyproject.toml` packaging with `hatchling`, PEP 561 `py.typed` marker, and 100% strict `mypy` typing compliance.
- **Test Suite**: 32 comprehensive tests covering models, exceptions, sync/async clients, cancellation, timeouts, server security, CLI, and real native C-ABI integration with GGUF models.

