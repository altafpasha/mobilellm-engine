# Agent 4 Specification — Python SDK & Tooling Engineer

> **Module Ownership:** `python/`, `server/`, `tools/developer-cli/`  
> **Direct Reports to:** Agent 0 (Lead Architect)  
> **Key Downstream Dependents:** Python Developers, Agent 6 (Automated Benchmarking)

---

## 1. Role Purpose & Domain Ownership

Agent 4 is responsible for building high-productivity developer interfaces, the asynchronous Python SDK (`mobilellm`), the command-line interface (`mobilellm-cli`), the optional local loopback HTTP server (supporting the OpenAI REST and SSE streaming specification), and desktop test bindings for cross-platform model verification.

### Complete Directory Ownership
```
python/
├── pyproject.toml             # Build & packaging specification (hatchling)
├── mobilellm/                 # Public Python SDK package
│   ├── __init__.py
│   ├── client.py              # Async HTTP/SSE client (httpx)
│   ├── models.py              # Pydantic data schemas (requests/responses)
│   └── exceptions.py          # MobileLLM typed exceptions
├── cli/                       # Terminal CLI tool
│   ├── __init__.py
│   ├── main.py                # CLI entry point (argparse / click / typer)
│   ├── commands/              # run, bench, info, validate commands
│   └── formatters.py          # Rich terminal rendering for tokens & metrics
├── bindings/                  # Optional native desktop FFI (ctypes / pybind11)
│   ├── native_client.py       # Direct C-ABI ctypes wrapper for local testing
│   └── CMakeLists.txt
└── tests/                     # Pytest suite with strict typing
server/
├── http/                      # Loopback HTTP server (FastAPI / aiohttp)
│   ├── app.py                 # Core server factory & route registry
│   └── routes.py              # /v1/models, /v1/health, /v1/chat/completions
├── openai_compat/             # OpenAI schema mapping & SSE stream formatter
├── streaming/                 # Server-Sent Events (SSE) streaming engine
└── security/                  # Loopback assertion & Bearer token validator
tools/developer-cli/           # Standalone developer utility scripts
```

---

## 2. Technical Standards & Constraints

1. **Python Environment & Packaging:**
   - Python target: Python 3.11+.
   - Modern packaging standard: `pyproject.toml` with `hatchling` build-backend.
   - Strict static typing: 100% type annotations validated via `mypy --strict` or `pyright`.
2. **Async Client Design:**
   - Asynchronous by default using `httpx.AsyncClient` or native `asyncio`.
   - Streaming API yields asynchronous iterators: `async for chunk in client.chat.create(...)`.
   - Clean disconnection and cancellation: dropping the async generator sends an abort signal to the server.
3. **Loopback Server Security Standards:**
   - **Localhost by Default:** The HTTP server binds strictly to `127.0.0.1`.
   - **Opt-in Network Access:** Binding to `0.0.0.0` or a public network interface requires explicit argument `--allow-remote-connections` combined with an `--api-key <secret>` flag.
   - **OpenAI Compatibility:** Must support `/v1/chat/completions` and `/v1/models` adhering to standard OpenAI wire format, allowing existing OpenAI client libraries to work out of the box.
4. **Desktop Native Verification:**
   - Provide an optional `ctypes` or `pybind11` bridge directly to `libmobilellm.so` / `libmobilellm.dylib` for developers testing models on macOS/Linux workstations without launching an Android emulator.

---

## 3. Initial Implementation Task Backlog

| Task ID | Task Description | Dependencies | Deliverable | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **T4.1** | Python Packaging & CI Setup | Agent 0 Contract | `python/pyproject.toml` | `pip install -e .` succeeds; `pytest` and `mypy` configured |
| **T4.2** | Pydantic Request/Response Schemas| None | `python/mobilellm/models.py` | Strict validation matching OpenAI API schema |
| **T4.3** | Async MobileLLM Client | T4.2 | `python/mobilellm/client.py` | Streams tokens via SSE using `httpx.AsyncClient` |
| **T4.4** | Terminal CLI Tool (`mobilellm-cli`)| T4.3 | `python/cli/` | `mobilellm run` streams model output in terminal |
| **T4.5** | Optional Loopback HTTP Server | Agent 1 (`c_api.h`) | `server/http/` | Local server serves `/v1/chat/completions` with SSE |
| **T4.6** | Security Hardening (Loopback/Auth)| T4.5 | `server/security/` | Fails to start with public IP unless API key is supplied |
| **T4.7** | OpenAI SDK Interoperability Test | T4.5 | `python/tests/test_openai.py`| Standard `openai` Python package works against local server |

---

## 4. Key Cross-Module Interfaces

### Inputs Required from Other Modules:
- **From Agent 1 (Runtime):** Stable C-ABI definitions (`c_api.h`) for desktop native bindings.
- **From Agent 6 (QA & Security):** Test models and attack vectors for API schema validation.

### Outputs Provided to Other Modules:
- **To End-User Developers:** Published `mobilellm` PyPI package and `mobilellm-cli` executable.
- **To Agent 6 (QA & Benchmarking):** Automation client for running standardized latency/throughput test suites.
