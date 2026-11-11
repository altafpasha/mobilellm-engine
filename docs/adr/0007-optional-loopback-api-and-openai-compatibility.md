# ADR-0007: Optional Loopback-Only HTTP Service & OpenAI API Compatibility

- **Status:** Accepted
- **Decider:** Agent 0 (Lead Architect) & Agent 4 (Python SDK Engineer)
- **Date:** 2026-11-11
- **Consulted:** Agent 6 (QA & Security)

---

## Context

Developers need to interact with MobileLLM Engine from Python scripts, automated benchmark suites, CLI tools, and external client frameworks. In modern AI tooling, the OpenAI REST/SSE API (`/v1/chat/completions`) has become the de facto industry standard for LLM serving.

However, exposing an unauthenticated network server on a mobile device or desktop introduces severe security and privacy risks: arbitrary apps on the same Wi-Fi network or device could submit malicious prompts, consume battery, or extract private responses.

## Decision

We implement an **optional, self-contained loopback HTTP server** supporting the OpenAI REST and Server-Sent Events (SSE) streaming wire protocol, subject to strict default security controls.

### Specification & Security Controls:
1. **Optional Component:** The HTTP server is completely optional. It is not bundled or launched for in-process Android Kotlin SDK usage (which uses direct JNI).
2. **Strict Loopback Binding by Default:**
   - The server binds exclusively to IPv4 loopback (`127.0.0.1`) or IPv6 loopback (`::1`).
   - It is never exposed to public interfaces, Wi-Fi LAN, or mobile cellular interfaces by default.
3. **Mandatory Token Authentication for LAN Exposure:**
   - Binding to `0.0.0.0` or a LAN interface requires explicit opt-in via `--allow-remote-connections` AND a non-empty `--api-key <secret>` flag.
   - All incoming requests must present a matching `Authorization: Bearer <secret>` header; unauthenticated requests are rejected with HTTP 401 Unauthorized.
4. **OpenAI Protocol Alignment:**
   - Supported Endpoints:
     - `GET  /v1/health`
     - `GET  /v1/models`
     - `POST /v1/chat/completions` (unary JSON & `text/event-stream` SSE)
     - `POST /v1/completions`
     - `GET  /v1/metrics`
     - `DELETE /v1/sessions/{id}`
   - Standard OpenAI client libraries (e.g., Python `openai`, TypeScript SDK) must function seamlessly against the local server simply by configuring `base_url="http://127.0.0.1:8080/v1"`.
5. **No Default Prompt Logging:**
   - Prompts, completions, and intermediate token logits are processed ephemerally in RAM and are never logged to console or disk in production mode.

## Consequences

### Positive:
- Instant compatibility with the vast existing ecosystem of OpenAI-compatible tools, libraries, and benchmarks.
- Clean separation between developer tooling (Python/CLI) and production in-app deployment (Kotlin/JNI).
- Robust defense against unauthorized local network access.

### Negative / Trade-offs:
- Maintenance overhead of ensuring OpenAI schema parity.

## Alternatives Considered

- **Custom Proprietary RPC / Protobuf Protocol:** Better wire efficiency, but completely incompatible with existing developer tooling and OpenAI client libraries, increasing developer onboarding friction.
- **Always-On Android Background HTTP Server:** Rejected in ADR-0004 due to battery consumption, sandbox socket permissions, and local IPC attack surfaces.
