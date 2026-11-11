# Architecture Decision Records (ADRs)

> **Purpose:** Document significant architectural decisions, their context, consequences, and alternatives considered.  
> **Standard:** Michael Nygard format (Context, Decision, Consequences, Alternatives Considered, Status).

---

## Decision Log

| ADR ID | Title | Date | Status | Primary Decider |
| :--- | :--- | :--- | :--- | :--- |
| [**ADR-0001**](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/adr/0001-core-language-and-cpp20-standards.md) | Adoption of C++20 for Native Engine & Core Runtime | 2026-11-11 | **Accepted** | Agent 0 (Architect) |
| [**ADR-0002**](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/adr/0002-initial-inference-backend-llamacpp.md) | Initial Inference Engine Foundation via llama.cpp / GGML | 2026-11-11 | **Accepted** | Agent 0 (Architect) |
| [**ADR-0003**](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/adr/0003-gguf-format-and-model-support-strategy.md) | GGUF as Initial Model Format & Model Matrix Strategy | 2026-11-11 | **Accepted** | Agent 0 (Architect) |
| [**ADR-0004**](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/adr/0004-android-jni-bridge-and-kotlin-sdk-architecture.md) | Native NDK / JNI Bridge Architecture & Kotlin SDK Design | 2026-11-11 | **Accepted** | Agent 0 & Agent 3 |
| [**ADR-0005**](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/adr/0005-hardware-backend-abstraction-and-fallback-policy.md) | Hardware Abstraction Layer & Automatic CPU Fallback Policy | 2026-11-11 | **Accepted** | Agent 0 & Agent 5 |
| [**ADR-0006**](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/adr/0006-mobile-memory-budgeting-and-paged-kv-cache.md) | Mobile Memory Budgeting, Paged KV-Cache & Zero SQLite Tensors | 2026-11-11 | **Accepted** | Agent 0 & Agent 2 |
| [**ADR-0007**](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/adr/0007-optional-loopback-api-and-openai-compatibility.md) | Optional Loopback-Only HTTP Service & OpenAI API Compatibility | 2026-11-11 | **Accepted** | Agent 0 & Agent 4 |
